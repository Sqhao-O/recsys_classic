"""用小数据检查序列推荐最容易出错的时间边界、采样和 Mask。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.din import DIN, RetailRocketDataset
from src.preprocess_retailrocket import load_category_properties, preprocess_events, sample_negative


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        rows = [(1, user, "view", item) for user, item in enumerate([10, 20, 30, 40, 50], 1)]
        for timestamp in range(2, 11):
            rows.extend([(timestamp, 1, "view", [10, 20, 30, 40][(timestamp - 1) % 4]),
                         (timestamp, 2, "view", [20, 30, 10, 40][(timestamp - 1) % 4])])
        rows.extend([(2, 1, "view", 30), (9, 9, "view", 99), (2, 1, "addtocart", 99)])
        self.events = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])
        self.events = pd.concat([self.events, self.events.iloc[:1]], ignore_index=True)  # 故意加入重复日志
        self.properties = pd.DataFrame([
            (0, 10, "categoryid", "100"), (0, 20, "categoryid", "100"),
            (0, 30, "categoryid", "200"), (0, 40, "categoryid", "300"),
            (3, 20, "categoryid", "200"),  # 训练期改变类别，旧历史应保留浏览当时的类别
            (8, 20, "categoryid", "999"),  # 验证期新类别应编码为未知 1
            (12, 50, "categoryid", "777"),  # 未来才有的属性，不能回填过去
            (0, 10, "available", "1"),
        ], columns=["timestamp", "itemid", "property", "value"])
        self.day = 24 * 60 * 60 * 1000
        self.base = int(pd.Timestamp("2020-01-01", tz="Asia/Shanghai").timestamp() * 1000)
        self.events["timestamp"] = self.base + self.events["timestamp"] * self.day
        self.properties["timestamp"] = self.base + self.properties["timestamp"] * self.day

    def at(self, day):
        return self.base + day * self.day

    def preprocess(self, events=None, properties=None, **kwargs):
        # 测试数据从 2020 年开始，显式指定窗口起点。
        options = dict(max_history_len=2, val_ratio=0.4,
                       sample_start_date="2020-01-03")
        options.update(kwargs)
        properties = self.properties if properties is None else properties
        categories = properties.loc[properties["property"] == "categoryid", ["timestamp", "itemid", "value"]]
        return preprocess_events(self.events if events is None else events,
                                 categories, **options)

    def test_load_category_properties(self):
        properties = self.properties.copy()
        properties.loc[properties["property"] == "available", "value"] = "true"
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder)
            properties.iloc[:4].to_csv(data_dir / "item_properties_part1.csv", index=False)
            properties.iloc[4:].to_csv(data_dir / "item_properties_part2.csv", index=False)
            categories = load_category_properties(data_dir, chunksize=2)
        expected = properties.loc[properties["property"] == "categoryid", ["timestamp", "itemid", "value"]]
        pd.testing.assert_frame_equal(categories, expected.reset_index(drop=True))
        actual, metadata, items, category_mapping = preprocess_events(
            self.events, categories, max_history_len=2, val_ratio=0.4, sample_start_date="2020-01-03")
        expected_data, expected_metadata, expected_items, expected_categories = self.preprocess()
        self.assertEqual((metadata, items, category_mapping), (expected_metadata, expected_items, expected_categories))
        for split in actual:
            for key in actual[split]:
                np.testing.assert_array_equal(actual[split][key], expected_data[split][key])

    def test_histories_negatives_and_time_split(self):
        shuffled = self.events.sample(frac=1, random_state=42)
        data, metadata, mapping, category_mapping = self.preprocess(
            shuffled, self.properties.sample(frac=1, random_state=42))
        self.assertNotIn(99, mapping)  # 验证期新商品和加购事件都不能扩大训练词表
        self.assertEqual(set(category_mapping), {100, 200, 300})
        self.assertEqual(metadata["category_vocab_size"], 5)
        raw_items = {encoded: raw for raw, encoded in mapping.items()}

        def category_at(item, timestamp):
            past = self.properties[(self.properties["itemid"] == item)
                                   & (self.properties["property"] == "categoryid")
                                   & (self.properties["timestamp"] <= timestamp)]
            return category_mapping.get(int(past.iloc[-1]["value"]), 1) if not past.empty else 1

        views = shuffled[shuffled["event"] == "view"].drop_duplicates().sort_values("timestamp", kind="stable")
        saw_rolling_history = False
        saw_simultaneous_clicks = False
        for split, arrays in data.items():
            np.testing.assert_array_equal(arrays["labels"][::2], 1)
            np.testing.assert_array_equal(arrays["labels"][1::2], 0)
            np.testing.assert_array_equal(arrays["history_ids"][::2], arrays["history_ids"][1::2])
            np.testing.assert_array_equal(arrays["history_category_ids"][::2], arrays["history_category_ids"][1::2])
            for i in range(0, len(arrays["labels"]), 2):
                user, timestamp = arrays["user_ids"][i], arrays["timestamps"][i]
                self.assertEqual(timestamp < metadata["split_timestamp"], split == "train")
                past = views[(views["visitorid"] == user) & (views["timestamp"] < timestamp)]
                known_past = past[past["itemid"].isin(mapping)]
                expected = [mapping[item] for item in known_past["itemid"].tail(2)]
                expected += [0] * (2 - len(expected))
                np.testing.assert_array_equal(arrays["history_ids"][i], expected)
                expected_categories = [category_at(row.itemid, row.timestamp)
                                       for row in known_past.tail(2).itertuples()]
                expected_categories += [0] * (2 - len(expected_categories))
                np.testing.assert_array_equal(arrays["history_category_ids"][i], expected_categories)
                for j in (i, i + 1):  # 正负候选分别查询预测时刻的类别
                    item = raw_items[arrays["target_ids"][j]]
                    self.assertEqual(arrays["target_category_ids"][j], category_at(item, timestamp))
                current = views[(views["visitorid"] == user) & (views["timestamp"] == timestamp)]
                excluded = {mapping[item] for item in pd.concat([past, current])["itemid"] if item in mapping}
                negative = arrays["target_ids"][i + 1]
                self.assertNotIn(negative, excluded)
                appeared = {mapping[item] for item in views[views["timestamp"] < timestamp]["itemid"] if item in mapping}
                self.assertIn(negative, appeared)  # 不能采到当前或未来首次出现的商品
                saw_rolling_history |= split == "val" and bool((past["timestamp"] >= metadata["split_timestamp"]).any())
                saw_simultaneous_clicks |= timestamp == self.at(2) and user == 1
        self.assertTrue(saw_rolling_history)
        self.assertTrue(saw_simultaneous_clicks)
        self.assertEqual(metadata["skipped_view_count"]["unknown_target"], 1)
        self.assertGreater(metadata["category_stats"]["val"]["positive"]["outside_train_vocab_count"], 0)

    def test_all_samples_and_determinism(self):
        first, metadata, _, _ = self.preprocess()
        second, _, _, _ = self.preprocess()
        for split in first:
            for key in first[split]:
                np.testing.assert_array_equal(first[split][key], second[split][key])
            self.assertEqual(metadata[f"{split}_sample_count"], len(first[split]["labels"]))
            self.assertTrue((np.diff(first[split]["timestamps"]) >= 0).all())
        self.assertEqual(metadata["view_count"], 25)
        self.assertEqual(metadata["train_sample_count"], 22)
        self.assertEqual(metadata["val_sample_count"], 16)
        pair_count = sum(len(arrays["labels"]) // 2 for arrays in first.values())
        self.assertEqual(pair_count + sum(metadata["skipped_view_count"].values()), metadata["window_view_count"])

    def test_window_and_warmup_history(self):
        data, metadata, mapping, _ = self.preprocess(sample_start_date="2020-01-06")
        self.assertEqual(metadata["sample_start_timestamp"], self.at(5))
        self.assertEqual(metadata["warmup_view_count"], 12)
        train = data["train"]
        self.assertEqual(train["timestamps"][0], self.at(5))
        # 窗口开始时，用户 1 已有窗口前第 3、4 天浏览的商品 30、40。
        np.testing.assert_array_equal(train["history_ids"][0], [mapping[30], mapping[40]])
        for arrays in data.values():
            self.assertTrue((arrays["timestamps"] >= self.at(5)).all())

    def test_missing_day_does_not_change_fixed_start(self):
        events = self.events.loc[self.events["timestamp"] != self.at(3)]
        _, metadata, _, _ = self.preprocess(events, sample_start_date="2020-01-06")
        self.assertEqual(metadata["sample_start_timestamp"], self.at(5))

    def test_validation_keeps_missing_categories_and_new_users(self):
        extra = pd.DataFrame([(self.at(10), 1, "view", 50),
                              (self.at(6), 100, "view", 10),
                              (self.at(1), 200, "view", 60)], columns=self.events.columns)
        properties = pd.concat([self.properties, pd.DataFrame([
            (self.at(0), 60, "categoryid", "100")
        ], columns=self.properties.columns)], ignore_index=True)  # 多留一个可抽的负商品
        data, metadata, mapping, _ = self.preprocess(
            pd.concat([self.events, extra], ignore_index=True), properties)
        val = data["val"]
        missing = (val["target_ids"] == mapping[50]) & (val["labels"].ravel() == 1)
        self.assertTrue(missing.any())
        self.assertTrue((val["target_category_ids"][missing] == 1).all())
        self.assertGreater(metadata["category_stats"]["val"]["positive"]["no_past_record_count"], 0)
        self.assertEqual(metadata["skipped_view_count"]["empty_history"], 1)

    def test_missing_and_future_categories_do_not_change_fixed_start(self):
        future = self.properties.copy()
        future["timestamp"] += 20 * self.day
        for properties in (self.properties.iloc[:0], future):
            data, metadata, _, category_mapping = self.preprocess(properties=properties)
            self.assertEqual(metadata["sample_start_timestamp"], self.at(2))
            self.assertEqual(category_mapping, {})
            for arrays in data.values():
                self.assertTrue((arrays["target_category_ids"] == 1).all())
                real_history = arrays["history_ids"] != 0
                self.assertTrue((arrays["history_category_ids"][real_history] == 1).all())

    def test_start_after_data(self):
        with self.assertRaisesRegex(ValueError, "指定起点"):
            self.preprocess(sample_start_date="2020-02-01")

    def test_no_negative_and_invalid_data(self):
        self.assertIsNone(sample_negative([1, 2], {1, 2}, np.random.default_rng(42)))
        self.assertIsNone(sample_negative([], set(), np.random.default_rng(42)))
        for kwargs in [{"max_history_len": 0}, {"val_ratio": 1}]:
            with self.assertRaises(ValueError):
                self.preprocess(**kwargs)
        with self.assertRaises(ValueError):
            self.preprocess(self.events.iloc[:1])

    def test_saved_dataset(self):
        data, _, _, _ = self.preprocess()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train_common.npz"
            np.savez_compressed(path, **data["train"])
            dataset = RetailRocketDataset(path)
            target, target_category, history, history_category, label = dataset[0]
            self.assertEqual(len(dataset), len(data["train"]["labels"]))
            self.assertEqual(target.dtype, torch.long)
            self.assertEqual(history.dtype, torch.long)
            self.assertEqual(target_category.dtype, torch.long)
            self.assertEqual(history_category.dtype, torch.long)
            self.assertEqual(history_category.shape, history.shape)
            torch.testing.assert_close(history_category, torch.from_numpy(data["train"]["history_category_ids"][0]).long())
            model = DIN(item_vocab_size=6, category_vocab_size=5, embedding_dim=4)
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
            self.assertEqual(model(*batch[:-1]).shape, (2, 1))
            self.assertEqual(label.dtype, torch.float32)
            self.assertEqual(label.shape, (1,))


class DINTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 设备")
    def test_cuda_training_and_cpu_evaluation(self):
        model = DIN(item_vocab_size=8, category_vocab_size=6, embedding_dim=4).to("cuda")
        target = torch.tensor([1, 2], device="cuda")
        category = torch.tensor([2, 3], device="cuda")
        history = torch.tensor([[3, 4], [0, 0]], device="cuda")
        history_category = torch.tensor([[2, 4], [0, 0]], device="cuda")
        labels = torch.tensor([[1.0], [0.0]], device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logits = model(target, category, history, history_category)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels) + 1e-4 * model.embedding_l2()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            values = model(target, category, history, history_category).cpu().numpy()
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(np.isfinite(values).all())

    def test_embedding_regularization_excludes_padding(self):
        model = DIN(item_vocab_size=4, category_vocab_size=3, embedding_dim=2)
        with torch.no_grad():
            model.item_embedding.weight.fill_(1)
            model.category_embedding.weight.fill_(1)
            model.item_embedding.weight[0].fill_(100)  # 补齐行即使非零，也不能计入正则
            model.category_embedding.weight[0].fill_(100)
        penalty = model.embedding_l2()
        self.assertEqual(penalty.item(), 10)  # (3 个商品 + 2 个类别) × 每个 2 维
        penalty.backward()
        for embedding in (model.item_embedding, model.category_embedding):
            torch.testing.assert_close(embedding.weight.grad[0], torch.zeros(2))
            torch.testing.assert_close(embedding.weight.grad[1:], torch.full_like(embedding.weight[1:], 2))
        self.assertIsNone(model.attention[0].weight.grad)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        self.assertLess(model.embedding_l2().item(), 10)  # 正则梯度确实让有效向量缩小

    def test_padding_empty_history_and_backward(self):
        torch.manual_seed(42)
        model = DIN(item_vocab_size=8, category_vocab_size=6, embedding_dim=4)
        target = torch.tensor([1, 2])
        target_category = torch.tensor([2, 3])
        history = torch.tensor([[3, 4], [0, 0]])
        history_category = torch.tensor([[1, 4], [0, 0]])  # 未知类别 1 的行为也应参与注意力
        output = model(target, target_category, history, history_category)
        extended = torch.cat([history, torch.zeros(2, 3, dtype=torch.long)], dim=1)
        extended_category = torch.cat([history_category, torch.zeros(2, 3, dtype=torch.long)], dim=1)
        torch.testing.assert_close(output, model(target, target_category, extended, extended_category))  # 多补几个 0 不应改变结果
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(torch.isfinite(output).all())  # 包含全空历史也不能出现 NaN
        interest = model.get_interest(model.embed_items(target, target_category),
                                      model.embed_items(history, history_category), history != 0)
        torch.testing.assert_close(interest[1], torch.zeros(8))
        loss = torch.nn.BCEWithLogitsLoss()(output, torch.tensor([[1.0], [0.0]]))
        loss.backward()
        torch.testing.assert_close(model.item_embedding.weight.grad[0], torch.zeros(4))
        self.assertGreater(model.item_embedding.weight.grad[3:5].abs().sum().item(), 0)
        torch.testing.assert_close(model.category_embedding.weight.grad[0], torch.zeros(4))
        self.assertGreater(model.category_embedding.weight.grad[1].abs().sum().item(), 0)
        self.assertGreater(model.category_embedding.weight.grad[4].abs().sum().item(), 0)
        self.assertGreater(model.attention[0].weight.grad.abs().sum().item(), 0)

    def test_masked_values_and_candidate_dependent_interest(self):
        torch.manual_seed(42)
        model = DIN(item_vocab_size=8, category_vocab_size=6, embedding_dim=4)
        history = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, True, False]])
        target = torch.randn(1, 8)
        interest = model.get_interest(target, history, mask)
        modified = history.clone()
        modified[:, 2] = 1000  # 即使遮住的位置有非零向量，也必须被 Mask 排除
        torch.testing.assert_close(interest, model.get_interest(target, modified, mask))
        other_interest = model.get_interest(target + 1, history, mask)
        self.assertGreater((interest - other_interest).abs().sum().item(), 0)

    def test_attention_uses_item_and_category_features(self):
        torch.manual_seed(42)
        model = DIN(item_vocab_size=8, category_vocab_size=6, embedding_dim=4)
        target = model.embed_items(torch.tensor([1]), torch.tensor([2]))
        history = model.embed_items(torch.tensor([[3, 4]]), torch.tensor([[2, 3]]))
        mask = torch.tensor([[True, True]])
        # 检查注意力权重对候选、历史的商品部分和类别部分都有梯度。
        weights = model.attention(torch.cat([target.unsqueeze(1).expand_as(history), history,
                                            target.unsqueeze(1) - history, target.unsqueeze(1) * history], dim=-1))
        target_grad, history_grad = torch.autograd.grad(weights.sum(), (target, history))
        for grad in (target_grad, history_grad):
            self.assertGreater(grad[..., :4].abs().sum().item(), 0)
            self.assertGreater(grad[..., 4:].abs().sum().item(), 0)
        changed = model.embed_items(torch.tensor([[3, 4]]), torch.tensor([[4, 3]]))
        self.assertGreater((model.get_interest(target, history, mask)
                            - model.get_interest(target, changed, mask)).abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
