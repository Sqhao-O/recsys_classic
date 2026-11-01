"""用小数据检查序列推荐最容易出错的时间边界、采样和 Mask。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.din import DIN, RetailRocketDataset
from src.preprocess_retailrocket import preprocess_events, sample_negative


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        rows = [(1, user, "view", item) for user, item in enumerate([10, 20, 30, 40, 50], 1)]
        for timestamp in range(2, 11):
            rows.extend([(timestamp, 1, "view", [10, 20, 30, 40][(timestamp - 1) % 4]),
                         (timestamp, 2, "view", [20, 30, 10, 40][(timestamp - 1) % 4])])
        rows.extend([(2, 1, "view", 30), (9, 9, "view", 99), (2, 1, "addtocart", 99)])
        self.events = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])
        self.events = pd.concat([self.events, self.events.iloc[:1]], ignore_index=True)  # 故意加入重复日志

    def test_histories_negatives_and_time_split(self):
        shuffled = self.events.sample(frac=1, random_state=42)
        data, metadata, mapping = preprocess_events(shuffled,
                                                  max_events=None, max_history_len=2, val_ratio=0.3)
        self.assertNotIn(99, mapping)  # 验证期新商品和加购事件都不能扩大训练词表
        views = shuffled[shuffled["event"] == "view"].drop_duplicates().sort_values("timestamp", kind="stable")
        saw_rolling_history = False
        saw_simultaneous_clicks = False
        for split, arrays in data.items():
            np.testing.assert_array_equal(arrays["labels"][::2], 1)
            np.testing.assert_array_equal(arrays["labels"][1::2], 0)
            np.testing.assert_array_equal(arrays["history_ids"][::2], arrays["history_ids"][1::2])
            for i in range(0, len(arrays["labels"]), 2):
                user, timestamp = arrays["user_ids"][i], arrays["timestamps"][i]
                self.assertEqual(timestamp < metadata["split_timestamp"], split == "train")
                past = views[(views["visitorid"] == user) & (views["timestamp"] < timestamp)]
                known_past = past[past["itemid"].isin(mapping)]
                expected = [mapping[item] for item in known_past["itemid"].tail(2)]
                expected += [0] * (2 - len(expected))
                np.testing.assert_array_equal(arrays["history_ids"][i], expected)
                current = views[(views["visitorid"] == user) & (views["timestamp"] == timestamp)]
                excluded = {mapping[item] for item in pd.concat([past, current])["itemid"] if item in mapping}
                negative = arrays["target_ids"][i + 1]
                self.assertNotIn(negative, excluded)
                appeared = {mapping[item] for item in views[views["timestamp"] < timestamp]["itemid"] if item in mapping}
                self.assertIn(negative, appeared)  # 不能采到当前或未来首次出现的商品
                saw_rolling_history |= split == "val" and bool((past["timestamp"] >= metadata["split_timestamp"]).any())
                saw_simultaneous_clicks |= timestamp == 2 and user == 1
        self.assertTrue(saw_rolling_history)
        self.assertTrue(saw_simultaneous_clicks)
        self.assertEqual(metadata["skipped_view_count"]["unknown_target"], 1)

    def test_determinism_and_timestamp_limit(self):
        first, metadata, _ = preprocess_events(self.events, max_events=None)
        second, _, _ = preprocess_events(self.events, max_events=None)
        for split in first:
            for key in first[split]:
                np.testing.assert_array_equal(first[split][key], second[split][key])
        self.assertEqual(metadata["selected_view_count"], 25)
        _, limited, _ = preprocess_events(self.events, max_events=11)
        self.assertEqual(limited["selected_view_count"], 12)  # 第 11 条落在 t=4，同毫秒的两条必须一起保留

    def test_no_negative_and_invalid_data(self):
        self.assertIsNone(sample_negative([1, 2], {1, 2}, np.random.default_rng(42)))
        self.assertIsNone(sample_negative([], set(), np.random.default_rng(42)))
        for kwargs in [{"max_history_len": 0}, {"val_ratio": 1}, {"max_events": 1}]:
            with self.assertRaises(ValueError):
                preprocess_events(self.events, **kwargs)
        with self.assertRaises(ValueError):
            preprocess_events(self.events.iloc[:1])

    def test_saved_dataset(self):
        data, _, _ = preprocess_events(self.events, max_events=None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train_common.npz"
            np.savez_compressed(path, **data["train"])
            dataset = RetailRocketDataset(path)
            target, history, label = dataset[0]
            self.assertEqual(len(dataset), len(data["train"]["labels"]))
            self.assertEqual(target.dtype, torch.long)
            self.assertEqual(history.dtype, torch.long)
            self.assertEqual(label.dtype, torch.float32)
            self.assertEqual(label.shape, (1,))


class DINTest(unittest.TestCase):
    def test_padding_empty_history_and_backward(self):
        torch.manual_seed(42)
        model = DIN(item_vocab_size=8, embedding_dim=4)
        target = torch.tensor([1, 2])
        history = torch.tensor([[3, 4], [0, 0]])
        output = model(target, history)
        extended = torch.cat([history, torch.zeros(2, 3, dtype=torch.long)], dim=1)
        torch.testing.assert_close(output, model(target, extended))  # 多补几个 0 不应改变结果
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(torch.isfinite(output).all())  # 包含全空历史也不能出现 NaN
        interest = model.get_interest(model.item_embedding(target), model.item_embedding(history), history != 0)
        torch.testing.assert_close(interest[1], torch.zeros(4))
        loss = torch.nn.BCEWithLogitsLoss()(output, torch.tensor([[1.0], [0.0]]))
        loss.backward()
        torch.testing.assert_close(model.item_embedding.weight.grad[0], torch.zeros(4))
        self.assertGreater(model.item_embedding.weight.grad[3:5].abs().sum().item(), 0)
        self.assertGreater(model.attention[0].weight.grad.abs().sum().item(), 0)

    def test_masked_values_and_candidate_dependent_interest(self):
        torch.manual_seed(42)
        model = DIN(item_vocab_size=8, embedding_dim=4)
        history = torch.randn(1, 3, 4)
        mask = torch.tensor([[True, True, False]])
        target = torch.randn(1, 4)
        interest = model.get_interest(target, history, mask)
        modified = history.clone()
        modified[:, 2] = 1000  # 即使遮住的位置有非零向量，也必须被 Mask 排除
        torch.testing.assert_close(interest, model.get_interest(target, modified, mask))
        other_interest = model.get_interest(target + 1, history, mask)
        self.assertGreater((interest - other_interest).abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
