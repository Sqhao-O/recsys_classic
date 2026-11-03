import json  # 保存词表和数据配置，供 DIN 读取
from collections import defaultdict, deque  # defaultdict 自动创建用户记录；deque 只保留最近的行为
from itertools import groupby  # 将时间相同的事件放在一起处理
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 1. 设置数据路径 ====================

base_dir = Path(__file__).resolve().parent
filepath = base_dir / "datasets/RetailRocket/events.csv"  # 复用项目中已有的原始数据
save_dir = base_dir / "datasets/RetailRocket/processed"

# 分块读取两份属性文件
def load_category_properties(data_dir, chunksize=200000):
    category_parts = []
    for part in (1, 2):
        for chunk in pd.read_csv(data_dir / f"item_properties_part{part}.csv",
                                 chunksize=chunksize, # 传入chunksize后返回的是可逐块迭代的读取器
                                 usecols=["timestamp", "itemid", "property", "value"],
                                 dtype={"property": str, "value": str}):
            category_parts.append(chunk.loc[chunk["property"] == "categoryid", ["timestamp", "itemid", "value"]])
    return pd.concat(category_parts, # 将所有块筛出的类别记录纵向拼成一个DataFrame
                     ignore_index=True) # 行索引重新编号0,1,2...

# ==================== 2. 定义负采样方法 ====================

'''基于历史商品池的均匀负采样
1.只从当前时间之前出现过的商品中抽取负样本
2.排除用户过去浏览过当前同一时刻浏览的商品
3.每个正样本对应一个负样本，保持正负样本比例为 1:1
4.不使用未来行为信息，避免数据泄漏，因为预测时还不知道未来
'''
def sample_negative(item_pool, # item_pool:到目前为止，全体用户已经浏览过的商品池
                    excluded,  # excluded 包含用户已浏览和当前浏览的商品(当前用户不能抽成负样本的商品集合)
                    rng): # 随机数生成器
    # 随机尝试为提高运行效率
    for _ in range(20):  # 通常一两次就能找到，避免每条样本都遍历整个商品池
        if not item_pool:
            return None
        item_id = item_pool[rng.integers(len(item_pool))]  # 从已出现的不同商品中均匀抽取一个
        if item_id not in excluded: # 判断商品是否可以作为负样本
            return item_id
    # 20次尝试都失败才遍历整个商品池，创建剩余候选列表
    candidates = [item_id for item_id in item_pool if item_id not in excluded] # [放进新列表的表达式 for 变量 in 可迭代对象 if 筛选条件]
    return candidates[rng.integers(len(candidates))] if candidates else None

# ==================== 3. 清洗浏览事件，设置固定样本起点 ====================

def preprocess_events(events, item_properties, max_history_len=20,
                      val_ratio=0.2, sample_start_date="2015-07-02", seed=42):
    """item_properties 只接收类别记录，列结构与 load_category_properties 的返回值一致。"""
    if max_history_len < 1 or not 0 < val_ratio < 1:
        raise ValueError("历史长度至少为 1,验证比例在 0 和 1 之间。")

    views = events.loc[events["event"] == "view", ["timestamp", "visitorid", "itemid"]].dropna().copy()  # df.loc[行的选择条件, 列的选择条件]，本例只学习浏览
    views = views.astype("int64").drop_duplicates()  # 删除同一用户、商品、毫秒完全相同的重复日志
    views = views.sort_values("timestamp", kind="stable").reset_index(drop=True)  # 原文件并非严格按时间排序，stable 表示时间相同时保留原有排列
    if len(views) < 2:
        raise ValueError("至少需要两条不同时间的浏览事件，请增加读取的数据量。")

    # 类别已在读取时筛选；这里只清洗有效值、限制时间范围并排序。
    categories = item_properties.loc[
        item_properties["timestamp"] <= views["timestamp"].max(),
        ["timestamp", "itemid", "value"]
    ].dropna().copy().astype("int64").sort_values("timestamp", kind="stable")

    '''时间
              sample_start            split_time
    历史预热期       │       训练样本期       │      验证样本期
    '''
    sample_start = int(pd.Timestamp(sample_start_date, tz="Asia/Shanghai").timestamp() * 1000) # 为减少工作量硬编码
    window_views = views.loc[views["timestamp"] >= sample_start]
    if len(window_views) < 2:
        raise ValueError("指定起点后至少需要两条浏览，请调整 sample_start 或增加数据量。")
    split_index = int(len(window_views) * (1 - val_ratio))
    split_time = int(window_views.iloc[split_index]["timestamp"])
    if not (window_views["timestamp"] < split_time).any():
        raise ValueError("无法按时间划分：训练窗口为空，请增加不同时间的浏览事件。")
    train_views = views.loc[views["timestamp"] < split_time] # 包含预热期

    # ==================== 4. 建立商品、类别词表，整理类别变化记录 ====================

    item_mapping = {int(item): i + 1
                    for i, item in enumerate(train_views["itemid"].unique())} # 训练截止前被浏览过的商品ID字典，不会有重复因为unique去重
    # 与 Criteo 一样，已知商品从 1 开始编码；验证期新商品查不到编号，就先填成 0，后面跳过
    views["item_id"] = views["itemid"].map(item_mapping).fillna(0).astype("int32") # itemid：原始商品 ID；item_id：映射后的商品 ID
    categories = categories.loc[categories["itemid"].isin(item_mapping)] # split_time前浏览过的商品对应的全部类别记录都会保留
    # 只用训练截止前的类别建立词表；categories 保留截止前浏览商品的全部类别记录，供后面按时间更新类别。
    category_mapping = {int(category): i + 2
                        for i, category in enumerate(categories.loc[categories["timestamp"] < split_time, "value"].unique())}
    # 类别 0 表示补齐，1 表示未知；已知类别从 2 开始编号，验证期不扩大词表。
    categories["item_id"] = categories["itemid"].map(item_mapping) # 把原始商品 ID 对应的“新编号”加入 item_id 列
    # 获取迭代器按时间逐条读取元组 (时间, 商品编号, 原始类别)
    category_changes = iter(categories[["timestamp", "item_id", "value"]].itertuples(index=False, name=None))
    next_change = next(category_changes, None)
    # 用于记录每个商品在当前时间的类别。用商品编号作下标，例如 current_categories[3] 表示当前时间商品 3 当前的类别。
    current_categories = np.full(len(item_mapping) + 1, -1, dtype=np.int64)  # -1 表示没有过去类别记录

    # ==================== 5. 按时间生成历史序列和正负样本 ====================

    rng = np.random.default_rng(seed)  # 固定负采样结果，便于复现
    # 最近 L 次浏览供模型使用；全部已浏览商品用于负采样排除，两份记录作用不同。
    histories = defaultdict(lambda: deque(maxlen=max_history_len))  # 每条历史保存 (商品编号, 当时的原始类别)
    clicked = defaultdict(set)  # 保存全部过去浏览过的商品，用于负采样；不随历史截断而丢弃
    item_pool = []  # 保存已出现的不同商品，供随机抽取；添加前直接检查列表，避免重复
    samples = {"train": [], "val": []}  # 列表里每个元组是一条样本，正负样本相邻存放
    skipped = {"unknown_target": 0, "empty_history": 0, "no_negative": 0}

    # 整体流程：每次处理同一毫秒的浏览，先生成样本，再把当前浏览加入历史。
    rows = views[["timestamp", "visitorid", "item_id"]].itertuples(index=False, name=None) # 返回迭代器，每次取出一个元组
    for timestamp, group in groupby(rows, key=lambda row: row[0]):  # rows 已排序，groupby 返回分组依据和分好组的迭代器
        group = list(group)  # 保存这一毫秒的所有浏览，供下面三轮处理重复使用

        # 时间走到这里，先应用不晚于当前时刻的类别变化；正负候选都查询这份当前类别。
        while next_change is not None and next_change[0] <= timestamp:
            _, changed_item, category_id = next_change
            current_categories[changed_item] = category_id
            next_change = next(category_changes, None)

        # 第一轮：收集每个用户当前浏览的商品，防止把它们抽成负样本。
        current_clicks = defaultdict(set) #每处理一个新的时间戳，current_clicks 都会重新变成一个空字典
        for _, user_id, item_id in group:
            if item_id != 0:
                # defaultdict()可以在键不存在时，自动创建一个空集合，括号里的内容对应 value 的类型
                current_clicks[user_id].add(item_id)  # 同时浏览的其他商品也不能被误采成负样本

        # 第二轮：历史暂时不变，用过去的浏览为当前浏览生成一正一负两条样本。
        for _, user_id, item_id in group:
            if timestamp < sample_start:
                continue  # 窗口前只在第三轮积累历史和商品池，不生成样本
            history = histories[user_id]  # 取出该用户的过去历史；新用户得到空队列
            # 未知商品或没有历史时，不生成样本。
            if item_id == 0:
                skipped["unknown_target"] += 1
                continue
            if not history: # py允许很多数据类型直接参与逻辑判断，空容器为False
                skipped["empty_history"] += 1  # 首次浏览只积累历史，从后续浏览开始生成学习样本
                continue
            # 抽一个负商品，排除该用户过去和当前浏览过的商品。
            negative_id = sample_negative(item_pool, clicked[user_id] | current_clicks[user_id], rng)
            if negative_id is None:
                skipped["no_negative"] += 1  # 找不到负样本就跳过整对，保持正负样本 1:1
                continue

            split = "train" if timestamp < split_time else "val"
            # 分开历史商品和原始类别；商品补 0，原始类别暂补 -1，保存时再统一编码。
            padding = [0] * (max_history_len - len(history))
            history_ids = [item for item, _ in history] + padding
            history_categories = [category for _, category in history] + [-1] * len(padding)
            # 每条样本：(候选商品, 原始类别, 历史商品, 历史原始类别, 标签, 用户, 时间)。
            # 每次实际浏览生成一条正样本，再配一条负样本，全部保留。
            samples[split].append((item_id, current_categories[item_id], history_ids, history_categories, 1.0, user_id, timestamp))
            samples[split].append((negative_id, current_categories[negative_id], history_ids, history_categories, 0.0, user_id, timestamp))

        # 第三轮：样本全部生成后，再更新用户历史、浏览集合和全局商品池。
        # 已知商品即使没有生成样本，也要记入历史，供后续浏览使用。
        for _, user_id, item_id in group:
            if item_id != 0:
                histories[user_id].append((item_id, current_categories[item_id]))  # 记住浏览当时的类别，后续属性变化不改写旧历史
                clicked[user_id].add(item_id)
                if item_id not in item_pool:
                    item_pool.append(item_id)
        # 验证期也按时间更新真实历史，模拟逐次预测；验证标签不会参与模型参数更新。

    # ==================== 6. 转为与现有模型相近的 NumPy 格式 ====================

    datasets, category_stats = {}, {}
    for split, records in samples.items(): # items方法遍历字典的键和值
        if not records:
            raise ValueError(f"{split} 没有可用样本，请调整窗口规则或提供更长时间范围的数据。")
        # records 已按时间加入，每个元组是一条样本；zip(*records) 将相同字段收集到一起。
        targets, target_categories, history_ids, history_categories, labels, user_ids, timestamps = zip(*records)
        target_categories = np.asarray(target_categories, dtype=np.int64)
        history_categories = np.asarray(history_categories, dtype=np.int64)
        # 原始类别映射为模型编号；类别 1 是未知，只有历史补齐位置改成 0。
        target_category_ids = pd.Series(target_categories).map(category_mapping).fillna(1).to_numpy(dtype=np.int32)
        history_category_ids = (pd.Series(history_categories.ravel()).map(category_mapping).fillna(1)
                                .to_numpy(dtype=np.int32).reshape(history_categories.shape))
        history_ids = np.asarray(history_ids, dtype=np.int32)
        history_category_ids[history_ids == 0] = 0
        # 每个样本包含以下字段，每行是一条点击记录
        datasets[split] = {
            "target_ids": np.asarray(targets, dtype=np.int32),  # (N,)，该商品id
            "target_category_ids": target_category_ids,  # (N,)，该商品当前类别编码
            "history_ids": history_ids,  # (N, L)，该用户的浏览历史商品id
            "history_category_ids": history_category_ids,  # (N, L)，历史序列中各商品在被浏览时的类别编码
            "labels": np.asarray(labels, dtype=np.float32).reshape(-1, 1),  # (N, 1)，标签，正负样本1:1
            "user_ids": np.asarray(user_ids, dtype=np.int64),  # 该用户id，仅用于检查样本不输入模型
            "timestamps": np.asarray(timestamps, dtype=np.int64)  # 此刻时间戳
        }

        # 分别统计正候选、负候选、真实历史位置，区分“查不到属性”和“类别不在训练词表”。
        category_stats[split] = {}
        groups = {
            "positive": (target_categories[::2], target_category_ids[::2]),
            "negative": (target_categories[1::2], target_category_ids[1::2]),
            "history": (history_categories[::2][history_ids[::2] != 0],
                        history_category_ids[::2][history_ids[::2] != 0])  # 共用历史只统计一次，不含补齐
        }
        for name, (raw, encoded) in groups.items():
            category_stats[split][name] = {
                "count": len(raw),
                "known_ratio": float((encoded > 1).mean()),
                "no_past_record_count": int((raw == -1).sum()),
                "outside_train_vocab_count": int(((raw != -1) & (encoded == 1)).sum())
            }

    metadata = {
        "item_vocab_size": len(item_mapping) + 1,  # Embedding 行数包含补齐用的第 0 行
        "category_vocab_size": len(category_mapping) + 2,  # 包含补齐 0 和未知类别 1
        "padding_id": 0,
        "unknown_category_id": 1,
        "category_alignment": "target_as_of_prediction_history_as_of_event",
        "max_history_len": max_history_len,
        "split_timestamp": split_time,
        "view_count": len(views),
        "warmup_view_count": int((views["timestamp"] < sample_start).sum()),
        "window_view_count": len(window_views),
        "sample_start_timestamp": sample_start,
        "end_timestamp": int(views["timestamp"].max()),
        "timezone": "Asia/Shanghai",
        "train_sample_count": len(datasets["train"]["labels"]),
        "val_sample_count": len(datasets["val"]["labels"]),
        "category_stats": category_stats,  # 保留样本的统计，历史按每对一次计数
        "skipped_view_count": skipped,  # 记录哪些浏览没有生成正负样本对
        "negative_sampling": "one_uniform_past_item_excluding_past_and_current_clicks",
        "validation_protocol": "rolling_history_known_items_only",
        "seed": seed
    }
    return datasets, metadata, item_mapping, category_mapping

def main():
    # ==================== 7. 读取原始数据并保存处理结果 ====================

    events = pd.read_csv(filepath, usecols=["timestamp", "visitorid", "event", "itemid"])
    # 完整读取浏览时间线，按固定日期选窗，保留训练、验证区间的全部有效样本。
    item_properties = load_category_properties(filepath.parent)
    datasets, metadata, item_mapping, category_mapping = preprocess_events(events, item_properties)
    save_dir.mkdir(parents=True, exist_ok=True)
    for split, data in datasets.items():
        np.savez_compressed(save_dir / f"{split}_common.npz", **data)  # 沿用已有脚本的 NPZ 文件名和 labels 字段
    for filename, data in [("metadata.json", metadata), ("item_mapping.json", item_mapping), ("category_mapping.json", category_mapping)]:
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  # JSON 中商品原始 ID 会保存为字符串键

    # ==================== 8. 检查最终结果 ====================

    for name, key in [("样本窗口开始", "sample_start_timestamp"), ("训练/验证边界", "split_timestamp"), ("窗口结束", "end_timestamp")]:
        date = pd.to_datetime(metadata[key], unit="ms", utc=True).tz_convert(metadata["timezone"])
        print(f"{name}：{date}")
    print(f"预热浏览数：{metadata['warmup_view_count']}，窗口内浏览数：{metadata['window_view_count']}")
    for split, data in datasets.items():
        print(f"{split}: 候选商品 {data['target_ids'].shape}, 历史序列 {data['history_ids'].shape}, 标签 {data['labels'].shape}")
        print(f"  正负样本对数：{len(data['labels']) // 2}")
        for name, stats in metadata["category_stats"][split].items():
            print(f"  {name} 类别已知比例 {stats['known_ratio']:.2%}，"
                  f"无过去属性 {stats['no_past_record_count']}，不在训练词表 {stats['outside_train_vocab_count']}")
    print(f"商品词表大小（含补齐 0）：{metadata['item_vocab_size']}")
    print(f"类别词表大小（含补齐 0 和未知 1）：{metadata['category_vocab_size']}")
    print("未生成样本的浏览：", metadata["skipped_view_count"])
    print("RetailRocket 数据预处理完成！")

if __name__ == "__main__":  # 直接运行才处理数据；import 时可单独学习和检查上面的函数
    main()
