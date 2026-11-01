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

    # 遍历整个商品池，创建剩余候选列表
    candidates = [item_id for item_id in item_pool if item_id not in excluded]
    return candidates[rng.integers(len(candidates))] if candidates else None

# ==================== 3. 清洗浏览事件并按时间划分 ====================

def preprocess_events(events, max_events=200000, max_history_len=20, val_ratio=0.2, seed=42):
    if max_history_len < 1 or not 0 < val_ratio < 1 or (max_events is not None and max_events < 2):
        raise ValueError("历史长度至少为 1，验证比例在 0 和 1 之间，事件上限至少为 2。")

    cols = ["timestamp", "visitorid", "itemid"]
    views = events.loc[events["event"] == "view", cols].dropna().copy()  # 本例只学习浏览，不把加购/购买当负样本
    views = views.astype("int64").drop_duplicates()  # 删除同一用户、商品、毫秒完全相同的重复日志
    views = views.sort_values("timestamp", kind="stable").reset_index(drop=True)  # 原文件并非严格按时间排序，stable 表示时间相同时保留原有排列
    if max_events is not None and len(views) > max_events:
        end_time = views.iloc[max_events - 1]["timestamp"] # 取出要求保留的浏览事件个数最后的事件的时间戳
        views = views.loc[views["timestamp"] <= end_time]  # 先排序再取时间早的一段，并完整保留边界处同一毫秒的事件
    if len(views) < 2:
        raise ValueError("至少需要两条不同时间的浏览事件，请增加读取的数据量。")

    split_index = int(len(views) * (1 - val_ratio))
    split_time = int(views.iloc[split_index]["timestamp"])  # 边界时间
    train_views = views.loc[views["timestamp"] < split_time]  # 小于时间边界的用于训练
    if train_views.empty:
        raise ValueError("无法按时间划分：训练区间为空，请增加不同时间的浏览事件。")

    # ==================== 4. 仅用训练区间建立商品词表 ====================

    item_mapping = {int(item): i + 1 for i, item in enumerate(train_views["itemid"].unique())}
    # 与 Criteo 一样，已知商品从 1 开始编码；验证期新商品查不到编号，就先填成 0，后面跳过
    views["item_id"] = views["itemid"].map(item_mapping).fillna(0).astype("int32") # itemid：原始商品 ID；item_id：映射后的商品 ID

    # ==================== 5. 按时间生成历史序列和正负样本 ====================

    rng = np.random.default_rng(seed)  # 固定随机数生成器，重新预处理也能得到相同的数据
    '''遍历用到的变量
    histories:每个用户最近最多 20 次已知商品浏览
    clicked:每个用户过去浏览过的全部已知商品集合
    item_pool:全体用户在过去浏览过的不同商品，供随机抽取
    appeared:记录商品是否已入池，避免重复加入
    samples:存放生成的训练、验证样本
    skipped:统计因各种原因未生成样本的浏览
    '''
    histories = defaultdict(lambda: deque(maxlen=max_history_len))  # 创建字典，value是队列，每个用户保留最近 L 次已知商品浏览，允许重复浏览
    clicked = defaultdict(set)  # 保存全部过去浏览过的商品，用于负采样；不随历史截断而丢弃
    item_pool, appeared = [], set()  # 列表用于随机抽取，集合用于去重；只有过去已出现的商品才能入池
    samples = {"train": [], "val": []}
    skipped = {"unknown_target": 0, "empty_history": 0, "no_negative": 0}

    # 整体流程：每次处理同一毫秒的浏览，先生成样本，再把当前浏览加入历史。
    rows = views[["timestamp", "visitorid", "item_id"]].itertuples(index=False, name=None) # 每一行以元组取出
    for timestamp, group in groupby(rows, key=lambda row: row[0]):  # rows 已排序，groupby 返回分组依据和分好组的迭代器
        group = list(group)  # 保存这一毫秒的所有浏览，供下面三轮处理重复使用

        # 第一轮：收集每个用户当前浏览的商品，防止把它们抽成负样本。
        current_clicks = defaultdict(set)
        for _, user_id, item_id in group:
            if item_id != 0:
                # defaultdict()可以在键不存在时，自动创建一个空集合，括号里的内容对应 value 的类型
                current_clicks[user_id].add(item_id)  # 同时浏览的其他商品也不能被误采成负样本

        # 第二轮：历史暂时不变，用过去的浏览为当前浏览生成一正一负两条样本。
        for _, user_id, item_id in group:
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

            # 补齐历史，并按时间把这对样本存入训练集或验证集。
            history_ids = list(history) + [0] * (max_history_len - len(history))  # 按从旧到新排列，右侧补 0
            split = "train" if timestamp < split_time else "val"
            # 同一对样本共用完全相同的过去历史，只改变候选商品和标签；当前点击还没有加入历史。
            samples[split].append((item_id, history_ids, 1.0, user_id, timestamp))
            samples[split].append((negative_id, history_ids, 0.0, user_id, timestamp))

        # 第三轮：样本全部生成后，再更新用户历史、浏览集合和全局商品池。
        # 已知商品即使没有生成样本，也要记入历史，供后续浏览使用。
        for _, user_id, item_id in group:
            if item_id != 0:
                histories[user_id].append(item_id)
                clicked[user_id].add(item_id)
                if item_id not in appeared:
                    item_pool.append(item_id)
                    appeared.add(item_id)
        # 验证期也按时间更新真实历史，模拟逐次预测；验证标签不会参与模型参数更新。

    # ==================== 6. 转为与现有模型相近的 NumPy 格式 ====================

    datasets = {}
    for split, records in samples.items(): # items方法遍历字典的键和值
        if not records:
            raise ValueError(f"{split} 没有可用样本，请增加 max_events，或用 max_events=None 读取全部浏览。")
        targets, history_ids, labels, user_ids, timestamps = zip(*records)
        datasets[split] = {
            "target_ids": np.asarray(targets, dtype=np.int32),  # (N,)，每条样本的候选商品
            "history_ids": np.asarray(history_ids, dtype=np.int32),  # (N, L)，定长的过去浏览序列
            "labels": np.asarray(labels, dtype=np.float32).reshape(-1, 1),  # (N, 1)，与 DeepFM 相同
            "user_ids": np.asarray(user_ids, dtype=np.int64),  # 仅用于检查样本，不输入模型
            "timestamps": np.asarray(timestamps, dtype=np.int64)  # 保留毫秒时间戳，便于核对时间划分
        }

    metadata = {
        "item_vocab_size": len(item_mapping) + 1,  # Embedding 行数包含补齐用的第 0 行
        "padding_id": 0,
        "max_history_len": max_history_len,
        "split_timestamp": split_time,
        "selected_view_count": len(views),
        "train_sample_count": len(samples["train"]),
        "val_sample_count": len(samples["val"]),
        "skipped_view_count": skipped,  # 记录哪些浏览没有生成正负样本对
        "negative_sampling": "one_uniform_past_item_excluding_past_and_current_clicks",
        "validation_protocol": "rolling_history_known_items_only",
        "seed": seed
    }
    return datasets, metadata, item_mapping

def main():
    # ==================== 7. 读取原始数据并保存处理结果 ====================

    events = pd.read_csv(filepath, usecols=["timestamp", "visitorid", "event", "itemid"])
    # 全量读取这四列后再排序，确保取到全局最早的浏览，而不是原文件前 20 万行。
    datasets, metadata, item_mapping = preprocess_events(events, max_events=200000, max_history_len=20)
    # 学习时先用最早约 20 万次浏览；改为 max_events=None 可处理全部数据，内存和训练时间也会增加。
    save_dir.mkdir(parents=True, exist_ok=True)
    for split, data in datasets.items():
        np.savez_compressed(save_dir / f"{split}_common.npz", **data)  # 沿用已有脚本的 NPZ 文件名和 labels 字段
    for filename, data in [("metadata.json", metadata), ("item_mapping.json", item_mapping)]:
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  # JSON 中商品原始 ID 会保存为字符串键

    # ==================== 8. 检查最终结果 ====================

    for split, data in datasets.items():
        print(f"{split}: 候选商品 {data['target_ids'].shape}, 历史序列 {data['history_ids'].shape}, 标签 {data['labels'].shape}")
    print(f"商品词表大小（含补齐 0）：{metadata['item_vocab_size']}")
    print("未生成样本的浏览：", metadata["skipped_view_count"])
    print("RetailRocket 数据预处理完成！")

if __name__ == "__main__":  # 直接运行才处理数据；import 时可单独学习和检查上面的函数
    main()
