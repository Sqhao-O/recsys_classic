import json  # 读取预处理保存的商品词表大小
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

# ==================== 1. 设置数据路径 ====================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "datasets/RetailRocket/processed"

# ==================== 2. 定义数据集 ====================

class RetailRocketDataset(Dataset):
    def __init__(self, common_path):
        # 沿用 DeepFM 的 NPZ -> Tensor 写法，只把数值/类别特征换成候选商品和历史序列。
        with np.load(common_path) as data:
            self.target_ids = torch.from_numpy(data["target_ids"]).long()  # (N,)，整数 ID 用于查 Embedding
            self.history_ids = torch.from_numpy(data["history_ids"]).long()  # (N, L)，0 表示补齐位置
            self.y_data = torch.from_numpy(data["labels"])  # (N, 1)，float32 标签用于计算二分类损失

    def __getitem__(self, index):
        return self.target_ids[index], self.history_ids[index], self.y_data[index]

    def __len__(self):
        return self.y_data.shape[0]

# ==================== 3. 定义 DIN 模型 ====================

class DIN(torch.nn.Module):
    def __init__(self, item_vocab_size, embedding_dim=8):
        super().__init__()
        # 候选商品和历史商品共用一张表，同一个商品在两处查到的是同一个向量。
        self.item_embedding = torch.nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        torch.nn.init.normal_(self.item_embedding.weight[1:], std=0.01)  # 小随机数初始化，保留第 0 行为全零

        # 局部激活单元：分别比较候选商品 q 与每条历史行为 h，输出这条行为的相关性分数。
        # 拼接 [q, h, q-h, q*h]，让网络同时看到二者本身、差异和逐元素交互；* 不是矩阵乘法。
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim, 32),  # 每个历史位置的 4K 维特征映射为 32 维
            torch.nn.PReLU(),  # 正半轴保持不变，负半轴斜率可学习；比 ReLU 多一个可学习参数
            torch.nn.Linear(32, 16),
            torch.nn.PReLU(),  # 为保持简洁，用 PReLU，不引入论文中的 Dice 和 mini-batch aware 正则
            torch.nn.Linear(16, 1)  # 每条历史行为输出一个标量权重，暂不求和
        )
        # 得到与当前候选相关的兴趣向量后，像 DeepFM 一样交给 MLP 输出 Logit。
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(3 * embedding_dim, 32),  # 输入为 [候选向量, 兴趣向量, 二者逐元素乘积]
            torch.nn.PReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.PReLU(),
            torch.nn.Linear(16, 1)  # (B, 1)，末尾不加 Sigmoid，与 BCEWithLogitsLoss 配套
        )

    def get_interest(self, target, history, mask):
        # B 是 Batch 大小，L 是历史最大长度，K 是 Embedding 维度。
        # target: (B, K)，history: (B, L, K)，mask: (B, L)，True 表示真实浏览。
        query = target.unsqueeze(1).expand_as(history)  # (B, K) -> (B, 1, K) -> (B, L, K)，与每条历史对齐
        attention_input = torch.cat([query, history, query - history, query * history], dim=-1)  # (B, L, 4K)
        weights = self.attention(attention_input).squeeze(-1)  # Linear 只改变最后一维，(B, L, 1) -> (B, L)
        weights = weights.masked_fill(~mask, 0.0)  # ~ 表示布尔取反；补齐位置必须明确置零，不能参与兴趣汇总
        # 这里采用 DIN 论文中的非归一化加权和，不做 Softmax；权重之和不必为 1，也可以为负。
        # 保留权重整体大小，才能表达兴趣强度；空历史全被遮住，兴趣向量自然为零，不会产生 NaN。
        interest = (weights.unsqueeze(-1) * history).sum(dim=1)  # (B, L, 1) × (B, L, K)，沿历史维求和为 (B, K)
        return interest

    def forward(self, target_ids, history_ids):
        target = self.item_embedding(target_ids)  # (B,) -> (B, K)，当前候选商品的向量
        history = self.item_embedding(history_ids)  # (B, L) -> (B, L, K)，每条过去浏览的向量
        mask = history_ids != 0  # (B, L)，区分真正的历史行为和右侧补齐的 0
        interest = self.get_interest(target, history, mask)  # 同一段历史面对不同候选，会得到不同的兴趣向量
        deep_input = torch.cat([target, interest, target * interest], dim=1)  # (B, 3K)，保留候选与兴趣的交互
        return self.deep(deep_input)  # 返回 (B, 1) 的 Logit；预测概率时再调用 torch.sigmoid


def main(num_epochs=20):
    # ==================== 4. 创建训练集和验证集 ====================

    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError("请先从项目根目录执行 python src/preprocess_retailrocket.py。")
    train_data = RetailRocketDataset(data_dir / "train_common.npz")
    val_data = RetailRocketDataset(data_dir / "val_common.npz")

    # ==================== 5. 创建 DataLoader ====================

    torch.manual_seed(42)  # 在创建模型和遍历 DataLoader 前固定随机种子
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)  # 预处理已按时间切好数据，训练时打乱不会改变每条样本的历史

    # ==================== 6. 初始化模型 ====================

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    model = DIN(item_vocab_size=metadata["item_vocab_size"], embedding_dim=8)

    # ==================== 7. 定义损失函数和优化器 ====================

    criterion = torch.nn.BCEWithLogitsLoss()  # 与 DeepFM 相同，直接输入原始 Logit
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ==================== 8. 训练并逐轮验证 ====================

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for target_train, history_train, label_train in train_loader:
            output_train = model(target_train, history_train)
            loss = criterion(output_train, label_train)
            optimizer.zero_grad()  # 清空上个 Batch 的梯度
            loss.backward()  # 同时更新商品向量、注意力网络和最终 MLP
            optimizer.step()
            total_train_loss += loss.item() * label_train.shape[0]  # 按样本数累加，正确处理最后一个小 Batch
        avg_train_loss = total_train_loss / len(train_data)

        model.eval()
        correct = 0
        total_val_loss = 0.0
        val_logits, val_labels = [], []
        with torch.no_grad():  # 验证只计算分数，不记录梯度，也不更新参数
            for target_val, history_val, label_val in val_loader:
                logits = model(target_val, history_val)
                loss = criterion(logits, label_val)
                total_val_loss += loss.item() * label_val.shape[0]
                correct += ((logits >= 0) == label_val).sum().item()  # Logit >= 0 等价于概率 >= 0.5
                val_logits.append(logits)
                val_labels.append(label_val)

        avg_val_loss = total_val_loss / len(val_data)
        accuracy = correct / len(val_data)
        # 复用 DeepFM 的指标计算：收集整个验证集后算 AUC，不对每个 Batch 的 AUC 取平均。
        val_auc = roc_auc_score(
            torch.cat(val_labels).numpy().ravel(),
            torch.cat(val_logits).numpy().ravel()  # Sigmoid 不改变排序，AUC 可直接使用 Logit
        )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}, "
            f"Val Accuracy: {accuracy * 100:.2f}%"
        )


if __name__ == "__main__":  # import DIN 类时不自动加载数据或开始训练
    main()