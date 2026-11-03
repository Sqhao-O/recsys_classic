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
        # 从 NPZ 读取候选和历史，每部分都包含商品 ID、类别 ID。
        with np.load(common_path) as data:
            self.target_ids = torch.from_numpy(data["target_ids"]).long()  # (N,)，整数 ID 用于查 Embedding
            self.target_category_ids = torch.from_numpy(data["target_category_ids"]).long()  # (N,)
            self.history_ids = torch.from_numpy(data["history_ids"]).long()  # (N, L)，0 表示补齐位置
            self.history_category_ids = torch.from_numpy(data["history_category_ids"]).long()  # (N, L)，与历史商品逐位置对应
            self.y_data = torch.from_numpy(data["labels"])  # (N, 1)，float32 标签用于计算二分类损失

    def __getitem__(self, index):
        return (self.target_ids[index], self.target_category_ids[index],
                self.history_ids[index], self.history_category_ids[index], self.y_data[index])

    def __len__(self):
        return self.y_data.shape[0]

# ==================== 3. 定义 DIN 模型 ====================

class DIN(torch.nn.Module):
    def __init__(self, item_vocab_size, category_vocab_size, embedding_dim=8):
        super().__init__()
        # 商品和类别各有一张表；候选与历史共用这两张表，保证相同 ID 查到相同向量。
        # 历史用 ID 0 补齐（如 [3, 7, 0, 0, 0]），因此预留第0行为零向量，缺失的0查第一行即可
        self.item_embedding = torch.nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.category_embedding = torch.nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        torch.nn.init.normal_(self.item_embedding.weight[1:], std=0.01)  # 小随机数初始化，保留第 0 行为全零
        torch.nn.init.normal_(self.category_embedding.weight[1:], std=0.01)  # 未知类别 1 可学习，只有补齐 0 固定为零
        feature_dim = 2 * embedding_dim  # 拼接商品、类别向量；默认 8 + 8 = 16 维

        # 局部激活单元：分别比较候选商品 q 与【每条】历史行为 h，输出这条行为的相关性分数。
        # 拼接 [q, h, q-h, q*h]，让网络同时看到二者本身、差异和逐元素交互；* 不是矩阵乘法。
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(4 * feature_dim, 32),  # 每个历史位置的 4K 维特征映射为 32 维
            torch.nn.PReLU(),  # 正半轴保持不变，负半轴斜率可学习；比 ReLU 多一个可学习参数
            torch.nn.Linear(32, 16),
            torch.nn.PReLU(),  # 为保持简洁，用 PReLU，不引入论文中的 Dice 和 mini-batch aware 正则
            torch.nn.Linear(16, 1)  # 每条历史行为输出一个标量权重，暂不求和
        )
        # 得到与当前候选相关的兴趣向量后，像 DeepFM 一样交给 MLP 输出 Logit。
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(3 * feature_dim, 32),  # 输入为 [候选向量, 兴趣向量, 二者逐元素乘积]
            torch.nn.PReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.PReLU(),
            torch.nn.Linear(16, 1)  # (B, 1)，末尾不加 Sigmoid，与 BCEWithLogitsLoss 配套
        )

    def embed_items(self, item_ids, category_ids):
        # 一个商品的表示 = [商品 ID 向量, 类别向量]，注意力会同时看到这两部分信息。
        return torch.cat([self.item_embedding(item_ids), self.category_embedding(category_ids)], dim=-1)

    def embedding_l2(self):
        # 普通 L2：限制商品、类别向量的大小，减轻对稀疏 ID 的记忆；不包含补齐用的第 0 行。
        return self.item_embedding.weight[1:].square().sum() + self.category_embedding.weight[1:].square().sum()

    def get_interest(self, target, history, mask):
        # B 是 Batch 大小，L 是历史最大长度，K 是商品和类别拼接后的维度。
        # target: (B, K)，history: (B, L, K)，mask: (B, L)，True 表示真实浏览。
        query = target.unsqueeze(1).expand_as(history)  # (B, K) -> (B, 1, K) -> (B, L, K)，与每条历史对齐
        attention_input = torch.cat([query, history, query - history, query * history], dim=-1)  # (B, L, 4K)
        weights = self.attention(attention_input).squeeze(-1)  # Linear 只改变最后一维，(B, L, 1) -> (B, L)
        weights = weights.masked_fill(~mask, 0.0)  # ~ 表示布尔取反；补齐位置必须明确置零，不能参与兴趣汇总
        # 这里采用 DIN 论文中的非归一化加权和，不做 Softmax；权重之和不必为 1，也可以为负。
        # 保留权重整体大小，才能表达兴趣强度；空历史全被遮住，兴趣向量自然为零，不会产生 NaN。
        interest = (weights.unsqueeze(-1) * history).sum(dim=1)  # (B, L, 1) × (B, L, K)，沿历史维求和为 (B, K)
        return interest

    def forward(self, target_ids, target_category_ids, history_ids, history_category_ids):
        target = self.embed_items(target_ids, target_category_ids)  # (B, K)，候选商品 ID + 类别
        history = self.embed_items(history_ids, history_category_ids)  # (B, L, K)，历史商品 ID + 当时类别
        mask = history_ids != 0  # (B, L)，按商品判断是否补齐；未知类别 1 仍然是真实浏览
        interest = self.get_interest(target, history, mask)  # 同一段历史面对不同候选，会得到不同的兴趣向量
        deep_input = torch.cat([target, interest, target * interest], dim=1)  # (B, 3K)，保留候选与兴趣的交互
        return self.deep(deep_input)  # 返回 (B, 1) 的 Logit；预测概率时再调用 torch.sigmoid


def main(num_epochs=20, batch_size=1024):
    # ==================== 4. 创建训练集和验证集 ====================

    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError("请先从项目根目录执行 python src/preprocess_retailrocket.py。")
    train_data = RetailRocketDataset(data_dir / "train_common.npz")
    val_data = RetailRocketDataset(data_dir / "val_common.npz")
    print(f"训练样本：{len(train_data)}，验证样本：{len(val_data)}")

    # ==================== 5. 创建 DataLoader ====================

    torch.manual_seed(42)  # 在创建模型和遍历 DataLoader 前固定随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    # 数据先留在 CPU 内存，每个 Batch 再搬到 GPU；页锁定内存配合异步传输。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=use_cuda)
    print(f"训练设备：{torch.cuda.get_device_name(device) if use_cuda else 'CPU'}，Batch 大小：{batch_size}")

    # ==================== 6. 初始化模型 ====================

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    model = DIN(item_vocab_size=metadata["item_vocab_size"],
                category_vocab_size=metadata["category_vocab_size"], embedding_dim=8).to(device)

    # ==================== 7. 定义损失函数和优化器 ====================

    criterion = torch.nn.BCEWithLogitsLoss()  # 与 DeepFM 相同，直接输入原始 Logit
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    l2_lambda = 1e-4  # 商品和类别 Embedding 的 L2 系数，可在这里调整强度
    # 训练目标 = BCE + l2_lambda × Embedding 向量平方和。
    print(f"Embedding L2 系数：{l2_lambda:g}（Train/Val Loss 只统计 BCE，Reg Loss 单独显示）")

    # ==================== 8. 训练并逐轮验证 ====================

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_reg_loss = 0.0
        for batch in train_loader:
            # 候选、类别、历史和标签一起搬到模型所在设备，保证计算时设备一致。
            target_train, target_category_train, history_train, history_category_train, label_train = (
                tensor.to(device, non_blocking=use_cuda) for tensor in batch
            )
            output_train = model(target_train, target_category_train, history_train, history_category_train)
            data_loss = criterion(output_train, label_train)
            reg_loss = l2_lambda * model.embedding_l2()
            loss = data_loss + reg_loss  # 正则项参与反向传播，验证时不加入
            optimizer.zero_grad()  # 清空上个 Batch 的梯度
            loss.backward()  # 同时更新商品向量、类别向量、注意力网络和最终 MLP
            optimizer.step()
            total_train_loss += data_loss.item() * label_train.shape[0]  # 只统计 BCE，和验证 Loss 保持相同口径
            total_reg_loss += reg_loss.item() * label_train.shape[0]
        avg_train_loss = total_train_loss / len(train_data)
        avg_reg_loss = total_reg_loss / len(train_data)

        model.eval()
        correct = 0
        total_val_loss = 0.0
        val_logits, val_labels = [], []
        with torch.no_grad():  # 验证只计算分数，不记录梯度，也不更新参数
            for batch in val_loader:
                target_val, target_category_val, history_val, history_category_val, label_val = (
                    tensor.to(device, non_blocking=use_cuda) for tensor in batch
                )
                logits = model(target_val, target_category_val, history_val, history_category_val)
                loss = criterion(logits, label_val)
                total_val_loss += loss.item() * label_val.shape[0]
                correct += ((logits >= 0) == label_val).sum().item()  # Logit >= 0 等价于概率 >= 0.5
                val_logits.append(logits.cpu())  # AUC 由 sklearn 在 CPU 上计算，不能直接将 CUDA 张量转为 NumPy
                val_labels.append(label_val.cpu())

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
            f"Reg Loss: {avg_reg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}, "
            f"Val Accuracy: {accuracy * 100:.2f}%"
        )


if __name__ == "__main__":  # import DIN 类时不自动加载数据或开始训练
    main()
