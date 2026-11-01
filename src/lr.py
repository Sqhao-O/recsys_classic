import numpy as np  # 导入 NumPy，用于读取和处理数组
import torch  # 导入 PyTorch，用于建立和训练 LR 模型

from pathlib import Path  # 导入 Path，用于处理文件路径
from scipy import sparse  # 导入 SciPy 稀疏矩阵模块
from sklearn.metrics import roc_auc_score  # 导入 AUC 指标，用于评价点击分数的排序能力
from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 数据集和加载器


# ==================== 1. 设置数据路径 ====================

base_dir = Path(__file__).resolve().parent  # __file__ 是当前脚本路径；resolve() 转为绝对路径，parent 取得其所在目录
data_dir = base_dir / "datasets/Criteo/processed"  # 设置预处理数据所在目录

# ==================== 2. 定义数据集 ====================

class CriteoDataset(Dataset):  # 定义 Criteo 数据集类，继承 PyTorch Dataset

    def __init__(self, feature_path, common_path):  # 初始化时传入特征文件和通用数据文件路径
        self.x_data = sparse.load_npz(feature_path)  # 预处理已将 LR 特征保存为 float32 的 CSR 矩阵
        with np.load(common_path) as data:  # 打开通用数据文件
            self.y_data = torch.from_numpy(data["labels"])  # 标签已是 float32，直接转换为张量

    def __getitem__(self, index):  # 定义如何获取指定索引的一条样本
        '''note
        1. NumPy 数组和 PyTorch 张量使用整数索引取行时，会移除被索引的维度。
        2. SciPy 二维稀疏矩阵使用整数索引取行时，通常仍保留二维结构 (1, n)。
        3. toarray()：将稀疏矩阵转换为普通 NumPy 数组。
        4. ravel()：将数组展平为一维，形状从 (1, n) 变为 (n,)。
        '''
        x = torch.from_numpy(self.x_data[index].toarray().ravel())
        return x, self.y_data[index]  # 返回特征与标签组成的元组

    def __len__(self):  # 定义如何获取数据集长度
        return self.x_data.shape[0]  # 直接读取样本总数，无需额外保存

# ==================== 3. 创建训练集和验证集 ====================

train_data = CriteoDataset(
    data_dir / "train_lr.npz",  # 传入 LR 训练特征文件
    data_dir / "train_common.npz"  # 传入训练标签所在文件
)

val_data = CriteoDataset(
    data_dir / "val_lr.npz",  # 传入 LR 验证特征文件
    data_dir / "val_common.npz"  # 传入验证标签所在文件
)

# ==================== 4. 创建 DataLoader ====================

# 默认 num_workers=0，在主进程中加载数据；验证集默认不打乱。
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256)

# ==================== 5. 定义 LR 模型 ====================

class LR(torch.nn.Module):  # 定义逻辑回归模型

    def __init__(self, input_dim):  # 初始化模型，传入输入特征维度
        super().__init__()  # 初始化父类 nn.Module
        self.linear = torch.nn.Linear(input_dim, out_features=1)

    def forward(self, x):  # 定义前向传播方法
        return self.linear(x)  # 计算线性组合并返回原始 Logit

# ==================== 6. 初始化模型 ====================

torch.manual_seed(42)  # 固定 PyTorch 随机种子，方便复现实验

input_dim = train_data.x_data.shape[1]  # 自动获取 LR 编码后的输入特征数量
model = LR(input_dim)  # 创建逻辑回归模型

# ==================== 7. 定义损失函数和优化器 ====================

criterion = torch.nn.BCEWithLogitsLoss()  # 定义二分类交叉熵损失函数，内部已经包含 Sigmoid
optimizer = torch.optim.SGD(
    model.parameters(),  # 指定优化器更新的模型参数
    lr=0.01  # 设置学习率
)

# ==================== 8. 训练并逐轮验证 ====================

num_epochs = 100  # 设置总训练轮数

for epoch in range(num_epochs):  # 循环执行多个 Epoch
    model.train()  # 将模型切换为训练模式
    total_train_loss = 0.0  # 初始化当前 Epoch 的累计损失

    for input_train, label_train in train_loader:  # 按 Batch 获取训练特征和标签
        output_train = model(input_train)  # 前向传播，计算模型输出
        loss = criterion(output_train, label_train)  # 计算当前 Batch 的平均损失
        optimizer.zero_grad()  # 清空上一轮反向传播留下的梯度
        loss.backward()  # 反向传播，计算模型参数的梯度
        optimizer.step()  # 使用 SGD 更新模型参数
        total_train_loss += loss.item() * input_train.shape[0]  # 当前 Batch 的平均损失乘以样本数量，累加为损失总和

    avg_train_loss = total_train_loss / len(train_data)  # 根据训练样本总数计算平均损失

    model.eval()  # 每轮训练结束后切换为评估模式
    correct = 0  # 初始化当前轮预测正确的样本数
    total_val_loss = 0.0  # 初始化当前轮验证集累计损失
    val_logits = []  # 保存整个验证集的原始分数，用于计算 AUC
    val_labels = []  # 保存与分数一一对应的真实标签

    with torch.no_grad():  # 验证时关闭梯度计算
        for input_val, label_val in val_loader:  # 按 Batch 读取验证集
            logits = model(input_val)  # 前向传播，得到原始 Logit
            loss = criterion(logits, label_val)  # 计算当前 Batch 的平均验证损失
            total_val_loss += loss.item() * input_val.shape[0]  # 按样本数量累加，正确处理最后一个较小的 Batch
            pred_label = logits >= 0  # 等价于 Sigmoid 概率 >= 0.5，无需额外转换
            correct += (pred_label == label_val).sum().item()  # 统计预测正确的样本数
            val_logits.append(logits)
            val_labels.append(label_val)

    avg_val_loss = total_val_loss / len(val_data)  # 计算整个验证集的平均损失
    accuracy = correct / len(val_data)  # 计算验证集准确率
    # AUC 使用整个验证集的排序分数计算；Logit 可直接使用，无需先计算 Sigmoid。
    val_auc = roc_auc_score(
        torch.cat(val_labels).numpy().ravel(),
        torch.cat(val_logits).numpy().ravel()
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val AUC: {val_auc:.4f}, "
        f"Val Accuracy: {accuracy * 100:.2f}%"
    )
