import json  # 导入 JSON 模块，用于读取预处理保存的特征配置信息
from pathlib import Path  # 导入 Path，方便拼接路径并定位当前脚本所在目录

import numpy as np  # 导入 NumPy，用于读取通用数据集中的数组
import torch  # 导入 PyTorch，用于定义张量、模型和训练过程
from sklearn.metrics import roc_auc_score  # 导入 AUC 指标，评价点击样本相对未点击样本的排序能力
from torch.utils.data import Dataset, DataLoader  # Dataset 定义如何取样，DataLoader 负责打乱和组成 Batch


# ==================== 1. 设置数据路径 ====================

base_dir = Path(__file__).resolve().parent  # __file__ 是当前脚本路径；resolve() 转为绝对路径，parent 取得其所在目录
data_dir = base_dir / "datasets/Criteo/processed"  # Path 用 / 拼接路径，定位现有的预处理数据目录


# ==================== 2. 定义数据集 ====================

# 数据形状约定：N 表示整个数据集的样本数，B 表示当前 Batch 的样本数。
# DataLoader 会把多条样本中同一位置的张量分别堆叠，因此每批仍返回三个张量。
class CriteoDataset(Dataset):  # 继承 Dataset，定义三个新模型共用的数据读取方式
    def __init__(self, common_path):  # 创建数据集时执行，common_path 是一个通用数据文件的路径
        with np.load(common_path) as data:  # 打开 npz 文件，使用其中的数组；退出 with 时关闭文件
            self.numeric = torch.from_numpy(data["numeric"])  # from_numpy 将数组转为共享内存的张量，形状 (N, 13)，保留 float32 类型
            self.categorical = torch.from_numpy(data["categorical_ids"]).long()  # long() 转为 int64 整数张量，形状 (N, 26)，用于类别查表
            self.y_data = torch.from_numpy(data["labels"])  # 读取 float32 标签，形状 (N, 1)，与 LR 的标签格式相同

    def __getitem__(self, index):  # DataLoader 根据索引调用此方法，获取一条样本
        return self.numeric[index], self.categorical[index], self.y_data[index]  # 单条样本的三个形状分别为 (13,)、(26,)、(1,)

    def __len__(self):  # 定义 len(dataset) 的返回值，供 DataLoader 确定样本数量
        return self.y_data.shape[0]  # shape[0] 是第 0 维长度，也就是当前数据集的样本数


# ==================== 3. 创建训练集和验证集 ====================

train_data = CriteoDataset(data_dir / "train_common.npz")  # 读取训练集的数值特征、类别 ID 和标签
val_data = CriteoDataset(data_dir / "val_common.npz")  # 读取验证集，使用与训练集一致的预处理规则


# ==================== 4. 创建 DataLoader ====================

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)  # 每批最多 256 条，shuffle=True 表示每轮重新打乱训练样本
val_loader = DataLoader(val_data, batch_size=256)  # 验证集默认不打乱；默认 num_workers=0，在主进程中加载


# ==================== 5. 定义 DeepFM 模型 ====================

class DeepFM(torch.nn.Module):  # 继承 nn.Module，定义共享 Embedding 的 FM 与深度网络
    def __init__(self, numeric_dim, vocab_sizes, embedding_dim=8):  # 接收数值字段数、各类别字段词表大小和每个向量的维度 K
        super().__init__()  # 初始化 nn.Module，让后续赋值的子模块和 Parameter 能被模型管理
        # 一阶部分就是 LR：数值特征做线性变换，类别 ID 查出对应的标量权重。
        self.numeric_linear = torch.nn.Linear(numeric_dim, 1)  # Linear(输入维数, 输出维数)：将 (B, 13) 映射为 (B, 1)，默认含偏置
        # ModuleList 可以像 Python 列表一样遍历，但会把里面的子模块注册到模型中。
        # 因此 model.parameters() 能找到每张表的权重；只存入普通列表不会自动注册。
        # ModuleList 本身不定义前向计算，后面需要通过循环逐张调用这些查找表。
        # Embedding(size, 1) 是一张查找表：输入类别 ID，返回该 ID 对应的一个标量权重。
        # 它等价于取出该字段被激活的 One-Hot 特征的线性系数，无需显式构造 One-Hot。
        # padding_idx=0 表示第 0 行默认初始化为 0，且 Embedding 反向传播不更新这一行。
        self.categorical_linear = torch.nn.ModuleList([  # ModuleList 保存并注册各字段的一阶查找表，方便统一训练
            torch.nn.Embedding(size, 1, padding_idx=0) for size in vocab_sizes  # 每个字段建一张 (size, 1) 的表，每个已知类别对应一个可学习的标量
        ])

        # 每个数值特征对应一个向量；每个类别字段对应一张向量表。
        # Parameter 把普通张量标记为模型的可学习参数，赋给 self 后会出现在 model.parameters() 中。
        # Parameter 默认 requires_grad=True，反向传播时会计算损失对它的梯度。
        # torch.randn(行数, 列数) 从均值为 0、标准差为 1 的正态分布生成随机数；乘 0.01 缩小初始值。
        # FM 的二阶向量使用小随机数初始化，避免全零向量互相相乘时二阶项的梯度也为 0。
        self.numeric_embedding = torch.nn.Parameter(torch.randn(numeric_dim, embedding_dim) * 0.01)  # 注册形状为 (13, K) 的可学习向量矩阵，每个数值特征对应一行
        # Embedding(size, K) 的 weight 形状为 (size, K)：每个类别 ID 对应一行 K 维向量。
        # 输入形状为 (B,) 的整数 ID 后，输出形状为 (B, K)，不是把 ID 数值本身当作连续特征。
        # 每个字段使用独立的表，因此 C1 的 ID 1 与 C2 的 ID 1 可以学到不同的向量。
        self.categorical_embedding = torch.nn.ModuleList([  # 用 ModuleList 保存各类别字段的 K 维向量查找表
            torch.nn.Embedding(size, embedding_dim, padding_idx=0) for size in vocab_sizes  # 每张表形状为 (size, K)，同一字段的不同类别对应不同的向量
        ])

        # init 模块用于设置参数初始值；函数名末尾的下划线表示直接修改原张量。
        # 一阶权重可以从 0 开始学习，其梯度仍由输入特征提供。
        for embedding in self.categorical_linear:  # 依次取出各字段的一阶权重表，设置初始值
            torch.nn.init.zeros_(embedding.weight)  # zeros_ 原地填 0；weight 是该 Embedding 的可学习权重矩阵
        # weight[1:] 只选择 ID 1 及之后的行，跳过保留给未知类别的第 0 行。
        # 这样已知类别得到小随机向量，未知类别仍对应全零向量。
        for embedding in self.categorical_embedding:  # 依次取出各字段的 K 维向量表，设置初始值
            torch.nn.init.normal_(embedding.weight[1:], std=0.01)  # normal_ 用均值为 0、标准差为 0.01 的正态分布填充 ID 1 及之后的行

        # 在 FM 基础上增加 MLP；输入为全部字段的 Embedding 展平结果。
        # Sequential 与 ModuleList 不同：Sequential 会把输入依次传给各层，ModuleList 只负责保存子模块。
        # ReLU 引入非线性；如果多层 Linear 之间没有非线性激活，整体仍可合并成一个线性变换。
        self.deep = torch.nn.Sequential(  # Sequential 按声明顺序执行各层，组成多层感知机 MLP
            torch.nn.Linear((numeric_dim + len(vocab_sizes)) * embedding_dim, 32),  # 将展平的 39 × K 维字段向量映射到 32 个神经元
            torch.nn.ReLU(),  # ReLU 逐元素计算 max(0, x)，引入非线性，张量形状不变
            torch.nn.Linear(32, 16),  # 将上一层的 32 维表示映射为 16 维
            torch.nn.ReLU(),  # ReLU 逐元素计算 max(0, x)，引入非线性，张量形状不变
            torch.nn.Linear(16, 1, bias=False)  # 输出 (B, 1) 的深度分数；bias=False，因为一阶项已提供全局偏置
        )

    # 形状约定：B 是本批样本数，K 是 embedding_dim，当前 K=8。
    # 维度下标从 0 开始：对于 (B, F, K)，dim=0 是样本维，dim=1 是字段维，dim=2 是向量分量维。
    # numeric 中保存标准化后的数值，categorical 中保存每个字段各自的整数类别 ID。
    def forward(self, numeric, categorical):  # 定义前向计算；两个输入的形状分别为 (B, 13) 和 (B, 26)
        linear = self.numeric_linear(numeric)  # 计算数值特征的一阶项及全局偏置，得到 (B, 1)
        for i, embedding in enumerate(self.categorical_linear):  # enumerate 同时返回字段下标 i 和该字段的一阶查找表
            linear = linear + embedding(categorical[:, i])  # [:, i] 取出这一列的 B 个类别 ID；查出 (B, 1) 权重并累加

        # 数值特征使用 x_i * v_i；类别特征的 One-Hot 值为 1，直接查向量。
        # unsqueeze(2) 在下标为 2 的位置插入长度为 1 的维度：(B, 13) -> (B, 13, 1)。
        # 广播机制将 (B, 13, 1) 和 (13, K) 对齐，逐元素相乘得到 (B, 13, K)。
        # 此处 * 是逐元素乘法：第 i 个数值 x_i 乘以自己的 K 维向量 v_i。
        numeric_embeddings = numeric.unsqueeze(2) * self.numeric_embedding  # 先给数值输入插入一个维度，再与各字段向量相乘，得到 (B, 13, K)
        # stack 与 cat 的区别：stack 新建一个维度；cat 沿已有维度拼接，维度数量不变。
        # 这里列表中有 26 个 (B, K) 张量；stack(..., dim=1) 为它们增加一个字段维度。
        categorical_embeddings = torch.stack([  # stack 在新维度上堆叠各字段查出的向量，保留字段之间的区分
            embedding(categorical[:, i])  # 取第 i 个字段的 B 个 ID，查出对应的向量，形状为 (B, K)
            for i, embedding in enumerate(self.categorical_embedding)  # 遍历 26 个字段：i 是字段位置，embedding 是该字段的向量表
        ], dim=1)  # 在 dim=1 插入字段维度，26 个 (B, K) 合成 (B, 26, K)
        embeddings = torch.cat([numeric_embeddings, categorical_embeddings], dim=1)  # cat 沿已有的字段维度拼接，得到 (B, 13 + 26, K)，即 (B, 39, K)

        # FM 二阶项：先求和再平方，减去先平方再求和，最后除以 2，消除重复配对。
        # FM 原始二阶项为 Σ(i<j) <v_i, v_j> x_i x_j；这里只需对加权后的字段向量求和与平方。
        # 对任意两个分量 a、b，有 (a+b)^2 - (a^2+b^2) = 2ab。
        # 推广到全部字段，减法会消除自身平方项，剩下每个不同字段对的交叉项，每对恰好出现两次。
        # 注意下面三处 sum(dim=1) 的含义：前两行的 dim=1 是字段维，最后一行的 dim=1 是 K 维。
        sum_square = embeddings.sum(dim=1).pow(2)  # sum(dim=1) 先对 39 个字段求和；pow(2) 逐元素平方，得到 (B, K)
        square_sum = embeddings.pow(2).sum(dim=1)  # 先对每个向量逐元素平方，再对字段求和，得到 (B, K)
        interaction = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)  # 再对 K 个向量分量求和；keepdim=True 保留输出为 (B, 1)

        # flatten(start_dim=1) 从第 1 维开始展平，保留第 0 维的 B，不会把不同样本混在一起。
        # 这里复用上面计算 FM 的同一个 embeddings，FM 与 MLP 的梯度会共同更新同一组向量。
        deep = self.deep(embeddings.flatten(start_dim=1))  # 保留 Batch 维，将 (B, 39, K) 展平成 (B, 39 × K)，送入 MLP
        return linear + interaction + deep  # 将一阶项、二阶项和 MLP 输出相加，返回最终的 (B, 1) Logit


# ==================== 6. 初始化模型 ====================

torch.manual_seed(42)  # 固定 PyTorch 随机种子，控制参数初始化和 DataLoader 打乱时使用的随机数

with open(data_dir / "metadata.json", encoding="utf-8") as f:  # 以 UTF-8 编码打开预处理配置文件
    metadata = json.load(f)  # json.load 将 JSON 文件解析成 Python 字典

model = DeepFM(  # 创建 DeepFM 模型，同时初始化 FM 和 MLP 的参数
    numeric_dim=metadata["numeric_dim"],  # 数值特征数量，当前 Criteo 数据为 13
    vocab_sizes=metadata["categorical_vocab_sizes"],  # 26 个类别字段各自的词表大小，用于确定每张 Embedding 表的行数
    embedding_dim=8  # 设置向量维度 K=8，控制每个类别或数值字段的向量长度
)


# ==================== 7. 定义损失函数和优化器 ====================

criterion = torch.nn.BCEWithLogitsLoss()  # 二分类交叉熵损失，内部融合 Sigmoid；模型直接输出 Logit
# parameters() 遍历已注册的可学习参数，包括 Linear 权重、Embedding 权重和自定义 Parameter。
# Adam 利用梯度的一阶、二阶滑动平均，自适应调整各参数的更新幅度；训练调用顺序与 LR 相同。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 接收模型的全部可学习参数，lr 是基础学习率


# ==================== 8. 训练并逐轮验证 ====================

num_epochs = 20  # 训练 20 轮；每轮完整遍历一次训练集，再计算验证指标

for epoch in range(num_epochs):  # 遍历训练轮数，epoch 从 0 开始计数
    model.train()  # 切换到训练模式，影响 Dropout、BatchNorm 等模块的行为
    total_train_loss = 0.0  # 重置本轮训练样本的累计损失

    # 与 LR 相比，只是把一个输入拆成了数值特征和类别 ID。
    for numeric_train, categorical_train, label_train in train_loader:  # 依次获取形状为 (B, 13)、(B, 26)、(B, 1) 的训练 Batch
        output_train = model(numeric_train, categorical_train)  # 执行 forward，得到每条样本的 Logit，形状为 (B, 1)
        loss = criterion(output_train, label_train)  # 计算当前 Batch 的平均二分类损失，返回一个标量张量
        optimizer.zero_grad()  # 清空上一次反向传播留下的梯度，避免跨 Batch 累加
        loss.backward()  # 自动反向传播，沿各计算分支求出损失对参数的梯度
        optimizer.step()  # 根据梯度和 Adam 保存的状态，更新模型参数
        total_train_loss += loss.item() * label_train.shape[0]  # item() 取出标量数值，乘本批样本数，累加为损失总和

    avg_train_loss = total_train_loss / len(train_data)  # 除以训练样本总数，正确处理最后一个不足 256 条的 Batch

    model.eval()  # 切换为评估模式；eval() 本身不会关闭梯度计算
    correct = 0  # 重置本轮验证集中预测正确的样本数
    total_val_loss = 0.0  # 重置本轮验证集累计损失
    val_logits = []  # 创建列表，收集各个验证 Batch 的原始预测分数
    val_labels = []  # 创建列表，收集与各个预测分数对应的真实标签

    # eval() 管理层的训练/评估行为，no_grad() 管理是否记录梯度，两者作用不同。
    # 当前网络没有 Dropout、BatchNorm，但仍保留与 LR 一致的训练和验证切换方式。
    with torch.no_grad():  # 关闭此代码块的梯度记录，验证时无需为反向传播保存计算图
        for numeric_val, categorical_val, label_val in val_loader:  # 逐批读取验证集，输入和标签的形状与训练时一致
            logits = model(numeric_val, categorical_val)  # 前向计算验证样本的原始分数，形状为 (B, 1)
            loss = criterion(logits, label_val)  # 计算当前验证 Batch 的平均损失
            total_val_loss += loss.item() * label_val.shape[0]  # 按本批样本数加权累加，保证每条验证样本权重相同
            pred_label = logits >= 0  # 得到布尔预测类别；Logit >= 0 等价于 Sigmoid 概率 >= 0.5
            correct += (pred_label == label_val).sum().item()  # 逐条比较预测与 0/1 标签，sum() 统计正确数，item() 转为 Python 数值
            val_logits.append(logits)  # append 将当前 (B, 1) 分数加入列表，保留各 Batch 顺序
            val_labels.append(label_val)  # 按相同顺序加入标签，确保分数与标签一一对应

    avg_val_loss = total_val_loss / len(val_data)  # 计算整个验证集的平均损失
    accuracy = correct / len(val_data)  # 预测正确数除以样本总数，得到准确率
    # 汇总全部验证样本后计算 AUC，与 LR 的做法相同。
    # 此处的张量位于 CPU，且验证时未记录梯度，因此可调用 numpy() 转成 NumPy 数组。
    # AUC 评价排序，Logit 与 Sigmoid 概率的顺序一致，因此可直接使用原始分数。
    val_auc = roc_auc_score(  # 根据全体验证样本的分数计算 AUC，不对各 Batch 的 AUC 取平均
        torch.cat(val_labels).numpy().ravel(),  # cat 默认沿 dim=0 合并为 (N, 1)，转为 NumPy 后展平成 (N,)
        torch.cat(val_logits).numpy().ravel()  # 以相同方式整理预测分数；二分类 AUC 可直接使用 Logit
    )

    print(  # 输出本轮训练与验证结果，便于观察学习进展和过拟合
        f"Epoch [{epoch + 1}/{num_epochs}], "  # epoch + 1 转成从 1 开始显示的轮数；相邻 f 字符串会自动拼接
        f"Train Loss: {avg_train_loss:.4f}, "  # .4f 表示以小数点后 4 位显示训练损失
        f"Val Loss: {avg_val_loss:.4f}, "  # 以小数点后 4 位显示验证损失
        f"Val AUC: {val_auc:.4f}, "  # 以小数点后 4 位显示验证 AUC
        f"Val Accuracy: {accuracy * 100:.2f}%"  # 准确率乘 100 转为百分比，.2f 保留小数点后 2 位
    )
