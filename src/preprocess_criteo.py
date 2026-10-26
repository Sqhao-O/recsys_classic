import pandas as pd  # 导入 Pandas，用于读取和处理数据表
import numpy as np  # 导入 NumPy，用于数值计算
import joblib  # 导入 joblib，用于保存预处理模型
import json  # 导入 json，用于保存数据集的配置信息

from pathlib import Path  # 导入 Path，用于处理文件路径
from scipy import sparse  # 导入 SciPy 稀疏矩阵模块
from sklearn.model_selection import train_test_split  # 导入数据集划分方法
from sklearn.preprocessing import StandardScaler  # 导入数值标准化工具
from sklearn.preprocessing import OneHotEncoder  # 导入 One-Hot 类别编码工具

# ==================== 1. 设置文件路径 ====================

base_dir = Path(__file__).resolve().parent  # 获取当前 Python 文件所在目录
filepath = base_dir / "datasets/Criteo/raw/train.txt"  # 设置 Criteo 原始训练数据路径
save_dir = base_dir / "datasets/Criteo/processed"  # 设置预处理结果保存目录
save_dir.mkdir(parents=True, exist_ok=True)  # 如果保存目录不存在，就自动创建

# ==================== 2. 定义数据列名 ====================

numeric_cols = [f"I{i}" for i in range(1, 14)]  # 生成 13 个数值特征列名(colunms)，等号后是生成了包含 13 个字符串的列表
categorical_cols = [f"C{i}" for i in range(1, 27)]  # 生成 26 个类别特征列名
columns = ["label"] + numeric_cols + categorical_cols  # 按照原始数据顺序定义 40 个字段

# ==================== 3. 读取原始数据 ====================

df = pd.read_csv( # 该方法能读取结构化的纯文本文件
    filepath,  # 指定数据文件路径
    sep="\t",  # 指定使用制表符分隔
    header=None,  # 原始数据没有表头
    names=columns,  # 使用自己定义的列名
    nrows=2000,  # 暂时读取 1000 条数据
    dtype={  # 指定各特征的数据类型
        **{col: "float32" for col in numeric_cols},  # 数值特征读取为 float32
        **{col: "string" for col in categorical_cols}  # 类别特征读取为字符串
    }
)

print("原始数据形状：", df.shape)  # 打印数据维度
print("原始缺失值数量：", df.isnull().sum().sum())  # 统计所有缺失值

# ==================== 4. 划分训练集和验证集 ====================

train_df, val_df = train_test_split(
    df,  # 需要划分的数据集
    test_size=0.2,  # 20% 作为验证集
    random_state=42,  # 固定随机种子
    stratify=df["label"]  # 按照点击标签比例进行分层抽样(随机可能导致严重不均衡)
)

train_df = train_df.copy()  # 创建训练集独立副本
val_df = val_df.copy()  # 创建验证集独立副本

# ==================== 5. 处理缺失值 ====================

train_df[numeric_cols] = train_df[numeric_cols].fillna(0)  # 训练集数值特征缺失值填 0
val_df[numeric_cols] = val_df[numeric_cols].fillna(0)  # 验证集采用相同的填充规则
train_df[categorical_cols] = train_df[categorical_cols].fillna("__MISSING__")  # 类别缺失值替换成特殊字符串
val_df[categorical_cols] = val_df[categorical_cols].fillna("__MISSING__")  # 验证集采用相同规则

# ==================== 6. 数值特征非线性变换 ====================

train_df[numeric_cols] = np.arcsinh(
    train_df[numeric_cols]  # 对训练集数值特征进行反双曲正弦变换
).astype(np.float32)  # 保持 float32 类型，变换会变类型为 float64

val_df[numeric_cols] = np.arcsinh(
    val_df[numeric_cols]  # 对验证集使用相同的非线性变换
).astype(np.float32)  # 保持 float32 类型

# ==================== 7. 数值特征标准化 ====================

scaler = StandardScaler()  # 创建标准化器

train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols]).astype(np.float32)  # scaler对象学习训练数据的统计特征，再进行变换
val_df[numeric_cols] = scaler.transform(val_df[numeric_cols]).astype(np.float32)  # 直接用训练集的统计特征进行变换，保证变换相同

joblib.dump(
    scaler,  # 需要保存的标准化器
    save_dir / "scaler.pkl"  # 指定保存路径
)

# ==================== 8. 类别特征 One-Hot 编码 ====================

encoder = OneHotEncoder(
    handle_unknown="ignore",  # 遇到训练集中没见过的类别时不报错
    dtype=np.float32,  # 编码后的数值类型为 float32
    sparse_output=True  # 输出稀疏矩阵，避免存储大量的零
)

train_cat = encoder.fit_transform(train_df[categorical_cols].astype(str))  # 在训练集上encoder对象学习类别集合并编码，返回稀疏矩阵变量
val_cat = encoder.transform(val_df[categorical_cols].astype(str))  # 使用相同的类别集合编码验证集

joblib.dump(
    encoder,  # 保存已经学习好类别映射的编码器
    save_dir / "onehot_encoder.pkl"  # 指定保存路径
)

print("One-Hot 训练特征形状：", train_cat.shape)  # 查看编码后训练集的维度
print("One-Hot 验证特征形状：", val_cat.shape)  # 查看编码后验证集的维度

# ==================== 9. 生成类别 ID ====================

# 这一部分为后续 FM、DeepFM 等模型准备整数类别 ID

train_cat_ids = []  # 创建列表，用于保存训练集各字段的类别 ID

val_cat_ids = []  # 创建列表，用于保存验证集各字段的类别 ID

vocab_sizes = []  # 创建列表，记录每个类别字段的词表大小

for i, col in enumerate(categorical_cols):  # 逐个处理 26 个类别字段

    categories = encoder.categories_[i]  # 获取 OneHotEncoder 为当前字段学习到的类别集合

    mapping = {value: idx + 1 
               for idx, value in enumerate(categories)}  # 为已知类别分配从 1 开始的 ID

    train_ids = train_df[col].astype(str).map(mapping).fillna(0).astype(np.int32)  # 编码训练集，未知类别记为 0

    val_ids = val_df[col].astype(str).map(mapping).fillna(0).astype(np.int32)  # 编码验证集，未知类别记为 0

    train_cat_ids.append(train_ids.to_numpy())  # 将当前训练字段的 ID 保存到列表

    val_cat_ids.append(val_ids.to_numpy())  # 将当前验证字段的 ID 保存到列表

    vocab_sizes.append(len(categories) + 1)  # 记录词表大小，额外包含未知类别 ID 0

train_cat_ids = np.column_stack(train_cat_ids).astype(np.int32)  # 将 26 列 ID 拼接为训练集矩阵

val_cat_ids = np.column_stack(val_cat_ids).astype(np.int32)  # 将 26 列 ID 拼接为验证集矩阵

print("整数编码训练集形状：", train_cat_ids.shape)  # 检查训练集类别 ID 维度

print("整数编码验证集形状：", val_cat_ids.shape)  # 检查验证集类别 ID 维度


# ==================== 10. 提取数值特征 ====================

train_numeric = train_df[numeric_cols].to_numpy(dtype=np.float32)  # 提取训练集数值特征

val_numeric = val_df[numeric_cols].to_numpy(dtype=np.float32)  # 提取验证集数值特征


# ==================== 11. 提取标签 ====================

train_labels = train_df["label"].to_numpy(dtype=np.float32).reshape(-1, 1)  # 提取训练标签，保持二维

val_labels = val_df["label"].to_numpy(dtype=np.float32).reshape(-1, 1)  # 提取验证标签，保持二维


# ==================== 12. 保存通用数据集 ====================

np.savez_compressed(
    save_dir / "train_common.npz",  # 训练集保存路径
    numeric=train_numeric,  # 保存标准化后的数值特征
    categorical_ids=train_cat_ids,  # 保存类别 ID
    labels=train_labels  # 保存点击标签
)

np.savez_compressed(
    save_dir / "val_common.npz",  # 验证集保存路径
    numeric=val_numeric,  # 保存标准化后的数值特征
    categorical_ids=val_cat_ids,  # 保存类别 ID
    labels=val_labels  # 保存点击标签
)


# ==================== 13. 构建 LR 专用稀疏特征 ====================

train_numeric_sparse = sparse.csr_matrix(train_numeric)  # 将训练集数值特征转换为 CSR 稀疏矩阵

val_numeric_sparse = sparse.csr_matrix(val_numeric)  # 将验证集数值特征转换为 CSR 稀疏矩阵

train_lr = sparse.hstack(
    [train_numeric_sparse, train_cat],  # 横向拼接数值特征和 One-Hot 类别特征
    format="csr"  # 指定输出为 CSR 稀疏矩阵
)

val_lr = sparse.hstack(
    [val_numeric_sparse, val_cat],  # 验证集使用完全相同的特征顺序
    format="csr"  # 指定输出为 CSR 稀疏矩阵
)

sparse.save_npz(
    save_dir / "train_lr.npz",  # 设置训练集 LR 特征的保存路径
    train_lr  # 保存训练集稀疏特征矩阵
)

sparse.save_npz(
    save_dir / "val_lr.npz",  # 设置验证集 LR 特征的保存路径
    val_lr  # 保存验证集稀疏特征矩阵
)

# ==================== 14. 保存特征配置信息 ====================

metadata = {  # 创建记录特征结构和编码规则的字典
    "numeric_cols": numeric_cols,  # 记录数值字段顺序
    "categorical_cols": categorical_cols,  # 记录类别字段顺序
    "numeric_dim": len(numeric_cols),  # 记录数值特征维度
    "categorical_vocab_sizes": vocab_sizes,  # 记录每个类别字段的词表大小
    "lr_input_dim": train_lr.shape[1],  # 记录 LR 的总输入维度
    "unknown_category_id": 0,  # 记录未知类别的 ID
    "numeric_transform": "arcsinh_then_standard_scaler"  # 记录数值特征的处理流程
}

with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:  # 创建并打开配置文件
    json.dump(metadata, f, ensure_ascii=False, indent=4)  # 将配置信息保存为 JSON

# ==================== 15. 检查最终结果 ====================

print("训练集数值特征：", train_numeric.shape)  # 输出训练集数值特征维度

print("训练集类别 ID:", train_cat_ids.shape)  # 输出训练集类别 ID 维度

print("训练集标签：", train_labels.shape)  # 输出训练标签维度

print("LR 训练特征：", train_lr.shape)  # 输出 LR 训练特征维度

print("LR 验证特征：", val_lr.shape)  # 输出 LR 验证特征维度

print("LR 训练特征非零元素数:", train_lr.nnz)  # 统计稀疏矩阵中实际存储的非零元素

print("Criteo 数据预处理完成！")  # 输出完成提示