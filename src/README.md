# 从 LR 到 FM、DeepFM、Wide & Deep

建议按 `lr.py → fm.py → deepfm.py → wide_deep.py` 的顺序阅读。每个脚本都能独立运行，保留与 LR 相同的 8 个步骤；三个新脚本的数据集和训练循环相同，重点对照第 5 部分的模型定义。

| 文件 | 在 LR 基础上新增的内容 | 最终输出的 Logit |
| --- | --- | --- |
| `lr.py` | 数值特征和 One-Hot 特征的线性组合 | 一阶项（含偏置） |
| `fm.py` | 用 Embedding 计算两两特征交互 | 一阶项 + 二阶项 |
| `deepfm.py` | 在 FM 上增加 MLP，两条分支共享 Embedding | 一阶项 + 二阶项 + MLP 输出 |
| `wide_deep.py` | Wide 使用 LR 和人工交叉，Deep 使用 Embedding 和 MLP | Wide 输出 + Deep 输出 |

## 运行

在装有 NumPy、PyTorch、SciPy、pandas、scikit-learn、joblib 的 Python 环境中，从项目根目录执行：

```bash
# 已有 processed 数据时，可以直接运行模型。
python src/fm.py
python src/deepfm.py
python src/wide_deep.py
```

首次准备数据时，先运行 `python src/preprocess_criteo.py`。三个新模型直接读取现有的 `train_common.npz`、`val_common.npz` 和 `metadata.json`，不需要重新生成一种数据格式。

## 与 LR 相比，数据输入改了什么

LR 读取拼接好的稀疏特征，再按行转成稠密张量；三个新模型直接读取通用数据：

| 每个 batch 的输入 | 当前形状 | 类型 |
| --- | --- | --- |
| `numeric` | `(B, 13)` | `float32` |
| `categorical` | `(B, 26)` | `long`，从保存的整数 ID 转换一次 |
| `label` | `(B, 1)` | `float32`，与 LR 相同 |

`nn.Embedding` 就是一张可学习的查找表：`Embedding(size, 1)` 查到某个类别的一阶权重，`Embedding(size, 8)` 查到一个 8 维向量。类别 ID 本身没有大小关系，不直接当连续数值输入 Linear。

每个类别字段都有自己的词表，因此使用 `ModuleList` 保存 26 张表。C1 的 ID 1 和 C2 的 ID 1 查询的是不同的表。`padding_idx=0` 将未知类别的向量固定为 0，与 LR 中未知类别 One-Hot 全零的处理一致；缺失值先在预处理中替换为 `__MISSING__`，再参与类别编码。

## FM：新增一个二阶项

```text
linear = 数值特征的线性变换 + 每个类别对应的一阶权重
二阶项 = Σ(i < j) 内积(v_i, v_j) × x_i × x_j
logit = linear + 二阶项
```

对数值特征，要先计算 `x_i * v_i`；对类别特征，激活的 One-Hot 值是 1，直接查出 `v_i` 即可。拼接后的 `embeddings` 形状是 `(B, 39, 8)`，包含 13 个数值字段和 26 个类别字段。

代码用下面的等价公式计算二阶项，避免显式写两层配对循环：

```python
sum_square = embeddings.sum(dim=1).pow(2)
square_sum = embeddings.pow(2).sum(dim=1)
interaction = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
```

先求和再平方会同时包含自身项和交叉项；减去各向量的平方后，只剩交叉项；每对出现两次，所以再除以 2。

## DeepFM：在 FM 上增加 MLP

`deepfm.py` 保留了 FM 的全部计算，仅增加一个 `32 → 16 → 1` 的 MLP，把同一个 `embeddings` 展平后送进去。FM 二阶项和 MLP 使用同一组向量，两部分的梯度一起更新这些向量，这就是“共享 Embedding”。

## Wide & Deep：在线性分支旁边增加 MLP

Wide 部分在 LR 一阶项上增加一个 `C1 × C2` 的交叉权重，Deep 部分把 13 个数值特征和 26 个类别向量拼接后送进 MLP。

交叉 ID 的计算是 `C1_id * C2_vocab_size + C2_id`。例如 C2 词表大小为 10，`(C1_id=2, C2_id=3)` 的交叉 ID 就是 23；不同组合查询不同的标量权重。C1、C2 只是演示人工交叉的例子，不表示已验证它们在业务上有特殊关联。

交叉权重从 0 开始学习，因此训练中没有出现过的组合保持为 0。Deep 分支通过类别向量和 MLP 学习组合关系。

## 训练时继续沿用 LR 的知识

- 三个模型都输出 `(B, 1)` 的 Logit，并使用 `BCEWithLogitsLoss`，不用在模型末尾加 Sigmoid。
- 前向传播、清空梯度、反向传播、参数更新、逐轮验证，与 LR 相同。
- 新模型统一使用 `Adam(lr=0.001)`；相较 LR 的 SGD，只替换优化器这一行，训练步骤不变。
- FM 向量用小随机数初始化；一阶类别权重从 0 开始学习。MLP 最后一层不再添加偏置，全局偏置由数值线性层提供。
- 每轮都输出 Train Loss、Val Loss、Val AUC、Val Accuracy；验证损失按样本数加权，AUC 在整个验证集上计算。
- 默认 Embedding 维度为 8、训练 20 轮。当前预处理只读取 2000 条样本，先观察每轮验证指标，再调整 `num_epochs`；小样本下复杂模型可能过拟合，不能据此确定模型优劣。

## 原始资料

- [FM 作者的 libFM 页面与论文入口](https://libfm.org/)
- [DeepFM 原始论文](https://ijcai.org/proceedings/2017/239)
- [Wide & Deep 原始论文](https://arxiv.org/abs/1606.07792)

## DIN：从固定字段转向用户历史行为

接着阅读 `preprocess_retailrocket.py → din.py`。前者负责生成序列样本，后者延续已有的 Dataset、DataLoader、Embedding、MLP、损失函数和训练循环写法。新增的核心是：根据**当前候选商品**，给用户的每条过去浏览计算权重，再汇总为兴趣向量。

### 先处理 RetailRocket，再训练

原始数据使用项目已有的 `src/datasets/RetailRocket/events.csv`。本例只读取行为表的四列，不需要商品属性表和类别树：

| 字段 | 含义 | 本例用途 |
| --- | --- | --- |
| `timestamp` | 毫秒时间戳 | 排序、划分数据、确定历史范围 |
| `visitorid` | 用户 ID | 将同一用户的浏览串起来，不输入模型 |
| `event` | 浏览、加购或购买 | 只保留 `view` 作为点击行为 |
| `itemid` | 原始商品 ID | 编码为从 1 开始的整数，供 Embedding 查表 |

```bash
python src/preprocess_retailrocket.py
python src/din.py
```

预处理先读取四列、清洗重复浏览并按时间排序，默认选取最早约 20 万次浏览；边界同一毫秒的事件完整保留。可以在预处理脚本末尾将 `max_events` 改为 `None` 使用全部数据，将 `max_history_len` 改为其他历史长度。模型默认向量维度为 8、训练 20 轮，参数位置与前面的模型相近。

### 一条浏览如何变成两条样本

假设用户过去依次浏览过 `[鞋子, 袜子]`，现在浏览了运动鞋：

```text
候选商品：运动鞋    历史：[鞋子, 袜子, 0, ...]    标签：1
候选商品：马克杯    历史：[鞋子, 袜子, 0, ...]    标签：0
```

运动鞋这次浏览要在生成样本**之后**才加入历史；否则模型会提前看到要预测的行为。历史只保留最近 L 次浏览，按从旧到新排列，右侧补 0；重复浏览会保留。首次浏览只积累历史，从有历史的后续浏览开始生成样本。

每条正样本配一条均匀采样的负样本，候选池只包含当前时刻之前已经出现过的不同商品，并排除该用户全部过去和当前同时浏览的商品。没有可选负样本时跳过整对。同一毫秒没有可靠的先后顺序，所以先生成这一毫秒的全部样本，再统一更新历史与候选池。

RetailRocket 提供的是交互日志，没有完整的“曝光但未点击”记录。因此，这里的 0 表示**采样的未观察交互**，可能包含用户未来会浏览的商品；它不是真实曝光负反馈。AUC、Accuracy 和输出的 Sigmoid 分数都基于这个采样任务，不能直接解释为线上 CTR 效果或真实点击概率。

### 时间划分和输出格式

约前 80% 的浏览用于训练，其余用于验证，同一毫秒不会跨越两边。商品词表只在训练区间建立，验证期新商品的目标和历史都跳过；这意味着本例评估的是**已知商品、有已知浏览历史**的样本。控制台和 `metadata.json` 会报告跳过的浏览数量。

验证采用逐次预测：较早的真实验证浏览可以进入后续验证样本的历史，但模型参数只用训练集更新。这适用于每次交互后更新历史的场景。训练 DataLoader 的打乱只改变取样顺序，不会改变已经构造好的历史。

处理结果保存在 `src/datasets/RetailRocket/processed/`，与 Criteo 数据分开：

| 文件/字段 | 内容 |
| --- | --- |
| `train_common.npz`、`val_common.npz` | 训练集和验证集 |
| `target_ids` | `(N,)`，候选商品 ID |
| `history_ids` | `(N, L)`，历史商品 ID，0 为补齐 |
| `labels` | `(N, 1)`，float32 的 0/1 标签 |
| `user_ids`、`timestamps` | 核对历史和时间划分，不输入模型 |
| `metadata.json` | 词表大小、历史长度、划分时间及样本统计 |
| `item_mapping.json` | 原始商品 ID 到连续 ID 的映射，JSON 的键为字符串 |

### DIN 中的注意力在算什么

候选商品和历史商品共享一张 Embedding 表。设候选向量是 `q`、某条历史向量是 `h`：

```text
单条历史的权重 a = attention_mlp([q, h, q-h, q*h])
当前候选对应的兴趣向量 = Σ a × h（只对真实历史求和）
logit = prediction_mlp([q, 兴趣向量, q*兴趣向量])
```

例如，推荐运动鞋时，过去浏览的鞋袜可能获得较高权重；候选换成咖啡机时，同一段历史会重新计算权重。这种权重由训练学习得到，不是手工设定。`get_interest()` 将这一步单独写出，并注释了 `(B, L, K)` 各维度、扩展、拼接和加权求和。

本例使用非归一化加权和，权重可为负，和也不必为 1；不使用 Softmax，将兴趣强度保留在汇总结果中。补齐位置通过 Mask 将权重置 0，空历史得到全零兴趣向量。DIN 本身没有加入位置编码或循环网络，因此虽然历史按时间截取，注意力汇总不显式建模行为之间的顺序关系。

这是保留 DIN 核心机制的学习版：激活使用简洁的 PReLU，没有加入 Dice、mini-batch aware 正则或其他特征，不是论文全部实验设置的复现。模型输出 `(B, 1)` Logit，继续使用 `BCEWithLogitsLoss`、Adam，以及与 DeepFM 相同的整体验证 AUC 和按样本数加权的 Loss。

本地用默认 20 万次浏览试跑时，第 1、2 轮验证 AUC 约为 0.759、0.766；之后训练 Loss 继续下降，验证 Loss 却明显上升，出现过拟合。保留默认 20 轮便于对照已有脚本观察这个过程；实际选择轮数时应查看验证指标，可以先把 `main(num_epochs=20)` 的默认值改为 2，或尝试减小网络。不同环境下结果可能略有差异。

可以运行 `python -m unittest discover -s tests -v`，检查时间划分、负采样、历史截断和注意力的补齐屏蔽。

- [RetailRocket 数据集说明](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [DIN 原始论文](https://arxiv.org/abs/1706.06978)
- [作者公开实现](https://github.com/zhougr1993/DeepInterestNetwork)
