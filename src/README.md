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

原始数据使用项目已有的 `src/datasets/RetailRocket/events.csv`，读取行为表的四列：

| 字段 | 含义 | 本例用途 |
| --- | --- | --- |
| `timestamp` | 毫秒时间戳 | 排序、划分数据、确定历史范围 |
| `visitorid` | 用户 ID | 将同一用户的浏览串起来，不输入模型 |
| `event` | 浏览、加购或购买 | 只保留 `view` 作为点击行为 |
| `itemid` | 原始商品 ID | 编码为从 1 开始的整数，供 Embedding 查表 |

另外分块读取 `item_properties_part1.csv`、`item_properties_part2.csv`，只保留 `property=categoryid` 的类别记录。当前不使用类别树或其他商品属性。清洗和排序使用 DataFrame，生成样本时用列表收集，最后仍以 NumPy 数组保存到 NPZ；这延续 Criteo 的模型输入格式。

```bash
python src/preprocess_retailrocket.py
python src/din.py
```

预处理读取完整时间线，清洗重复浏览并按时间排序。固定从北京时间 `2015-07-02 00:00:00` 开始生成样本，窗口之前的浏览仍用于积累历史。训练和验证都保留全部有效正负样本，代码不再提供限量抽样分支。模型默认每张表的向量维度为 8、训练 20 轮，Batch 大小为 1024。

### 先选窗口，默认保留全部有效样本

1. 在 `preprocess_events` 的参数默认值中固定 `sample_start_date="2015-07-02"`，按北京时间零点转换为样本起点。修改这个日期即可调整窗口，不再计算每日类别覆盖率或自动选择起点。
2. 从日志最早时间开始累积历史、已浏览集合和负采样商品池；窗口之前只预热，不生成样本。新用户首次出现仍可能没有历史，这次浏览继续跳过，但会为后续积累历史。当前训练和评估不覆盖空历史用户。
3. 以窗口内约前 80%、后 20% 的浏览确定训练/验证边界，同一毫秒不拆开。窗口持续到日志结束，不按类别是否缺失剔除单条浏览。
4. 每个有效浏览配一个负商品，直接依次保存正样本和负样本。样本按照浏览时间生成，不再随机筛选、替换或重新排序。

```text
完整时间线 →
仅预热历史和商品池 | 训练区间（全部有效正负样本对） | 验证区间（全部有效正负样本对）
                 样本起点                       时间边界
```

不再生成 `daily_category_coverage.csv`；旧运行遗留的同名文件不代表当前预处理结果。

固定日期沿用此前扫描得到的样本起点：北京时间 `2015-07-02 00:00:00`，训练/验证边界为 `2015-09-01 03:54:19.447`，窗口结束于 `2015-09-18 10:59:47.788`；起点前有 1,171,729 次浏览用于预热。相同数据和起点下，预计保留训练 559,162 对（1,118,324 行）、验证 129,629 对（259,258 行）。重新运行预处理后，以控制台和 `metadata.json` 的实际统计为准。

### CUDA 运行

`din.py` 自动使用可用的 CUDA，否则使用 CPU。模型和每个 Batch 的候选、类别、历史、标签会移动到同一设备；验证结果转回 CPU 后计算 AUC。CUDA 下 DataLoader 启用页锁定内存和非阻塞传输，数据本身仍保存在 CPU 内存中。默认使用 float32，不引入混合精度。

在安装了支持 CUDA 的 PyTorch 的服务器上，从仓库根目录执行：

```bash
python -c 'import torch; print(torch.cuda.is_available())'
python src/preprocess_retailrocket.py
python src/din.py
```

第一条命令应输出 `True`，训练启动时还会打印 GPU 名称、Batch 大小和样本数量。预处理使用 pandas/NumPy，在 CPU 上执行。需要调整 Batch 大小时可修改 `main(num_epochs=20, batch_size=1024)` 的默认值，或执行 `python -c 'from src.din import main; main(batch_size=4096)'`。增加 Batch 大小会改变每轮更新次数，需要重新观察训练指标。

### 一条浏览如何变成两条样本

假设用户过去依次浏览过 `[鞋子, 袜子]`，现在浏览了运动鞋：

```text
候选商品：运动鞋    历史：[鞋子, 袜子, 0, ...]    标签：1
候选商品：马克杯    历史：[鞋子, 袜子, 0, ...]    标签：0
```

运动鞋这次浏览要在生成样本**之后**才加入历史；否则模型会提前看到要预测的行为。历史只保留最近 L 次浏览，按从旧到新排列，右侧补 0；重复浏览会保留。首次浏览只积累历史，从有历史的后续浏览开始生成样本。

每个候选还保存自己的类别，每个历史位置也保存浏览当时的类别。商品属性会随时间变化：候选（包括负商品）使用不晚于预测时刻的最近类别记录，历史使用不晚于当次浏览时刻的最近类别记录；后来的类别变化不会改写旧历史。没有过去属性时不使用未来记录回填，而是记为未知类别 1。类别词表只根据训练区间之前及区间内可用的已知商品属性建立，已知类别从 2 开始编号，0 留给序列补齐。

脚本分别报告保留样本中正候选、负候选、真实历史位置的类别已知比例；正负样本共用的历史只统计一次，补齐位置不参与统计。缺失还区分为“当时没有类别记录”和“类别可查但不在训练词表”。这些统计仅供检查，不控制样本起点；不会为了提高比例剔除单条未知类别样本或用未来属性回填。

每条正样本配一条均匀采样的负样本，候选池只包含当前时刻之前已经出现过的不同商品，并排除该用户全部过去和当前同时浏览的商品。没有可选负样本时跳过整对。同一毫秒没有可靠的先后顺序，所以先生成这一毫秒的全部样本，再统一更新历史与候选池。

RetailRocket 提供的是交互日志，没有完整的“曝光但未点击”记录。因此，这里的 0 表示**采样的未观察交互**，可能包含用户未来会浏览的商品；它不是真实曝光负反馈。AUC、Accuracy 和输出的 Sigmoid 分数都基于这个采样任务，不能直接解释为线上 CTR 效果或真实点击概率。

### 时间划分和输出格式

窗口内约前 80% 的浏览用于训练，其余用于验证，同一毫秒不会跨越两边。商品和类别词表使用训练截止时间之前可用的信息，包含预热数据；验证期新商品的目标和历史都跳过。这意味着本例评估的是**已知商品、有已知浏览历史**的样本。控制台和 `metadata.json` 会报告窗口内跳过的浏览数量，窗口前的浏览另记为预热。

验证采用逐次预测：较早的真实验证浏览可以进入后续验证样本的历史，但模型参数只用训练集更新。这适用于每次交互后更新历史的场景。训练 DataLoader 的打乱只改变取样顺序，不会改变已经构造好的历史。

处理结果保存在 `src/datasets/RetailRocket/processed/`，与 Criteo 数据分开：

| 文件/字段 | 内容 |
| --- | --- |
| `train_common.npz`、`val_common.npz` | 训练集和验证集 |
| `target_ids` | `(N,)`，候选商品 ID |
| `target_category_ids` | `(N,)`，候选商品在预测时刻的类别 ID |
| `history_ids` | `(N, L)`，历史商品 ID，0 为补齐 |
| `history_category_ids` | `(N, L)`，每次历史浏览当时的类别 ID，与历史商品逐位置对应 |
| `labels` | `(N, 1)`，float32 的 0/1 标签 |
| `user_ids`、`timestamps` | 核对历史和时间划分，不输入模型 |
| `metadata.json` | 商品/类别词表大小、固定样本起点、时间边界、预热数量、样本数及类别统计 |
| `item_mapping.json` | 原始商品 ID 到连续 ID 的映射，JSON 的键为字符串 |
| `category_mapping.json` | 原始类别 ID 到连续 ID 的映射，0 为补齐、1 为未知 |

加入类别后，需要重新运行预处理生成这两个新数组，再运行 DIN。`RetailRocketDataset` 每条样本返回 `(候选商品, 候选类别, 历史商品, 历史类别, 标签)`。

### DIN 中的注意力在算什么

商品 ID 和类别 ID 各有一张 Embedding 表，候选与历史共享这两张表。先将商品向量与类别向量拼接，再交给注意力；默认各 8 维，拼接后为 16 维。设候选向量是 `q`、某条历史向量是 `h`：

```text
q = concat(候选商品 Embedding, 候选类别 Embedding)
h = concat(历史商品 Embedding, 历史类别 Embedding)
单条历史的权重 a = attention_mlp([q, h, q-h, q*h])
当前候选对应的兴趣向量 = Σ a × h（只对真实历史求和）
logit = prediction_mlp([q, 兴趣向量, q*兴趣向量])
```

例如，推荐运动鞋时，过去浏览的鞋袜可能获得较高权重；候选换成咖啡机时，同一段历史会重新计算权重。这种权重由训练学习得到，不是手工设定。`get_interest()` 将这一步单独写出，并注释了 `(B, L, K)` 各维度、扩展、拼接和加权求和。

本例使用非归一化加权和，权重可为负，和也不必为 1；不使用 Softmax，将兴趣强度保留在汇总结果中。补齐位置根据 `history_ids != 0` 判断，通过 Mask 将权重置 0；未知类别 1 仍然对应真实浏览，会参与注意力。空历史得到全零兴趣向量。DIN 本身没有加入位置编码或循环网络，因此虽然历史按时间截取，注意力汇总不显式建模行为之间的顺序关系。

这是使用商品 ID 和类别特征、保留 DIN 核心机制的学习版：激活使用简洁的 PReLU，没有加入 Dice、mini-batch aware 正则或店铺、用户画像等特征，不是论文全部实验设置的复现。模型输出 `(B, 1)` Logit，继续使用 `BCEWithLogitsLoss`、Adam，以及与 DeepFM 相同的整体验证 AUC 和按样本数加权的 Loss。

### Embedding L2 正则化

`din.py` 使用普通 L2，系数在 `main()` 中设置为 `l2_lambda=1e-4`，训练目标为：

```text
训练目标 = BCE + l2_lambda × (商品 Embedding 平方和 + 类别 Embedding 平方和)
```

这会限制商品、类别向量的大小，减轻对稀疏 ID 的记忆。补齐行 0 不参与，未知类别 1 参与；MLP 不加这一项。这是普通 Embedding L2，不是论文按特征频次加权的 mini-batch aware 正则。

`Train Loss` 和 `Val Loss` 始终只报告 BCE；`Reg Loss` 单独报告已经乘过系数的正则项，反向传播时两项相加。验证阶段不加入正则项。

```bash
# 20 轮，Embedding L2 系数 1e-4；首次切换全量样本前需重新运行预处理
python src/din.py
```

此前限量阶段（训练 8 万对、验证 2 万对、Batch 大小 256、种子 42）的正则化训练，最好验证 AUC 为 0.9031，最低验证 BCE 为 0.4676；这些不是当前全量配置的结果。当前代码未运行全量训练或验证 CUDA 性能，需在目标服务器上实测。模型尚未加入早停，应根据验证指标判断合适轮次。

可以运行 `python -m unittest discover -s tests -v`，检查连续日期选窗、窗口前历史预热、全部样本保留与复现、时间划分、类别随时间变化，以及注意力对两种特征的使用和补齐屏蔽。小数据测试使用较低的覆盖率阈值，并单独验证默认连续三天的规则。

关于低频商品过滤、训练样本支持度，以及公开 RetailRocket 会话处理方法如何用于 DIN，参见 [预处理研究建议](RETAILROCKET_PREPROCESSING.md)。这些建议尚未应用到当前生成的数据中。

- [RetailRocket 数据集说明](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [DIN 原始论文](https://arxiv.org/abs/1706.06978)
- [作者公开实现](https://github.com/zhougr1993/DeepInterestNetwork)
