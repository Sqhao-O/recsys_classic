# 经典推荐模型学习

使用带中文注释的独立脚本学习 LR、FM、DeepFM、Wide & Deep 和 DIN。模型说明、数据结构与运行流程见 [src/README.md](src/README.md)。

## 环境

建议使用 Python 3.10 或更新版本，安装依赖：

```bash
pip install -r requirements.txt
```

使用 NVIDIA GPU 时，按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与服务器驱动相容的 CUDA 版 PyTorch，再确认：

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

## RetailRocket 与 DIN

从 [RetailRocket 官方数据页面](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)自行取得数据，遵守其使用条款。将以下文件放入 `src/datasets/RetailRocket/`：

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

从仓库根目录运行：

```bash
python src/preprocess_retailrocket.py
python src/din.py
```

预处理保留高类别覆盖窗口内全部有效训练、验证样本对，窗口前的浏览用于积累历史。DIN 同时使用商品 ID 和类别特征，加入 Embedding L2，自动使用 CUDA；CUDA 不可用时使用 CPU。默认 Batch 大小为 1024，可在 `src/din.py` 中调整。

## 检查

```bash
python -m unittest discover -s tests -v
```

数据集、预处理产物、模型权重、日志、本机配置和凭据不随代码发布。仓库不包含数据下载凭据；自行准备数据后再执行预处理和训练。
