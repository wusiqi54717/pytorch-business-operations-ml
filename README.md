# Predictive Modeling & Neural Networks for Business Operations

这个项目用于把原来的 R 语言业务预测分析扩展成 Python + PyTorch 的机器学习项目。

当前阶段已经完成一个完整的 classification workflow：

1. 数据检查
2. 数据预处理
3. PyTorch neural network 训练
4. Decision tree baseline 对比
5. 自动生成简历/面试 summary

## 已配置的 Python 环境

正式使用的环境是：

```powershell
.\.venv312\Scripts\python.exe
```

这个环境来自 Anaconda 的 Python 3.12.7。我们没有继续使用系统 Python 3.13.0，因为第一次安装后 PyTorch 在 Windows 上加载 DLL 失败。对初学项目来说，选择更稳定的 Python 版本更重要。

## VS Code 里应该怎么选解释器

在 VS Code 中按：

```text
Ctrl + Shift + P
```

然后搜索并选择：

```text
Python: Select Interpreter
```

再选择这个路径：

```text
.\.venv312\Scripts\python.exe
```

如果之后打开 notebook，也要在右上角 kernel 位置选择同一个 `.venv312` 环境。

## 如何验证环境

在 VS Code 终端中进入本项目文件夹后运行：

```powershell
.\.venv312\Scripts\python.exe src\check_environment.py
```

如果看到 PyTorch 版本、CUDA 状态和 tensor 计算结果，说明环境已经可以使用。

## 文件夹用途

```text
data/raw/          放原始数据，例如 online_shoppers_intention.csv
data/processed/    放清洗、编码、标准化后的数据
notebooks/         放适合学习和展示的分步 notebook
src/               放可重复运行的 Python 脚本
reports/figures/   放模型表现图、loss 曲线、评价结果图
```

## 当前分类任务

当前主要数据集是：

```text
data/raw/online_shoppers_intention.csv
```

目标变量是：

```text
Revenue
```

这个任务是一个 binary classification problem，目标是预测一次 online shopping session 是否会产生收入。

## 运行顺序

```powershell
.\.venv312\Scripts\python.exe src\01_inspect_data.py
.\.venv312\Scripts\python.exe src\02_prepare_classification_data.py
.\.venv312\Scripts\python.exe src\03_train_classification_nn.py
.\.venv312\Scripts\python.exe src\04_compare_classification_models.py
.\.venv312\Scripts\python.exe src\05_generate_model_comparison_summary.py
```

## 当前硬件判断

当前没有检测到 NVIDIA GPU/CUDA，所以安装的是 CPU 版 PyTorch：

```text
torch==2.14.0+cpu
```

这对我们接下来做表格数据的分类神经网络和回归神经网络已经足够。
