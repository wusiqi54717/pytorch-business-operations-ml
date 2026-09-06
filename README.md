# Predictive Modeling & Neural Networks for Business Operations

这个项目用于把原来的 R 语言业务预测分析扩展成 Python + PyTorch 的机器学习项目。

当前阶段已经完成一个完整的 classification workflow：

1. 数据检查
2. 数据预处理
3. PyTorch neural network 训练
4. Decision tree baseline 对比

## 已配置的 Python 环境

正式使用的环境是：

```powershell
.\.venv312\Scripts\python.exe
```

这个环境来自 Anaconda 的 Python 3.12.7。

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

# Classification Model Comparison Summary

## Project Context

This project extends a traditional decision tree classification workflow into a Python and PyTorch-based machine learning pipeline. The business objective is to predict whether an online shopping session will generate revenue, using behavioral, traffic, visitor, and page interaction features.

## Models Compared

Two supervised classification models were trained and evaluated on the same processed dataset:

- Decision Tree Classifier
- PyTorch Feedforward Neural Network

Both models used the same train-test split and the same preprocessed input features. Numeric features were standardized, categorical features were one-hot encoded, and the target variable `Revenue` was converted into a binary label.

## Test Set Results

| Model | Accuracy | ROC-AUC | Revenue Precision | Revenue Recall | Revenue F1 |
|---|---:|---:|---:|---:|---:|
| PyTorch Neural Network | 83.21% | 92.39% | 47.64% | 84.55% | 60.94% |
| Decision Tree | 86.13% | 92.98% | 53.29% | 84.82% | 65.45% |

## Explanation

In this project, I upgraded a traditional decision tree classification workflow into a neural network-based machine learning pipeline using Python and PyTorch. I first inspected the raw dataset, identified `Revenue` as the binary target variable, encoded categorical variables, standardized numerical features, and split the data into training and test sets.

I then trained a feedforward neural network and compared it with a decision tree baseline. The neural network achieved a ROC-AUC of 92.39% and a Revenue recall of 84.55%, while the decision tree achieved a ROC-AUC of 92.98% and a Revenue recall of 84.82%.

Because the positive Revenue class is much smaller than the non-Revenue class, I did not rely only on accuracy. Instead, I focused on ROC-AUC, recall, precision, and F1-score for the Revenue class. This helped me evaluate whether the model could identify high-value sessions rather than simply predicting the majority class.

In this experiment, the Decision Tree achieved the strongest ROC-AUC, and the Decision Tree achieved the strongest Revenue F1-score. The comparison shows that neural networks can learn meaningful behavioral patterns from structured business data, while decision trees remain a strong and interpretable baseline.

## 中文Explanation

原来只是在R里用 decision tree 做分类预测，现在升级成了一个更完整的 Python + PyTorch 机器学习项目。不是只换了一个模型，而是完成了从原始数据检查、变量处理、训练测试集切分、神经网络训练，到和传统模型比较的完整流程。

这个数据集的目标变量是 `Revenue`，也就是一次线上购物访问最后是否产生收入。因为真正产生收入的样本比例比较低，所以这个项目不能只看 accuracy。如果模型全部预测成“不产生收入”，accuracy 也可能看起来不低，但这对业务没有意义。因此我们重点看 ROC-AUC、Revenue recall、Revenue precision 和 Revenue F1-score。

从结果看，decision tree 在这次实验中略强一些，但 PyTorch neural network 已经取得了接近的表现，并且能够识别大部分 Revenue 用户。这是一个很适合面试表达的点：你不仅会训练神经网络，也知道如何把它和 baseline model 比较，并根据业务目标选择合适的评价指标。

