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

## Interview-Ready Explanation

In this project, I upgraded a traditional decision tree classification workflow into a neural network-based machine learning pipeline using Python and PyTorch. I first inspected the raw dataset, identified `Revenue` as the binary target variable, encoded categorical variables, standardized numerical features, and split the data into training and test sets.

I then trained a feedforward neural network and compared it with a decision tree baseline. The neural network achieved a ROC-AUC of 92.39% and a Revenue recall of 84.55%, while the decision tree achieved a ROC-AUC of 92.98% and a Revenue recall of 84.82%.

Because the positive Revenue class is much smaller than the non-Revenue class, I did not rely only on accuracy. Instead, I focused on ROC-AUC, recall, precision, and F1-score for the Revenue class. This helped me evaluate whether the model could identify high-value sessions rather than simply predicting the majority class.

In this experiment, the Decision Tree achieved the strongest ROC-AUC, and the Decision Tree achieved the strongest Revenue F1-score. The comparison shows that neural networks can learn meaningful behavioral patterns from structured business data, while decision trees remain a strong and interpretable baseline.

## CV Bullet Draft

- Upgraded a decision tree classification project into an end-to-end Python and PyTorch machine learning pipeline, including data inspection, feature preprocessing, train-test split, neural network training, and model evaluation.
- Compared a PyTorch feedforward neural network with a decision tree baseline for online shopping revenue prediction, evaluating model performance using accuracy, ROC-AUC, precision, recall, and F1-score on the test set.
- Addressed class imbalance by emphasizing Revenue-class recall and F1-score rather than relying only on overall accuracy, demonstrating practical evaluation judgment for business classification problems.

## Chinese Explanation

这个项目可以这样理解：原来你只是在 R 里用 decision tree 做分类预测，现在我们把它升级成了一个更完整的 Python + PyTorch 机器学习项目。我们不是只换了一个模型，而是完成了从原始数据检查、变量处理、训练测试集切分、神经网络训练，到和传统模型比较的完整流程。

这个数据集的目标变量是 `Revenue`，也就是一次线上购物访问最后是否产生收入。因为真正产生收入的样本比例比较低，所以这个项目不能只看 accuracy。如果模型全部预测成“不产生收入”，accuracy 也可能看起来不低，但这对业务没有意义。因此我们重点看 ROC-AUC、Revenue recall、Revenue precision 和 Revenue F1-score。

从结果看，decision tree 在这次实验中略强一些，但 PyTorch neural network 已经取得了接近的表现，并且能够识别大部分 Revenue 用户。这是一个很适合面试表达的点：你不仅会训练神经网络，也知道如何把它和 baseline model 比较，并根据业务目标选择合适的评价指标。
