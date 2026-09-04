from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = PROJECT_ROOT / "reports" / "classification_model_comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "classification_model_comparison_summary.md"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    results = pd.read_csv(COMPARISON_PATH)
    nn = results.loc[results["model"] == "PyTorch Neural Network"].iloc[0]
    tree = results.loc[results["model"] == "Decision Tree"].iloc[0]

    summary = f"""# Classification Model Comparison Summary

## Project Context

This project upgrades a traditional decision tree classification workflow into a Python and PyTorch machine learning pipeline. The goal is to predict whether an online shopping session will generate revenue.

## Test Set Results

| Model | Accuracy | ROC-AUC | Revenue Precision | Revenue Recall | Revenue F1 |
|---|---:|---:|---:|---:|---:|
| PyTorch Neural Network | {pct(nn["accuracy"])} | {pct(nn["roc_auc"])} | {pct(nn["precision_revenue"])} | {pct(nn["recall_revenue"])} | {pct(nn["f1_revenue"])} |
| Decision Tree | {pct(tree["accuracy"])} | {pct(tree["roc_auc"])} | {pct(tree["precision_revenue"])} | {pct(tree["recall_revenue"])} | {pct(tree["f1_revenue"])} |

## Interview Explanation

I built an end-to-end classification workflow using Python and PyTorch. I inspected the raw dataset, selected `Revenue` as the binary target variable, transformed categorical variables with one-hot encoding, standardized numerical features, and split the data into training and test sets.

I trained a feedforward neural network and compared it with a decision tree baseline. Because the positive Revenue class is much smaller than the non-Revenue class, I evaluated the models with ROC-AUC, precision, recall, and F1-score instead of relying only on accuracy.

The neural network achieved a ROC-AUC of {pct(nn["roc_auc"])} and Revenue recall of {pct(nn["recall_revenue"])}. The decision tree achieved a ROC-AUC of {pct(tree["roc_auc"])} and Revenue recall of {pct(tree["recall_revenue"])}. This comparison shows that the neural network learned useful behavioral patterns, while the decision tree remained a strong interpretable baseline.

## CV Bullet Draft

- Built an end-to-end Python and PyTorch classification pipeline for online shopping revenue prediction, including data inspection, preprocessing, neural network training, and model evaluation.
- Compared a PyTorch feedforward neural network with a decision tree baseline using accuracy, ROC-AUC, precision, recall, and F1-score on the test set.
- Addressed class imbalance by emphasizing Revenue-class recall and F1-score, demonstrating practical evaluation judgment for business classification problems.
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
