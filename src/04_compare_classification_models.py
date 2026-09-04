import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "online_shoppers_classification_data.npz"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MPL_CONFIG_DIR = PROJECT_ROOT / "work" / "matplotlib"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
matplotlib.use("Agg")

import matplotlib.pyplot as plt


RANDOM_SEED = 123
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30


class ClassificationNeuralNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)

    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
    )


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(RANDOM_SEED)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ClassificationNeuralNetwork(input_size=X_train.shape[1])

    positive_count = y_train_tensor.sum()
    negative_count = len(y_train_tensor) - positive_count
    positive_weight = negative_count / positive_count

    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training PyTorch neural network...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f"NN epoch {epoch + 1:02d}/{EPOCHS} - loss: {average_loss:.4f}")

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits).numpy().ravel()
        predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    print("\nTraining decision tree...")

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return predictions, probabilities


def calculate_metrics(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision_revenue": precision_score(y_true, predictions, zero_division=0),
        "recall_revenue": recall_score(y_true, predictions, zero_division=0),
        "f1_revenue": f1_score(y_true, predictions, zero_division=0),
    }


def save_comparison_chart(results: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "classification_model_comparison.png"

    chart_data = results.set_index("model")[["accuracy", "roc_auc", "precision_revenue", "recall_revenue", "f1_revenue"]]

    ax = chart_data.plot(kind="bar", figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Neural Network vs Decision Tree")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nComparison chart saved to: {output_path}")


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    print("Loaded processed classification data")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    y_test_int = y_test.astype(int)

    nn_predictions, nn_probabilities = train_neural_network(X_train, y_train, X_test)
    tree_predictions, tree_probabilities = train_decision_tree(X_train, y_train, X_test)

    results = pd.DataFrame(
        [
            calculate_metrics("PyTorch Neural Network", y_test_int, nn_predictions, nn_probabilities),
            calculate_metrics("Decision Tree", y_test_int, tree_predictions, tree_probabilities),
        ]
    )

    print("\nModel comparison")
    print(results.round(4).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "classification_model_comparison.csv"
    results.to_csv(output_path, index=False)

    print(f"\nComparison table saved to: {output_path}")
    save_comparison_chart(results)


if __name__ == "__main__":
    main()
