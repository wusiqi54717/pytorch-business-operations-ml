import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "online_shoppers_classification_data.npz"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MPL_CONFIG_DIR = PROJECT_ROOT / "work" / "matplotlib"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

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


def load_processed_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(DATA_PATH)

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[float]:
    losses = []

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
        losses.append(average_loss)

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {average_loss:.4f}")

    return losses


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> None:
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits).numpy().ravel()
        predictions = (probabilities >= 0.5).astype(int)

    y_true = y_test.numpy().ravel().astype(int)

    print("\nTest set evaluation")
    print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions))
    print("\nClassification report:")
    print(classification_report(y_true, predictions, target_names=["No Revenue", "Revenue"]))


def save_loss_curve(losses: list[float]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "classification_nn_loss_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Classification Neural Network Training Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nLoss curve saved to: {output_path}")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test = load_processed_data()

    print("Step 1: Load processed data")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    input_size = X_train.shape[1]
    model = ClassificationNeuralNetwork(input_size=input_size)

    print("\nStep 2: Build the neural network")
    print(model)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    positive_weight = negative_count / positive_count

    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStep 3: Train the neural network")
    losses = train_model(model, train_loader, loss_function, optimizer)

    print("\nStep 4: Evaluate the neural network")
    evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "online_shoppers_classification_nn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    save_loss_curve(losses)


if __name__ == "__main__":
    main()
