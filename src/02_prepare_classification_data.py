from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "Revenue"


def main() -> None:
    data = pd.read_csv(RAW_DATA_PATH)

    print("Step 1: Read the raw data")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)

    print("\nStep 2: Separate features X and target y")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True).rename("proportion"))

    categorical_features = ["Month", "VisitorType"]
    numeric_features = [column for column in X.columns if column not in categorical_features]

    print("\nStep 3: Identify feature types")
    print(f"Numeric/binary features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123,
        stratify=y,
    )

    print("\nStep 4: Split into training and test sets")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nStep 5: Scale numeric features and one-hot encode categorical features")
    print(f"Processed X_train shape: {X_train_processed.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "online_shoppers_classification_data.npz"
    np.savez(
        output_path,
        X_train=X_train_processed.astype(np.float32),
        X_test=X_test_processed.astype(np.float32),
        y_train=y_train.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=np.float32),
    )

    print("\nStep 6: Save processed arrays")
    print(f"Saved to: {output_path}")
    print("\nThis file is ready for the next PyTorch neural network step.")


if __name__ == "__main__":
    main()
