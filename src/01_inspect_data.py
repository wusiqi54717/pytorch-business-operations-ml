from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_csv(file_name: str) -> None:
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        print(f"\n[Missing] {file_path}")
        print("Please put this CSV file into data/raw first.")
        return

    data = pd.read_csv(file_path)

    print(f"\nFile: {file_name}")
    print(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print("\nColumn names:")
    print(list(data.columns))
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nMissing values by column:")
    print(data.isna().sum())
    print("\nData types:")
    print(data.dtypes)


def main() -> None:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return

    for csv_file in csv_files:
        inspect_csv(csv_file.name)


if __name__ == "__main__":
    main()
