"""Data ingestion, sanitization, and splitting module."""
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path: str = "data/ford.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")

    for col in ["model", "transmission", "fuelType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)]
    print(f"Dropped {before - len(df)} rows with invalid year (e.g. 2060 outlier).")

    before = len(df)
    df = df[df["engineSize"] > 0.0]
    print(f"Dropped {before - len(df)} rows with zero engine displacement.")

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with null values.")

    print(f"Clean dataset: {df.shape[0]:,} rows.\n")
    return df


def get_train_test_data(
    file_path: str = "data/ford.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_and_clean_data(file_path)
    X = df.drop(columns=["price"])
    y = df["price"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
