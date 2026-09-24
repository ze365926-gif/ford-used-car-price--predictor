"""Model training, evaluation, and pipeline export."""
import os
import sys
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from data_prep import get_train_test_data

NUMERIC_FEATURES  = ["year", "mileage", "tax", "mpg", "engineSize"]
CATEGORICAL_FEATURES = ["model", "transmission", "fuelType"]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150, max_depth=20,
            min_samples_split=5, random_state=42, n_jobs=-1
        )),
    ])


def train_and_evaluate(
    data_path: str = "data/ford.csv",
    output_path: str = "artifacts/car_price_pipeline.joblib",
    metrics_path: str = "artifacts/metrics.json",
):
    print("=" * 52)
    print("  Ford Car Price Predictor — Training Pipeline  ")
    print("=" * 52 + "\n")

    print("Step 1/3 — Loading and preparing data...")
    X_train, X_test, y_train, y_test = get_train_test_data(data_path)
    print(f"Train samples : {len(X_train):,}  |  Test samples : {len(X_test):,}\n")

    pipeline = build_pipeline()
    print("Step 2/3 — Fitting model pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    print("Step 3/3 — Evaluating on holdout test set...")
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 52)
    print(f"  R² Score:                {r2:.4f}")
    print(f"  Mean Absolute Error:     £{mae:,.2f}")
    print(f"  Root Mean Squared Error: £{rmse:,.2f}")
    print("=" * 52 + "\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"✅  Pipeline saved  → {output_path}")

    metrics = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅  Metrics saved  → {metrics_path}\n")
    return metrics


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/ford.csv"
    train_and_evaluate(data_path=data_path)
