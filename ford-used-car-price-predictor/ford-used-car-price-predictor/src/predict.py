"""Inference: single or batch predictions."""
import os, sys, joblib
import pandas as pd

MODEL_PATH = "artifacts/car_price_pipeline.joblib"
REQUIRED_COLUMNS = ["model","year","transmission","mileage","fuelType","tax","mpg","engineSize"]


def load_pipeline(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run: python src/train.py"
        )
    return joblib.load(model_path)


def predict_single(car: dict, model_path: str = MODEL_PATH) -> float:
    missing = [c for c in REQUIRED_COLUMNS if c not in car]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    pipeline = load_pipeline(model_path)
    return round(float(pipeline.predict(pd.DataFrame([car]))[0]), 2)


def predict_batch(csv_path: str, model_path: str = MODEL_PATH) -> pd.DataFrame:
    pipeline = load_pipeline(model_path)
    df = pd.read_csv(csv_path)
    for col in ["model","transmission","fuelType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["predicted_price"] = pipeline.predict(df[REQUIRED_COLUMNS]).round(2)
    return df


if __name__ == "__main__":
    sample = {
        "model":"Fiesta","year":2018,"transmission":"Manual",
        "mileage":15000,"fuelType":"Petrol","tax":145,"mpg":58.9,"engineSize":1.0,
    }
    price = predict_single(sample)
    print(f"\nPredicted price for sample vehicle: £{price:,.2f}")
