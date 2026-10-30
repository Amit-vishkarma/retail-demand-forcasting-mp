
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "demand_forecasting.csv"
MODEL = BASE / "demand_forecasting_model.pkl"
METRICS = BASE / "model_metrics.json"
PREDS = BASE / "test_predictions.csv"

TARGET = "Demand"

CAT = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
]

NUM = [
    "Inventory Level",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
]

CAL = [
    "dayofweek",
    "month",
    "dayofyear",
]

LAGS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
]

FEATURES = CAT + NUM + CAL + LAGS


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def add_features(df):
    x = df.copy()

    x["Date"] = pd.to_datetime(
        x["Date"], errors="coerce"
    )

    x = x.dropna(subset=["Date", TARGET])

    # Keep the history in chronological order.
    x = x.sort_values(
        ["Store ID", "Product ID", "Date"]
    ).copy()

    x["dayofweek"] = x["Date"].dt.dayofweek
    x["month"] = x["Date"].dt.month
    x["dayofyear"] = x["Date"].dt.dayofyear

    grouped = x.groupby(
        ["Store ID", "Product ID"]
    )[TARGET]

    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = grouped.shift(lag)

    x["rolling_mean_7"] = (
        x.groupby(["Store ID", "Product ID"])[TARGET]
        .transform(
            lambda s: s.shift(1).rolling(
                window=7,
                min_periods=7
            ).mean()
        )
    )

    return (
        x.sort_values(["Date", "Store ID", "Product ID"])
        .reset_index(drop=True)
    )


# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------


def make_pipeline():
    prep = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CAT
        ),
        (
            "num",
            SimpleImputer(strategy="median"),
            NUM + CAL + LAGS
        )
    ])

    est = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )

    return Pipeline([
        ("preprocessing", prep),
        ("model", est)
    ])


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    print("Loading dataset...")

    df = pd.read_csv(DATA)

    required_columns = set(
        ["Date", TARGET] + CAT + NUM
    )

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df = df.dropna(subset=["Date", TARGET])

    # Use consistent categorical types.
    for col in CAT:
        df[col] = df[col].astype(str)

    df = df.sort_values(
        ["Date", "Store ID", "Product ID"]
    ).reset_index(drop=True)

    print(f"Dataset rows: {len(df):,}")
    print("Creating lag features...")

    data = add_features(df)

    # Drop rows without enough historical information.
    data = data.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    dates = np.sort(data["Date"].unique())

    if len(dates) < 2:
        raise ValueError(
            "Not enough distinct dates for chronological evaluation."
        )

    cutoff_index = int(len(dates) * 0.8)
    cutoff_index = min(
        max(cutoff_index, 1),
        len(dates) - 1,
    )

    cutoff = dates[cutoff_index]

    train = data[data["Date"] < cutoff]
    test = data[data["Date"] >= cutoff]

    if train.empty or test.empty:
        raise ValueError(
            "Training or test data is empty. Check dataset dates."
        )

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print("Training optimized Extra Trees model...")

    model = make_pipeline()

    model.fit(
        train[FEATURES],
        train[TARGET],
    )

    print("Generating test predictions...")

    predictions = np.maximum(
        0,
        model.predict(test[FEATURES]),
    )

    actual = test[TARGET].to_numpy(dtype=float)

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    r2 = r2_score(actual, predictions)

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predictions).sum()
        / denominator
        * 100
        if denominator
        else None
    )

    metrics = {
        "model": "Extra Trees Regressor",
        "test_start": str(
            test["Date"].min().date()
        ),
        "test_end": str(
            test["Date"].max().date()
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "wape_percent": (
            float(wape) if wape is not None else None
        ),
        "note": (
            "Chronological holdout evaluation. "
            "Lag features use prior observed demand. "
            "Production forecasts use recursive estimates "
            "for future demand."
        ),
    }

    # Save evaluation predictions in the same format
    # expected by main.py.
    output = test[
        ["Date", "Store ID", "Product ID", TARGET]
    ].copy()

    output["Predicted_Demand"] = predictions

    print("Saving model and evaluation files...")

    joblib.dump(
        model,
        MODEL,
        compress=3,
    )

    METRICS.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    output.to_csv(
        PREDS,
        index=False,
    )

    print("\nMODEL EVALUATION RESULTS")
    print("-" * 35)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    if wape is not None:
        print(f"WAPE: {wape:.2f}%")

    print("-" * 35)
    print(f"Model saved: {MODEL}")
    print(
        f"Model file size: "
        f"{MODEL.stat().st_size / (1024 ** 2):.2f} MB"
    )
    print(f"Metrics saved: {METRICS}")
    print(f"Predictions saved: {PREDS}")


if __name__ == "__main__":
    main()

