
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "demand_forecasting.csv"
MODEL_PATH = BASE / "demand_forecasting_model.pkl"
METRICS_PATH = BASE / "model_metrics.json"
PREDS_PATH = BASE / "test_predictions.csv"

TARGET = "Demand"

CAT = ["Store ID", "Product ID", "Category", "Region"]

NUM = [
    "Inventory Level",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
]

CAL = ["dayofweek", "month", "dayofyear"]

LAGS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
]

FEATURES = CAT + NUM + CAL + LAGS


# --------------------------------------------------
# Product ID to Product Name mapping
# --------------------------------------------------
# Replace these example names with your actual product names.
# If the CSV contains a "Product Name" column, it is used
# automatically instead of this mapping.

PRODUCT_NAMES = {
    # "P0001": "Wireless Headphones",
    # "P0002": "Smartphone",
    # "P0003": "Laptop",
    # "P0004": "Bluetooth Speaker",
    # "P0005": "Smart Watch",
    # "P0006": "Keyboard",
    # "P0007": "Mouse",
    # "P0008": "Power Bank",
    # "P0009": "USB Cable",
    # "P0010": "Monitor",
}


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://retail-demand-forecasting-frontend-r.vercel.app",
        "https://retail-demand-forecasting-frontend-3.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class ForecastRequest(BaseModel):
    store_id: str
    category: str
    year: int = Field(ge=2024, le=2035)
    month: int = Field(ge=1, le=12)


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

def load_df():
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET])

    for col in CAT:
        df[col] = df[col].astype(str)

    # Normalize optional product names.
    if "Product Name" in df.columns:
        df["Product Name"] = df["Product Name"].fillna("").astype(str).str.strip()

    return df.sort_values(
        ["Date", "Store ID", "Product ID"]
    ).reset_index(drop=True)


DF = load_df()


# --------------------------------------------------
# Product name helper
# --------------------------------------------------

def get_product_name(product_id, product_name=None):
    """
    Return the actual product name when available.
    Priority:
    1. Product Name from the dataset
    2. PRODUCT_NAMES mapping
    3. Product ID as a fallback
    """

    if product_name is not None:
        name = str(product_name).strip()

        if name and name.lower() != "nan":
            return name

    return PRODUCT_NAMES.get(str(product_id), str(product_id))


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not trained. Run python train_model.py "
                "inside the backend folder."
            ),
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Product-wise forecasting
# --------------------------------------------------

def build_product_forecast(model, history_df, future_dates):
    pair = history_df.sort_values("Date")

    if len(pair) < 28:
        return None

    meta = pair.iloc[-1]

    product_id = str(meta["Product ID"])

    product_name = get_product_name(
        product_id,
        meta.get("Product Name"),
    )

    demand_history = pair[TARGET].astype(float).tolist()

    rows = []

    for dt in future_dates:
        row = {
            "Store ID": str(meta["Store ID"]),
            "Product ID": product_id,
            "Category": str(meta["Category"]),
            "Region": str(meta["Region"]),

            "Inventory Level": float(meta["Inventory Level"]),
            "Price": float(meta["Price"]),
            "Discount": float(meta["Discount"]),
            "Promotion": float(meta["Promotion"]),
            "Competitor Pricing": float(meta["Competitor Pricing"]),
            "Epidemic": float(meta["Epidemic"]),

            "dayofweek": dt.dayofweek,
            "month": dt.month,
            "dayofyear": dt.dayofyear,

            "lag_1": demand_history[-1],
            "lag_7": demand_history[-7],
            "lag_14": demand_history[-14],
            "lag_28": demand_history[-28],

            "rolling_mean_7": float(np.mean(demand_history[-7:])),
        }

        prediction = model.predict(
            pd.DataFrame([row], columns=FEATURES)
        )[0]

        prediction = max(0.0, float(prediction))
        prediction = int(round(prediction))

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Product ID": product_id,
            "Product Name": product_name,
            "Predicted Demand": prediction,
        })

        # Recursive forecasting
        demand_history.append(float(prediction))

    return rows


# --------------------------------------------------
# Health API
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_rows": int(len(DF)),
        "model_ready": MODEL_PATH.exists(),
    }


# --------------------------------------------------
# Summary API
# --------------------------------------------------

@app.get("/api/summary")
def summary():
    daily = (
        DF.groupby("Date", as_index=False)[TARGET]
        .sum()
        .sort_values("Date")
    )

    category = (
        DF.groupby("Category", as_index=False)[TARGET]
        .sum()
        .sort_values(TARGET, ascending=False)
    )

    # Aggregate demand by product ID.
    top = (
        DF.groupby("Product ID", as_index=False)[TARGET]
        .sum()
        .nlargest(10, TARGET)
    )

    # Build a product ID -> name lookup from the dataset.
    dataset_names = {}

    if "Product Name" in DF.columns:
        named_rows = DF[
            DF["Product Name"].fillna("").astype(str).str.strip() != ""
        ]

        dataset_names = (
            named_rows
            .drop_duplicates("Product ID")
            .set_index("Product ID")["Product Name"]
            .to_dict()
        )

    top_products = []

    for _, row in top.iterrows():
        product_id = str(row["Product ID"])

        product_name = get_product_name(
            product_id,
            dataset_names.get(product_id),
        )

        top_products.append({
            "product_id": product_id,
            "product_name": product_name,
            "demand": round(float(row[TARGET]), 2),
        })

    return {
        "rows": int(len(DF)),
        "stores": int(DF["Store ID"].nunique()),
        "products": int(DF["Product ID"].nunique()),

        "average_demand": round(float(DF[TARGET].mean()), 2),
        "total_demand": round(float(DF[TARGET].sum()), 2),

        "date_start": DF["Date"].min().strftime("%Y-%m-%d"),
        "date_end": DF["Date"].max().strftime("%Y-%m-%d"),

        "daily": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "demand": round(float(row.Demand), 2),
            }
            for row in daily.itertuples(index=False)
        ],

        "categories": [
            {
                "category": str(row.Category),
                "demand": round(float(row.Demand), 2),
            }
            for row in category.itertuples(index=False)
        ],

        "top_products": top_products,
    }


# --------------------------------------------------
# Dropdown options API
# --------------------------------------------------

@app.get("/api/options")
def options():
    stores = sorted(DF["Store ID"].unique().tolist())
    categories = sorted(DF["Category"].unique().tolist())

    return {
        "stores": stores,
        "categories": categories,
    }


# --------------------------------------------------
# Model evaluation API
# --------------------------------------------------

@app.get("/api/evaluation")
def evaluation():
    if not METRICS_PATH.exists():
        return {
            "available": False,
            "message": (
                "Run python train_model.py in the backend folder first."
            ),
        }

    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    result = {
        "available": True,
        **metrics,
    }

    if PREDS_PATH.exists():
        pred = pd.read_csv(
            PREDS_PATH,
            parse_dates=["Date"],
        )

        daily = (
            pred.groupby("Date", as_index=False)[
                ["Demand", "Predicted_Demand"]
            ]
            .sum()
            .sort_values("Date")
        )

        result["actual_predicted"] = [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "actual": round(float(row.Demand), 2),
                "predicted": round(float(row.Predicted_Demand), 2),
            }
            for row in daily.itertuples(index=False)
        ]

    else:
        result["actual_predicted"] = []

    return result


# --------------------------------------------------
# Forecast API
# --------------------------------------------------

@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    model = get_model()

    store = str(req.store_id)
    category = str(req.category)

    try:
        start = pd.Timestamp(
            year=req.year,
            month=req.month,
            day=1,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    next_month = start + pd.offsets.MonthBegin(1)

    future_dates = list(
        pd.date_range(
            start,
            next_month - pd.Timedelta(days=1),
            freq="D",
        )
    )

    selected = DF[
        (DF["Store ID"] == store)
        & (DF["Category"] == category)
    ]

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail="No products found for this store/category.",
        )

    # Use only history before the forecast month.
    all_rows = []

    for product in sorted(selected["Product ID"].unique()):
        history = DF[
            (DF["Store ID"] == store)
            & (DF["Product ID"] == product)
            & (DF["Date"] < start)
        ]

        if len(history) < 28:
            continue

        rows = build_product_forecast(
            model,
            history,
            future_dates,
        )

        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough historical data before the selected month. "
                "Choose a later month/year."
            ),
        )

    raw = pd.DataFrame(all_rows)

    daily = (
        raw.groupby("Date", as_index=False)["Predicted Demand"]
        .sum()
        .sort_values("Date")
    )

    daily.insert(1, "Store ID", store)
    daily.insert(2, "Category", category)

    return {
        "store_id": store,
        "category": category,
        "year": req.year,
        "month": req.month,

        "product_count": int(raw["Product ID"].nunique()),

        "total_demand": int(daily["Predicted Demand"].sum()),

        "average_daily_demand": round(
            float(daily["Predicted Demand"].mean()),
            2,
        ),

        "daily": daily.to_dict(orient="records"),

        "by_product": raw.to_dict(orient="records"),

        "assumption": (
            "Future non-demand features are held at each product's "
            "latest values before the forecast month. "
            "Multi-day forecasts are recursive estimates."
        ),
    }
