from pathlib import Path
from datetime import date
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "demand_forecasting.csv"
MODEL_PATH = BASE / "demand_forecasting_model.pkl"
METRICS_PATH = BASE / "model_metrics.json"
PREDS_PATH = BASE / "test_predictions.csv"
TARGET = "Demand"
CAT = ["Store ID", "Product ID", "Category", "Region"]
NUM = ["Inventory Level", "Price", "Discount", "Promotion", "Competitor Pricing", "Epidemic"]
CAL = ["dayofweek", "month", "dayofyear"]
LAGS = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7"]
FEATURES = CAT + NUM + CAL + LAGS

app = FastAPI(title="Retail Demand Forecasting API", version="1.0.0")
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

class ForecastRequest(BaseModel):
    store_id: str
    category: str
    year: int = Field(ge=2024, le=2035)
    month: int = Field(ge=1, le=12)


def load_df():
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")
    d = pd.read_csv(DATA_PATH)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", "Demand"])
    for c in CAT:
        d[c] = d[c].astype(str)
    return d.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

DF = load_df()


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python train_model.py inside backend folder.")
    return joblib.load(MODEL_PATH)


def build_product_forecast(model, history_df, future_dates):
    pair = history_df.sort_values("Date")
    if len(pair) < 28:
        return None
    meta = pair.iloc[-1]
    demand_history = pair[TARGET].astype(float).tolist()
    rows = []
    for dt in future_dates:
        row = {
            "Store ID": str(meta["Store ID"]), "Product ID": str(meta["Product ID"]),
            "Category": str(meta["Category"]), "Region": str(meta["Region"]),
            "Inventory Level": float(meta["Inventory Level"]), "Price": float(meta["Price"]),
            "Discount": float(meta["Discount"]), "Promotion": float(meta["Promotion"]),
            "Competitor Pricing": float(meta["Competitor Pricing"]), "Epidemic": float(meta["Epidemic"]),
            "dayofweek": dt.dayofweek, "month": dt.month, "dayofyear": dt.dayofyear,
            "lag_1": demand_history[-1], "lag_7": demand_history[-7],
            "lag_14": demand_history[-14], "lag_28": demand_history[-28],
            "rolling_mean_7": float(np.mean(demand_history[-7:]))
        }
        value = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        value = int(round(value))
        rows.append({"Date": dt.strftime("%Y-%m-%d"), "Product ID": str(meta["Product ID"]), "Predicted Demand": value})
        demand_history.append(float(value))
    return rows

@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_rows": int(len(DF)), "model_ready": MODEL_PATH.exists()}

@app.get("/api/summary")
def summary():
    daily = DF.groupby("Date", as_index=False)[TARGET].sum()
    category = DF.groupby("Category", as_index=False)[TARGET].sum().sort_values(TARGET, ascending=False)
    top = DF.groupby("Product ID", as_index=False)[TARGET].sum().nlargest(10, TARGET)
    return {
        "rows": int(len(DF)), "stores": int(DF["Store ID"].nunique()), "products": int(DF["Product ID"].nunique()),
        "average_demand": round(float(DF[TARGET].mean()), 2), "total_demand": round(float(DF[TARGET].sum()), 2),
        "date_start": DF["Date"].min().strftime("%Y-%m-%d"), "date_end": DF["Date"].max().strftime("%Y-%m-%d"),
        "daily": [{"date": r.Date.strftime("%Y-%m-%d"), "demand": round(float(r.Demand), 2)} for r in daily.itertuples(index=False)],
        "categories": [{"category": str(r.Category), "demand": round(float(r.Demand), 2)} for r in category.itertuples(index=False)],
        "top_products": [{"product_id": str(row["Product ID"]), "demand": round(float(row[TARGET]), 2)} for _, row in top.iterrows()]
    }

@app.get("/api/options")
def options():
    stores = sorted(DF["Store ID"].unique().tolist())
    categories = sorted(DF["Category"].unique().tolist())
    return {"stores": stores, "categories": categories}

@app.get("/api/evaluation")
def evaluation():
    if not METRICS_PATH.exists():
        return {"available": False, "message": "Run python train_model.py in the backend folder first."}
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    result = {"available": True, **metrics}
    if PREDS_PATH.exists():
        pred = pd.read_csv(PREDS_PATH, parse_dates=["Date"])
        daily = pred.groupby("Date", as_index=False)[["Demand", "Predicted_Demand"]].sum()
        result["actual_predicted"] = [{"date": r.Date.strftime("%Y-%m-%d"), "actual": round(float(r.Demand), 2), "predicted": round(float(r.Predicted_Demand), 2)} for r in daily.itertuples(index=False)]
    else:
        result["actual_predicted"] = []
    return result

@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    model = get_model()
    store = str(req.store_id)
    category = str(req.category)
    try:
        start = pd.Timestamp(year=req.year, month=req.month, day=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    next_month = start + pd.offsets.MonthBegin(1)
    future_dates = list(pd.date_range(start, next_month - pd.Timedelta(days=1), freq="D"))
    selected = DF[(DF["Store ID"] == store) & (DF["Category"] == category)]
    if selected.empty:
        raise HTTPException(status_code=404, detail="No products found for this store/category.")
    # Do not use future actual demand as history; use only observations before selected forecast month.
    all_rows = []
    for product in sorted(selected["Product ID"].unique()):
        history = DF[(DF["Store ID"] == store) & (DF["Product ID"] == product) & (DF["Date"] < start)]
        if len(history) < 28:
            continue
        rows = build_product_forecast(model, history, future_dates)
        if rows:
            all_rows.extend(rows)
    if not all_rows:
        raise HTTPException(status_code=400, detail="Not enough historical data before the selected month. Choose a later month/year.")
    raw = pd.DataFrame(all_rows)
    daily = raw.groupby("Date", as_index=False)["Predicted Demand"].sum()
    daily.insert(1, "Store ID", store)
    daily.insert(2, "Category", category)
    return {
        "store_id": store, "category": category, "year": req.year, "month": req.month,
        "product_count": int(raw["Product ID"].nunique()), "total_demand": int(daily["Predicted Demand"].sum()),
        "average_daily_demand": round(float(daily["Predicted Demand"].mean()), 2),
        "daily": daily.to_dict(orient="records"), "by_product": raw.to_dict(orient="records"),
        "assumption": "Future non-demand features are held at each product's latest values before the forecast month. Multi-day forecasts are recursive estimates."
    }
