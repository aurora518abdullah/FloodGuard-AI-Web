from __future__ import annotations
from datetime import date
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .config import DATA_PATHS, MODEL_PATH

app = FastAPI(
    title="FloodGuard AI API", version="0.1.0",
    description="Experimental ML prototype. Not an official flood warning service.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.github\.io",
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    station_id: str = Field(..., min_length=1, max_length=80)
    date: date


_cache = {"data": None, "model": None}


def load_dataset() -> pd.DataFrame:
    if _cache["data"] is not None:
        return _cache["data"]
    for path in DATA_PATHS:
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["station_id"] = df["station_id"].astype(str)
            _cache["data"] = df
            return df
    raise HTTPException(status_code=503, detail="Training dataset not installed. Add all_stations.parquet to backend/data/.")


def load_bundle():
    if _cache["model"] is not None:
        return _cache["model"]
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="ML model is not trained yet. Run: python train.py")
    _cache["model"] = joblib.load(MODEL_PATH)
    return _cache["model"]


def risk_band(probability: float) -> tuple[str, str]:
    # Display bands only; these are not calibrated official warning thresholds.
    if probability >= 0.66:
        return "উচ্চ", "উচ্চ স্কোর: স্থানীয় সরকারি সতর্কবার্তা ও নিরাপদ আশ্রয়ের প্রস্তুতি যাচাই করুন।"
    if probability >= 0.33:
        return "মাঝারি", "মাঝারি স্কোর: নদীর পানি ও আবহাওয়ার সরকারি আপডেট নিয়মিত দেখুন।"
    return "কম", "কম স্কোর হলেও ঝুঁকি শূন্য নয়; স্থানীয় নির্দেশনা অনুসরণ করুন।"


@app.get("/")
def root():
    return {"service": "FloodGuard AI API", "status": "running", "warning": "Experimental prototype; not an official warning."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_available": any(path.exists() for path in DATA_PATHS),
        "model_available": MODEL_PATH.exists(),
    }


@app.post("/api/predict-from-data")
def predict(request: PredictionRequest):
    df = load_dataset()
    bundle = load_bundle()
    rows = df[(df["station_id"] == str(request.station_id)) & (df["date"] == request.date)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="এই station_id ও তারিখের জন্য ডেটা পাওয়া যায়নি। ডেটাসেটে থাকা station_id/date ব্যবহার করুন।")
    row = rows.iloc[-1]
    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Dataset/model schema mismatch: {missing[:5]}")
    values = {}
    for column in feature_columns:
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        values[column] = value
    features = pd.DataFrame([values])
    results = {}
    for horizon, model in bundle["models"].items():
        probability = float(model.predict_proba(features)[0, 1])
        level, advice = risk_band(probability)
        results[horizon] = {
            "risk_score": round(probability * 100, 1),
            "risk_level": level,
            "guidance": advice,
        }
    return {
        "station_id": str(request.station_id),
        "date": request.date.isoformat(),
        "prediction": results,
        "model": "HistGradientBoostingClassifier",
        "dataset_period": {
            "from": bundle["dataset_first_date"],
            "to": bundle["dataset_last_date"],
        },
        "evaluation": bundle["metrics"],
        "notice": "পরীক্ষামূলক ML স্কোর; এটি সরকারি সতর্কবার্তা বা নিশ্চিত বন্যার পূর্বাভাস নয়। বাস্তব সিদ্ধান্তে FFWC/DDM-এর নির্দেশনা অনুসরণ করুন।",
    }
