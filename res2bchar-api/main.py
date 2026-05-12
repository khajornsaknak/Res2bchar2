"""
Res2bChar ML Prediction API
FastAPI server สำหรับ predict biochar properties ด้วย 5 ML models
Deploy บน Render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Res2bChar ML API",
    description="Predict biochar properties from pyrolysis conditions & feedstock",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Google Apps Script, Render, localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# โหลด models ตอน startup (ครั้งเดียว)
# ─────────────────────────────────────────────
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

MODEL_FILES = {
    "Yield_char":  "Yield_char__CatBoost.joblib",
    "C_char":      "C_char__HistGradientBoosting.joblib",
    "H_C_char":    "H_C_char__GradientBoosting.joblib",
    "O_C_char":    "O_C_char__CatBoost.joblib",
    "FC_char":     "FC_char__GradientBoosting.joblib",
}

META_FILES = {
    "Yield_char":  "Yield_char__CatBoost__meta.json",
    "C_char":      "C_char__HistGradientBoosting__meta.json",
    "H_C_char":    "H_C_char__GradientBoosting__meta.json",
    "O_C_char":    "O_C_char__CatBoost__meta.json",
    "FC_char":     "FC_char__GradientBoosting__meta.json",
}

models = {}
metas  = {}

@app.on_event("startup")
def load_models():
    for key, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"⚠️  Model not found: {path}")
            continue
        models[key] = joblib.load(path)
        print(f"✅ Loaded model: {key}")

    for key, filename in META_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                metas[key] = json.load(f)


# ─────────────────────────────────────────────
# Input schema
# ─────────────────────────────────────────────
class PredictInput(BaseModel):
    # Pyrolysis conditions
    Temp: float = Field(..., description="Temperature (°C)", example=500)
    RT:   float = Field(..., description="Residence time (min)", example=60)
    HR:   float = Field(..., description="Heating rate (°C/min)", example=10)

    # Proximate analysis
    VM_bio:  float = Field(0.0, description="Volatile matter (%)", example=70.0)
    Ash_bio: float = Field(0.0, description="Ash content (%)", example=5.0)
    FC_bio:  float = Field(0.0, description="Fixed carbon (%)", example=15.0)

    # Ultimate analysis
    C_bio: float = Field(0.0, description="Carbon in feedstock (%)", example=45.0)
    H_bio: float = Field(0.0, description="Hydrogen in feedstock (%)", example=6.0)
    N_bio: float = Field(0.0, description="Nitrogen in feedstock (%)", example=1.0)
    O_bio: float = Field(0.0, description="Oxygen in feedstock (%)", example=40.0)

    @validator("Temp")
    def temp_positive(cls, v):
        if v <= 0:
            raise ValueError("Temp must be > 0")
        return v


# ─────────────────────────────────────────────
# Feature engineering (derived features)
# ─────────────────────────────────────────────
def build_features(inp: PredictInput) -> dict:
    """
    คำนวณ derived features ให้ครบตามที่แต่ละโมเดลต้องการ
    """
    # ป้องกัน division by zero
    C_bio   = inp.C_bio   if inp.C_bio   > 0 else 1e-6
    Ash_bio = inp.Ash_bio if inp.Ash_bio > 0 else 1e-6
    H_bio   = inp.H_bio

    H_C_bio = H_bio / C_bio * 12           # H/C atomic ratio  (H%/C% × 12/1)
    O_C_bio = inp.O_bio / C_bio * 12 / 16  # O/C atomic ratio  (O%/C% × 12/16)

    VM_to_Ash  = inp.VM_bio  / Ash_bio
    FC_to_Ash  = inp.FC_bio  / Ash_bio

    # FC_bio_derived = 100 - VM_bio - Ash_bio  (ถ้าไม่ได้วัด FC โดยตรง)
    FC_bio_derived = max(0.0, 100.0 - inp.VM_bio - inp.Ash_bio)

    # O_bio_derived  = 100 - C - H - N - Ash  (by difference)
    O_bio_derived  = max(0.0, 100.0 - inp.C_bio - inp.H_bio - inp.N_bio - inp.Ash_bio)

    return {
        "Temp":           inp.Temp,
        "RT":             inp.RT,
        "HR":             inp.HR,
        "VM_bio":         inp.VM_bio,
        "Ash_bio":        inp.Ash_bio,
        "FC_bio":         inp.FC_bio,
        "C_bio":          inp.C_bio,
        "H_bio":          inp.H_bio,
        "N_bio":          inp.N_bio,
        "O_bio":          inp.O_bio,
        "H/C_bio":        H_C_bio,
        "O/C_bio":        O_C_bio,
        "VM_to_Ash":      VM_to_Ash,
        "FC_to_Ash":      FC_to_Ash,
        "FC_bio_derived": FC_bio_derived,
        "O_bio_derived":  O_bio_derived,
    }


def predict_one(model_key: str, feat: dict) -> float:
    """เลือก features ที่โมเดลนั้นต้องการ แล้ว predict
    ใช้ pandas DataFrame เพราะ Pipeline มี ColumnTransformer ที่ต้องการ column names
    """
    if model_key not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_key}' not loaded. Check models directory."
        )
    required = metas[model_key]["features"]
    df = pd.DataFrame([{f: feat[f] for f in required}])
    result = models[model_key].predict(df)[0]
    return float(round(result, 4))


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
def root():
    loaded = list(models.keys())
    return {
        "service": "Res2bChar ML API",
        "status": "ok",
        "models_loaded": loaded,
        "endpoints": {
            "predict_all":   "POST /predict",
            "predict_yield": "POST /predict/yield",
            "health":        "GET  /health",
            "docs":          "GET  /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "models_expected": list(MODEL_FILES.keys())
    }


@app.post("/predict")
def predict_all(inp: PredictInput):
    """
    Predict ทุกค่าพร้อมกัน (ใช้หลัก endpoint นี้)
    Returns:
        predicted_yield  : Biochar yield (%)
        predicted_C      : Carbon content in biochar (%)
        predicted_HC     : H/C ratio
        predicted_OC     : O/C ratio
        predicted_FC     : Fixed carbon in biochar (%)
    """
    feat = build_features(inp)

    result = {}
    errors = {}

    for key, out_name in [
        ("Yield_char", "predicted_yield"),
        ("C_char",     "predicted_C"),
        ("H_C_char",   "predicted_HC"),
        ("O_C_char",   "predicted_OC"),
        ("FC_char",    "predicted_FC"),
    ]:
        try:
            result[out_name] = predict_one(key, feat)
        except HTTPException as e:
            errors[key] = e.detail
            result[out_name] = None

    result["errors"] = errors if errors else None
    result["input"]  = inp.dict()
    return result


@app.post("/predict/yield")
def predict_yield_only(inp: PredictInput):
    """
    Predict เฉพาะ Biochar Yield (%)
    Compatible กับ Apps Script เดิมที่ใช้ { predicted_yield }
    """
    feat  = build_features(inp)
    value = predict_one("Yield_char", feat)
    return {
        "predicted_yield": value,
        "input": inp.dict()
    }
