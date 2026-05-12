from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# LOAD MODELS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "Yield_char":  joblib.load(BASE_DIR / "Yield_char__CatBoost.joblib"),
    "C_char":      joblib.load(BASE_DIR / "C_char__HistGradientBoosting.joblib"),
    "H_C_char":    joblib.load(BASE_DIR / "H_C_char__GradientBoosting.joblib"),
    "O_C_char":    joblib.load(BASE_DIR / "O_C_char__CatBoost.joblib"),
    "FC_char":     joblib.load(BASE_DIR / "FC_char__GradientBoosting.joblib"),
}

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Res2bChar API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# INPUT MODEL
# =====================================================
class InputData(BaseModel):
    # Pyrolysis conditions
    Temp:    float
    RT:      float
    HR:      float
    # Proximate
    VM_bio:  float = 0.0
    Ash_bio: float = 0.0
    FC_bio:  float = 0.0
    # Ultimate
    H_bio:   float = 0.0
    C_bio:   float = 0.0
    N_bio:   float = 0.0
    O_bio:   float = 0.0

# =====================================================
# HELPER
# =====================================================
def safe_div(a, b):
    return a / b if b != 0 else 0.0

def build_row(data: InputData) -> dict:
    vm  = data.VM_bio
    ash = data.Ash_bio
    fc  = data.FC_bio
    h   = data.H_bio
    c   = data.C_bio
    o   = data.O_bio
    n   = data.N_bio

    return {
        "Temp":           data.Temp,
        "RT":             data.RT,
        "HR":             data.HR,
        "VM_bio":         vm,
        "Ash_bio":        ash,
        "FC_bio":         fc,
        "C_bio":          c,
        "H_bio":          h,
        "N_bio":          n,
        "O_bio":          o,
        "H/C_bio":        safe_div(h, c) * 12,
        "O/C_bio":        safe_div(o, c) * 12 / 16,
        "VM_to_Ash":      safe_div(vm, ash),
        "FC_to_Ash":      safe_div(fc, ash),
        "FC_bio_derived": max(0.0, 100.0 - vm - ash),
        "O_bio_derived":  max(0.0, 100.0 - c - h - n - ash),
    }

def predict_model(key: str, row: dict) -> float:
    model    = MODELS[key]
    features = list(model.feature_names_in_)
    X        = pd.DataFrame([[row[f] for f in features]], columns=features)
    return round(float(model.predict(X)[0]), 4)

# =====================================================
# ENDPOINTS
# =====================================================
@app.get("/")
def root():
    return {
        "message": "Res2bChar API v2 is running",
        "models":  list(MODELS.keys()),
        "endpoints": {
            "predict_all":   "POST /predict",
            "predict_yield": "POST /predict/yield",
            "health":        "GET  /health",
            "docs":          "GET  /docs",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.post("/predict")
def predict_all(data: InputData):
    """
    Predict ทุกค่าพร้อมกัน — ใช้ endpoint นี้หลัก
    Returns:
        predicted_yield : Biochar Yield (%)
        predicted_C     : Carbon content in biochar (%)
        predicted_HC    : H/C ratio
        predicted_OC    : O/C ratio
        predicted_FC    : Fixed Carbon in biochar (%)
    """
    row    = build_row(data)
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
            result[out_name] = predict_model(key, row)
        except Exception as e:
            errors[key]      = str(e)
            result[out_name] = None

    result["errors"] = errors if errors else None
    return result

@app.post("/predict/yield")
def predict_yield(data: InputData):
    """Backward-compatible — return เฉพาะ predicted_yield"""
    row = build_row(data)
    return {"predicted_yield": predict_model("Yield_char", row)}
