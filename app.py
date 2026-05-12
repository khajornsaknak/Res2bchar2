from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from math import isfinite

# =====================================================
# LOAD MODELS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Yield_char": "Yield_char__CatBoost.joblib",
    "C_char":     "C_char__HistGradientBoosting.joblib",
    "H_C_char":   "H_C_char__GradientBoosting.joblib",
    "O_C_char":   "O_C_char__CatBoost.joblib",
    "FC_char":    "FC_char__GradientBoosting.joblib",
}

MODELS = {}

# =====================================================
# LOAD ALL MODELS SAFELY
# =====================================================

for key, filename in MODEL_FILES.items():

    model_path = BASE_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {filename}"
        )

    MODELS[key] = joblib.load(model_path)

# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Res2bChar API",
    version="2.0.0"
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# INPUT MODEL
# =====================================================

class InputData(BaseModel):

    # Pyrolysis Conditions
    Temp: float
    RT: float
    HR: float

    # Proximate Analysis
    VM_bio: float = 0.0
    Ash_bio: float = 0.0
    FC_bio: float = 0.0

    # Ultimate Analysis
    H_bio: float = 0.0
    C_bio: float = 0.0
    N_bio: float = 0.0
    O_bio: float = 0.0

# =====================================================
# SAFE FUNCTIONS
# =====================================================

def safe_number(x):

    if x is None:
        return 0.0

    try:
        x = float(x)

        if not isfinite(x):
            return 0.0

        return x

    except:
        return 0.0


def safe_div(a, b):

    a = safe_number(a)
    b = safe_number(b)

    if b == 0:
        return 0.0

    return a / b

# =====================================================
# BUILD FEATURE ROW
# =====================================================

def build_row(data: InputData):

    vm  = safe_number(data.VM_bio)
    ash = safe_number(data.Ash_bio)
    fc  = safe_number(data.FC_bio)

    h   = safe_number(data.H_bio)
    c   = safe_number(data.C_bio)
    n   = safe_number(data.N_bio)
    o   = safe_number(data.O_bio)

    row = {

        # =========================================
        # BASIC
        # =========================================
        "Temp": safe_number(data.Temp),
        "RT": safe_number(data.RT),
        "HR": safe_number(data.HR),

        # =========================================
        # PROXIMATE
        # =========================================
        "VM_bio": vm,
        "Ash_bio": ash,
        "FC_bio": fc,

        # =========================================
        # ULTIMATE
        # =========================================
        "C_bio": c,
        "H_bio": h,
        "N_bio": n,
        "O_bio": o,

        # =========================================
        # RATIOS
        # =========================================
        "H/C_bio": safe_div(h, c) * 12,
        "O/C_bio": safe_div(o, c) * 12 / 16,

        # =========================================
        # ENGINEERED FEATURES
        # =========================================
        "VM_to_Ash": safe_div(vm, ash),
        "FC_to_Ash": safe_div(fc, ash),

        "Temp_x_RT": safe_number(data.Temp) * safe_number(data.RT),
        "Temp_x_HR": safe_number(data.Temp) * safe_number(data.HR),
        "Temp_x_Ash": safe_number(data.Temp) * ash,
        "Temp_x_VM": safe_number(data.Temp) * vm,

        # =========================================
        # DERIVED
        # =========================================
        "FC_bio_derived": max(
            0.0,
            100.0 - vm - ash
        ),

        "O_bio_derived": max(
            0.0,
            100.0 - c - h - n - ash
        ),

        # =========================================
        # CATEGORICAL PLACEHOLDER
        # =========================================
        "Feedstock_type": "Unknown",
    }

    return row

# =====================================================
# GET FEATURE NAMES
# =====================================================

def get_feature_names(model):

    # CatBoost
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)

    # sklearn
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise ValueError(
        "Cannot determine model feature names"
    )

# =====================================================
# PREDICT SINGLE MODEL
# =====================================================

def predict_model(model_key: str, row: dict):

    model = MODELS[model_key]

    # =========================================
    # FEATURES
    # =========================================
    features = get_feature_names(model)

    # =========================================
    # BUILD INPUT
    # =========================================
    values = []

    for f in features:

        # if missing feature
        if f not in row:

            # categorical
            if "type" in f.lower():
                values.append("Unknown")

            else:
                values.append(0.0)

        else:
            values.append(row[f])

    X = pd.DataFrame(
        [values],
        columns=features
    )

    # =========================================
    # CONVERT OBJECT COLUMNS SAFELY
    # =========================================
    for col in X.columns:

        # skip categorical strings
        if isinstance(X[col].iloc[0], str):
            continue

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        ).fillna(0)

    # =========================================
    # PREDICT
    # =========================================
    pred = model.predict(X)[0]

    return round(float(pred), 4)

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {

        "message": "Res2bChar API v2 is running",

        "models_loaded": list(MODELS.keys()),

        "endpoints": {

            "predict_all": "POST /predict",
            "predict_yield": "POST /predict/yield",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {

        "status": "ok",
        "models_loaded": list(MODELS.keys())
    }

# =====================================================
# PREDICT ALL
# =====================================================

@app.post("/predict")
def predict_all(data: InputData):

    try:

        row = build_row(data)

        results = {}
        errors = {}

        targets = [

            ("Yield_char", "predicted_yield"),
            ("C_char", "predicted_C"),
            ("H_C_char", "predicted_HC"),
            ("O_C_char", "predicted_OC"),
            ("FC_char", "predicted_FC"),
        ]

        for model_key, output_name in targets:

            try:

                results[output_name] = predict_model(
                    model_key,
                    row
                )

            except Exception as e:

                results[output_name] = None

                errors[model_key] = str(e)

        results["errors"] = errors if errors else None

        return results

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================================
# PREDICT YIELD ONLY
# =====================================================

@app.post("/predict/yield")
def predict_yield(data: InputData):

    try:

        row = build_row(data)

        pred = predict_model(
            "Yield_char",
            row
        )

        return {
            "predicted_yield": pred
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================================
# DEBUG FEATURES
# =====================================================

@app.get("/debug/features")
def debug_features():

    output = {}

    for key, model in MODELS.items():

        try:

            output[key] = get_feature_names(model)

        except Exception as e:

            output[key] = str(e)

    return output
