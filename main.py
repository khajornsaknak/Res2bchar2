from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Biochar Prediction API",
    description="Predict biochar properties from pyrolysis conditions and biomass composition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load models ──────────────────────────────────────────────────────────────
MODELS = {}
MODEL_FILES = {
    "C_char":   "C_char__HistGradientBoosting.joblib",
    "H_C_char": "H_C_char__GradientBoosting.joblib",
    "O_C_char": "O_C_char__CatBoost.joblib",
    "FC_char":  "FC_char__GradientBoosting.joblib",
    "Yield":    "Yield_char__CatBoost.joblib",
}

MODEL_DIR = os.getenv("MODEL_DIR", "models")

for key, filename in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        MODELS[key] = joblib.load(path)
        print(f"✓ Loaded {key} from {path}")
    else:
        print(f"✗ Missing model file: {path}")


# ─── Input schema ─────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    # Pyrolysis conditions
    Temp: float = Field(..., description="Pyrolysis temperature (°C)", example=500)
    RT: float = Field(..., description="Residence time (min)", example=60)
    HR: float = Field(..., description="Heating rate (°C/min)", example=10)

    # Ultimate analysis of biomass
    C_bio: float = Field(..., description="Carbon content of biomass (%)", example=45.0)
    H_bio: float = Field(..., description="Hydrogen content of biomass (%)", example=6.0)
    N_bio: float = Field(..., description="Nitrogen content of biomass (%)", example=1.0)
    O_bio: float = Field(..., description="Oxygen content of biomass (%)", example=40.0)

    # Proximate analysis of biomass
    Ash_bio: float = Field(..., description="Ash content of biomass (%)", example=5.0)
    VM_bio: Optional[float] = Field(None, description="Volatile matter of biomass (%) — required for FC_char", example=75.0)
    FC_bio: Optional[float] = Field(None, description="Fixed carbon of biomass (%) — required for FC_char", example=15.0)

    class Config:
        json_schema_extra = {
            "example": {
                "Temp": 500, "RT": 60, "HR": 10,
                "C_bio": 45.0, "H_bio": 6.0, "N_bio": 1.0, "O_bio": 40.0,
                "Ash_bio": 5.0, "VM_bio": 75.0, "FC_bio": 15.0
            }
        }


def derive_features(data: PredictionInput) -> dict:
    """Calculate derived features used by models."""
    H_C_bio = data.H_bio / data.C_bio if data.C_bio else 0
    O_C_bio = data.O_bio / data.C_bio if data.C_bio else 0
    # FC derived from ultimate analysis (approximation if not provided)
    FC_bio_derived = 100 - (data.VM_bio or 0) - data.Ash_bio
    O_bio_derived = data.O_bio

    return {
        "Temp": data.Temp,
        "RT": data.RT,
        "HR": data.HR,
        "C_bio": data.C_bio,
        "H_bio": data.H_bio,
        "N_bio": data.N_bio,
        "Ash_bio": data.Ash_bio,
        "O_bio": data.O_bio,
        "H/C_bio": H_C_bio,
        "O/C_bio": O_C_bio,
        "FC_bio_derived": FC_bio_derived,
        "O_bio_derived": O_bio_derived,
        # FC_char specific
        "VM_bio": data.VM_bio or 0,
        "FC_bio": data.FC_bio or 0,
        "VM_to_Ash": (data.VM_bio / data.Ash_bio) if (data.VM_bio and data.Ash_bio) else 0,
        "FC_to_Ash": (data.FC_bio / data.Ash_bio) if (data.FC_bio and data.Ash_bio) else 0,
    }


# ─── Feature lists per model ──────────────────────────────────────────────────
FEATURES = {
    "C_char":   ['Temp','RT','HR','C_bio','H_bio','N_bio','Ash_bio','O_bio','H/C_bio','O/C_bio','FC_bio_derived','O_bio_derived'],
    "H_C_char": ['Temp','RT','HR','C_bio','H_bio','N_bio','Ash_bio','O_bio','H/C_bio','O/C_bio','FC_bio_derived','O_bio_derived'],
    "O_C_char": ['Temp','RT','HR','C_bio','H_bio','N_bio','Ash_bio','O_bio','H/C_bio','O/C_bio','FC_bio_derived','O_bio_derived'],
    "FC_char":  ['Temp','RT','HR','VM_bio','Ash_bio','FC_bio','C_bio','H/C_bio','O/C_bio','VM_to_Ash','FC_to_Ash','FC_bio_derived','O_bio_derived'],
    "Yield":    ['Temp','RT','HR','VM_bio','Ash_bio','FC_bio','C_bio','H/C_bio','O/C_bio','VM_to_Ash','FC_to_Ash','FC_bio_derived','O_bio_derived'],
}


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "loaded_models": list(MODELS.keys()),
        "missing_models": [k for k in MODEL_FILES if k not in MODELS]
    }


@app.post("/predict", tags=["Prediction"])
def predict_all(data: PredictionInput):
    """
    Predict all available biochar properties at once.
    Returns: C_char, H/C_char, O/C_char, FC_char, Yield
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")

    feats = derive_features(data)
    results = {}

    for key, model in MODELS.items():
        try:
            feat_list = FEATURES[key]
            X = pd.DataFrame([{f: feats[f] for f in feat_list}])
            pred = model.predict(X)[0]
            results[key] = round(float(pred), 4)
        except Exception as e:
            results[key] = {"error": str(e)}

    return {
        "inputs": data.model_dump(),
        "predictions": results,
        "units": {
            "C_char": "%",
            "H_C_char": "molar ratio",
            "O_C_char": "molar ratio",
            "FC_char": "%",
            "Yield": "%"
        }
    }


@app.post("/predict/{target}", tags=["Prediction"])
def predict_single(target: str, data: PredictionInput):
    """Predict a single biochar property. target = C_char | H_C_char | O_C_char | FC_char | Yield"""
    if target not in MODELS:
        available = list(MODELS.keys())
        raise HTTPException(status_code=404, detail=f"Model '{target}' not found. Available: {available}")

    feats = derive_features(data)
    feat_list = FEATURES[target]
    X = pd.DataFrame([{f: feats[f] for f in feat_list}])

    try:
        pred = MODELS[target].predict(X)[0]
        return {"target": target, "prediction": round(float(pred), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
