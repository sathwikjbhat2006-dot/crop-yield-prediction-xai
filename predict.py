"""
predict.py
==========
Standalone prediction script for quick testing.
Loads the saved model and runs a prediction for a sample crop input.
"""

import json
import joblib
import pandas as pd

# ── Load artefacts ───────────────────────────────────────
model = joblib.load("models/best_model.pkl")

with open("models/model_metadata.json") as f:
    meta = json.load(f)

MODEL_NAME    = meta["model_name"]
FEATURE_NAMES = meta["feature_names"]

print(f"Model      : {MODEL_NAME}")
print(f"R² Score   : {meta['r2_score']}")
print(f"Features   : {FEATURE_NAMES}")

# ── Sample input: Rice in Tamil Nadu ────────────────────
sample = pd.DataFrame([{
    "Fertilizer_per_ha": 160.0,   # kg/ha (N+P+K combined)
    "Annual_Rainfall":   1400.0,  # mm/year
    "N":                 80.0,    # kg/ha
    "P":                 40.0,    # kg/ha
    "K":                 40.0,    # kg/ha
    "temperature":       28.5,    # °C
    "humidity":          74.0,    # %
    "solar_radiation":   17.2,    # MJ/m²/day
    "wind_speed":        2.6,     # m/s
}])[FEATURE_NAMES]

# ── Predict ──────────────────────────────────────────────
predicted_yield = model.predict(sample)[0]

area_ha               = 10.0
total_production_tons = predicted_yield * area_ha
yield_kg_per_acre     = predicted_yield * 1000 / 2.47105

print(f"\n─── Prediction Results ───────────────────────")
print(f"  Predicted Yield       : {predicted_yield:.4f} tons/ha")
print(f"  For {area_ha} ha area : {total_production_tons:.2f} tons total")
print(f"  Equivalent            : {yield_kg_per_acre:.1f} kg/acre")