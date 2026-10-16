"""
shap_explain.py
===============
Standalone SHAP debug script — loads the best saved model and prints
feature contributions for a sample prediction.

Adapts automatically to the model type (TreeExplainer / LinearExplainer).
"""

import json
import joblib
import pandas as pd
import shap

# ── Load artefacts ───────────────────────────────────────
model   = joblib.load("models/best_model.pkl")
X_train = joblib.load("models/X_train.pkl")

with open("models/model_metadata.json") as f:
    meta = json.load(f)

MODEL_NAME    = meta["model_name"]
IS_TREE       = meta["is_tree_model"]
FEATURE_NAMES = meta["feature_names"]

print(f"Model      : {MODEL_NAME}")
print(f"Explainer  : {'TreeExplainer' if IS_TREE else 'LinearExplainer'}")

# ── Sample input (Rice in Chennai conditions) ────────────
sample = pd.DataFrame([{
    "Fertilizer_per_ha": 160.0,   # N+P+K combined (kg/ha)
    "Annual_Rainfall":   1400.0,  # mm/year
    "N":                 80.0,
    "P":                 40.0,
    "K":                 40.0,
    "temperature":       28.5,    # °C
    "humidity":          74.0,    # %
    "solar_radiation":   17.2,    # MJ/m²/day
    "wind_speed":        2.6,     # m/s
}])[FEATURE_NAMES]

# ── Predict ──────────────────────────────────────────────
prediction = model.predict(sample)[0]
print(f"\nPredicted Yield : {prediction:.4f} tons/hectare")

# ── SHAP ─────────────────────────────────────────────────
if IS_TREE:
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(sample)
    row_vals   = shap_vals[0] if hasattr(shap_vals[0], "__len__") else shap_vals
else:
    explainer  = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_vals  = explainer.shap_values(sample)
    row_vals   = shap_vals[0] if hasattr(shap_vals[0], "__len__") else shap_vals

print("\nFeature Contributions (SHAP values):")
print("-" * 40)
contributions = sorted(
    zip(sample.columns, row_vals),
    key=lambda x: abs(x[1]),
    reverse=True,
)
for feat, val in contributions:
    print(f"  {feat:<22} : {val:+.4f}")