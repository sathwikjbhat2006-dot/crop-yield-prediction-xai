"""
shap_visual.py
==============
Generates a bar-chart of SHAP feature importance for a sample prediction
and saves it to results/feature_importance.png.
"""

import json
import os
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ── Load artefacts ───────────────────────────────────────
model   = joblib.load("models/best_model.pkl")
X_train = joblib.load("models/X_train.pkl")

with open("models/model_metadata.json") as f:
    meta = json.load(f)

MODEL_NAME    = meta["model_name"]
IS_TREE       = meta["is_tree_model"]
FEATURE_NAMES = meta["feature_names"]

# ── Sample input (Wheat in Punjab conditions) ────────────
sample = pd.DataFrame([{
    "Fertilizer_per_ha": 220.0,
    "Annual_Rainfall":   700.0,
    "N":                 120.0,
    "P":                 60.0,
    "K":                 40.0,
    "temperature":       20.0,
    "humidity":          55.0,
    "solar_radiation":   15.0,
    "wind_speed":        3.1,
}])[FEATURE_NAMES]

# ── SHAP ─────────────────────────────────────────────────
if IS_TREE:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    row_vals  = list(shap_vals[0]) if hasattr(shap_vals[0], "__len__") else list(shap_vals)
else:
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(sample)
    row_vals  = list(shap_vals[0]) if hasattr(shap_vals[0], "__len__") else list(shap_vals)

# ── Plot ─────────────────────────────────────────────────
features = list(sample.columns)
colors   = ["#2ecc71" if v >= 0 else "#e74c3c" for v in row_vals]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(features[::-1], row_vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)

ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_title(f"SHAP Feature Importance\n({MODEL_NAME})", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("SHAP Value (impact on predicted yield)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

for bar, val in zip(bars, row_vals[::-1]):
    ax.text(
        bar.get_width() + (0.002 if val >= 0 else -0.002),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.4f}",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=9,
    )

plt.tight_layout()
out_path = "results/feature_importance.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"✓  Saved → {os.path.abspath(out_path)}")