"""
results/model_comparison_plot.py
=================================
Generates a grouped bar chart comparing MAE, RMSE, and R² for all
trained models, saved to results/model_comparison.png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/model_comparison.csv")

models  = df["Model"].tolist()
x       = np.arange(len(models))
width   = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Comparison — Crop Yield Prediction", fontsize=15, fontweight="bold", y=1.01)

# ── Left: MAE & RMSE ─────────────────────────────────────
ax1 = axes[0]
ax1.bar(x - width / 2, df["MAE"],  width, label="MAE",  color="#3498db", alpha=0.85)
ax1.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="#e67e22", alpha=0.85)

ax1.set_title("Error Metrics (lower is better)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha="right")
ax1.set_ylabel("Error (tons/ha)")
ax1.legend()
ax1.spines[["top", "right"]].set_visible(False)

for bar in ax1.patches:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=8,
    )

# ── Right: R² Score ──────────────────────────────────────
ax2 = axes[1]
colors = ["#2ecc71" if r2 == df["R2"].max() else "#95a5a6" for r2 in df["R2"]]
bars   = ax2.bar(x, df["R2"], color=colors, alpha=0.85, edgecolor="white")

ax2.set_title("R² Score (higher is better)", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15, ha="right")
ax2.set_ylabel("R² Score")
ax2.set_ylim(0, 1.05)
ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)

for bar, r2 in zip(bars, df["R2"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{r2:.4f}",
        ha="center", va="bottom", fontsize=9,
    )

plt.tight_layout()
out_path = "results/model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓  Saved → {os.path.abspath(out_path)}")