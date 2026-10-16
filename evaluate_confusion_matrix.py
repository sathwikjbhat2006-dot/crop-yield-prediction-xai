"""
evaluate_confusion_matrix.py
============================
Evaluates crop yield predictions as a 3-category classification task:
  - Low
  - Medium
  - High

Category thresholds are calculated strictly from the 33rd and 67th percentiles
of the training target values (y_train only).
These same thresholds are applied to actual test-set yields (y_test) and
model predictions on the test set (y_pred).

Calculates and saves:
  1. results/confusion_matrix.csv   — Confusion matrix counts (Actual vs Predicted)
  2. results/confusion_matrix.png   — Publication-quality annotated heatmap plot
  3. results/confusion_metrics.json  — Accuracy, macro metrics, per-category metrics,
                                      thresholds, category order, model name, sample counts
"""

import os
import sys
import json
import math
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# Feature definitions matching train_model.py
NUMERIC_COLS = [
    "Crop_Year",
    "Fertilizer_per_ha",
    "Annual_Rainfall",
    "N", "P", "K",
    "temperature",
    "humidity",
    "solar_radiation",
    "wind_speed"
]
CATEGORICAL_COLS = ["Crop", "State", "Season"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
CATEGORY_ORDER = ["Low", "Medium", "High"]


def categorize_yield(values, th_low: float, th_high: float) -> pd.Series:
    """
    Discretizes continuous yield values into three categories:
      - Low:    value <= th_low
      - Medium: th_low < value <= th_high
      - High:   value > th_high

    Uses bins [-inf, th_low, th_high, inf] with right-closed intervals.
    """
    categories = pd.cut(
        values,
        bins=[-np.inf, th_low, th_high, np.inf],
        labels=CATEGORY_ORDER,
        right=True
    )
    return categories.astype(str)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    model_name: str,
    accuracy: float,
    macro_f1: float,
    th_low: float,
    th_high: float,
    total_samples: int,
    output_png_path: str
):
    """
    Plots and saves a clearly labeled, publication-quality confusion matrix heatmap.
    Axes are explicitly labeled 'Actual Category' and 'Predicted Category'.
    """
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            count = cm[i, j]
            row_pct = (count / row_sum * 100) if row_sum > 0 else 0.0
            annot[i, j] = f"{count:,}\n({row_pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        cbar_kws={"label": "Sample Count"},
        linewidths=1.5,
        linecolor="#e0e0e0",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax
    )

    ax.set_title(
        f"Confusion Matrix — {model_name} (Crop Yield Classification)\n"
        f"Accuracy: {accuracy * 100:.2f}% | Macro F1: {macro_f1:.4f} | Test Samples: {total_samples:,}",
        fontsize=13,
        fontweight="bold",
        pad=16
    )
    ax.set_xlabel("Predicted Category", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Category", fontsize=12, fontweight="bold", labelpad=10)

    # Footnote explaining the derivation of categories and cell format
    note_text = (
        f"Thresholds (from y_train): Low \u2264 {th_low:.4f} t/ha | "
        f"Medium: ({th_low:.4f}, {th_high:.4f}] t/ha | "
        f"High > {th_high:.4f} t/ha\n"
        f"Cell format: Count (Row % of Actual Category)"
    )
    plt.figtext(0.5, -0.04, note_text, ha="center", fontsize=9.5, style="italic", color="#333333")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_confusion_matrix_evaluation(
    best_model=None,
    X_test=None,
    y_test=None,
    y_train=None,
    best_model_name: str = None,
    results_dir: str = "results",
    model_dir: str = "models",
    data_csv_path: str = "dataset/training_data.csv"
):
    """
    Executes confusion matrix evaluation using actual model predictions.

    If best_model, X_test, y_test, and y_train are not supplied, they are loaded
    directly from the existing fitted model artefact and dataset using the exact
    train-test split (test_size=0.2, random_state=42).
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load model and metadata if not provided
    if best_model is None:
        model_pkl_path = os.path.join(model_dir, "best_model.pkl")
        if not os.path.exists(model_pkl_path):
            raise FileNotFoundError(f"Model file not found: {model_pkl_path}")
        best_model = joblib.load(model_pkl_path)

    if best_model_name is None:
        meta_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            best_model_name = meta.get("selected_model", meta.get("model_name", "Best Model"))
        else:
            best_model_name = "Best Model"

    # 2. Reconstruct train-test split if data not provided
    if X_test is None or y_test is None or y_train is None:
        if not os.path.exists(data_csv_path):
            raise FileNotFoundError(f"Dataset not found: {data_csv_path}")
        df = pd.read_csv(data_csv_path)
        X = df[FEATURE_COLS]
        y = df["Yield"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 3. Generate actual model predictions
    y_pred = best_model.predict(X_test)

    # 4. Calculate thresholds from y_train ONLY (33rd and 67th percentiles)
    th_low = float(np.percentile(y_train, 33))
    th_high = float(np.percentile(y_train, 67))

    # 5. Discretize actual and predicted yields
    y_true_cat = categorize_yield(y_test, th_low, th_high)
    y_pred_cat = categorize_yield(y_pred, th_low, th_high)

    # 6. Compute confusion matrix (Rows: Actual, Columns: Predicted)
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORY_ORDER)

    # 7. Compute overall and per-category classification metrics
    accuracy = float(accuracy_score(y_true_cat, y_pred_cat))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_cat,
        y_pred_cat,
        labels=CATEGORY_ORDER,
        average="macro",
        zero_division=0
    )
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_cat,
        y_pred_cat,
        labels=CATEGORY_ORDER,
        average=None,
        zero_division=0
    )

    per_category = {}
    for idx, cat in enumerate(CATEGORY_ORDER):
        per_category[cat] = {
            "precision": round(float(precisions[idx]), 6),
            "recall": round(float(recalls[idx]), 6),
            "f1_score": round(float(f1s[idx]), 6),
            "support": int(supports[idx])
        }

    # 8. Save results/confusion_matrix.csv
    cm_df = pd.DataFrame(
        cm,
        index=CATEGORY_ORDER,
        columns=CATEGORY_ORDER
    )
    cm_df.index.name = "Actual Category"
    csv_path = os.path.join(results_dir, "confusion_matrix.csv")
    cm_df.to_csv(csv_path)

    # 9. Save results/confusion_matrix.png
    png_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm=cm,
        labels=CATEGORY_ORDER,
        model_name=best_model_name,
        accuracy=accuracy,
        macro_f1=float(macro_f1),
        th_low=th_low,
        th_high=th_high,
        total_samples=len(y_test),
        output_png_path=png_path
    )

    # 10. Save results/confusion_metrics.json
    metrics_summary = {
        "selected_model_name": best_model_name,
        "test_samples": int(len(y_test)),
        "training_samples": int(len(y_train)),
        "category_order": CATEGORY_ORDER,
        "category_thresholds": {
            "p33_threshold": round(th_low, 6),
            "p67_threshold": round(th_high, 6),
            "units": "tons/hectare",
            "definitions": {
                "Low": f"yield <= {th_low:.6f} tons/ha (33rd percentile of y_train)",
                "Medium": f"{th_low:.6f} < yield <= {th_high:.6f} tons/ha (33rd to 67th percentile of y_train)",
                "High": f"yield > {th_high:.6f} tons/ha (> 67th percentile of y_train)"
            }
        },
        "overall_metrics": {
            "accuracy": round(accuracy, 6),
            "macro_precision": round(float(macro_precision), 6),
            "macro_recall": round(float(macro_recall), 6),
            "macro_f1": round(float(macro_f1), 6)
        },
        "per_category_metrics": per_category,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_axes": {
            "rows": "Actual Category",
            "columns": "Predicted Category",
            "labels": CATEGORY_ORDER
        },
        "timestamp": datetime.now().isoformat()
    }

    json_path = os.path.join(results_dir, "confusion_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 11. Print comprehensive console output
    print("\n" + "=" * 65)
    print("  CONFUSION MATRIX EVALUATION")
    print("=" * 65)
    print(f"Selected Model   : {best_model_name}")
    print(f"Test Samples     : {len(y_test):,}")
    print(f"Training Samples : {len(y_train):,}")
    print(f"Category Order   : {CATEGORY_ORDER}")
    print(f"Thresholds (from y_train only):")
    print(f"  - Low    : yield <= {th_low:.4f} tons/ha (33rd percentile)")
    print(f"  - Medium : {th_low:.4f} < yield <= {th_high:.4f} tons/ha")
    print(f"  - High   : yield > {th_high:.4f} tons/ha (67th percentile)")

    print("\nConfusion Matrix (Rows: Actual Category, Columns: Predicted Category):")
    print(cm_df.to_string())

    print("\nOverall Classification Metrics:")
    print(f"  - Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  - Macro Precision : {macro_precision:.4f}")
    print(f"  - Macro Recall    : {macro_recall:.4f}")
    print(f"  - Macro F1-score  : {macro_f1:.4f}")

    print("\nPer-Category Metrics:")
    print(f"  {'Category':<10} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
    print("  " + "-" * 56)
    for cat in CATEGORY_ORDER:
        m = per_category[cat]
        print(f"  {cat:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['support']:<10}")

    print("\nGenerated Artefacts:")
    print(f"  [OK] Saved confusion matrix CSV  -> {csv_path}")
    print(f"  [OK] Saved confusion matrix plot -> {png_path}")
    print(f"  [OK] Saved confusion metrics     -> {json_path}")
    print("=" * 65 + "\n")

    return metrics_summary


if __name__ == "__main__":
    run_confusion_matrix_evaluation()
