"""
extract_methodology_info_v2.py

Run from D:\\mini project (the project root):

    python extract_methodology_info_v2.py

Pulls everything needed for IEEE Section IV (Methodology) and
Section V (Results) from your actual file layout:

    dataset/training_data.csv
    dataset/crop_npk_lookup.csv
    train_model.py
    models/model_metadata.json
    results/model_comparison.csv
    results/feature_importance.csv
    results/feature_group_importance.csv
    results/per_crop_metrics.csv
    results/ablation_study.csv
    results/cv_fold_results.csv
    results/loyo_results.csv
    results/loro_results.csv
    results/literature_comparison.csv
    results/generalization_report.json
    results/significance_test.json
    results/research_summary.json
    results/research_claims.json
    results/shap_stability_report.json

Saves everything to methodology_report.txt in the same folder.
Nothing is invented — missing files/columns are reported as NOT FOUND.
"""

import json
import os
import re

import pandas as pd

DATASET_CSV = "dataset/training_data.csv"
NPK_LOOKUP_CSV = "dataset/crop_npk_lookup.csv"
TRAIN_SCRIPT = "train_model.py"
METADATA_JSON = "models/model_metadata.json"

RESULTS_CSVS = [
    "results/model_comparison.csv",
    "results/feature_importance.csv",
    "results/feature_group_importance.csv",
    "results/per_crop_metrics.csv",
    "results/ablation_study.csv",
    "results/cv_fold_results.csv",
    "results/loyo_results.csv",
    "results/loro_results.csv",
    "results/literature_comparison.csv",
    "results/table_feature_importance.csv",
    "results/table_model_metrics.csv",
    "results/table_per_crop.csv",
]

RESULTS_JSONS = [
    "results/generalization_report.json",
    "results/significance_test.json",
    "results/research_summary.json",
    "results/research_claims.json",
    "results/shap_stability_report.json",
]

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(str(line))


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def show_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        log(f"--- {path}  (shape: {df.shape[0]} rows x {df.shape[1]} cols) ---")
        log(df.to_string(index=False))
    else:
        log(f"NOT FOUND: {path}")


def show_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        log(f"--- {path} ---")
        log(json.dumps(data, indent=2))
    else:
        log(f"NOT FOUND: {path}")


# 1. Dataset summary
section("1. DATASET SUMMARY (dataset/training_data.csv)")
if os.path.exists(DATASET_CSV):
    df = pd.read_csv(DATASET_CSV)
    log(f"Rows: {df.shape[0]}")
    log(f"Columns: {df.shape[1]}")
    log(f"Column names: {list(df.columns)}")
    log(f"\nDtypes:\n{df.dtypes}")
    log(f"\nMissing values per column:\n{df.isna().sum()}")

    for col in ["Crop", "crop", "Crop_Name"]:
        if col in df.columns:
            log(f"\nUnique crops ('{col}'): {df[col].nunique()}")
            log(f"Crop list: {sorted(df[col].unique().tolist())}")
            break
    else:
        log("\nCrop column NOT FOUND")

    for col in ["State", "state"]:
        if col in df.columns:
            log(f"\nUnique states ('{col}'): {df[col].nunique()}")
            break
    else:
        log("\nState column NOT FOUND")

    for col in ["Crop_Year", "Year", "year", "Season"]:
        if col in df.columns:
            try:
                log(f"\nYear/season range ('{col}'): {df[col].min()} to {df[col].max()}")
            except TypeError:
                log(f"\n'{col}' unique values: {sorted(df[col].astype(str).unique())}")
            break
    else:
        log("\nYear/Crop_Year/Season column NOT FOUND")

    log(f"\nFull describe():\n{df.describe(include='all')}")
else:
    log(f"NOT FOUND: {DATASET_CSV}")

section("1b. NPK LOOKUP TABLE (dataset/crop_npk_lookup.csv)")
show_csv(NPK_LOOKUP_CSV)

# 2. Training script details
section("2. TRAINING SCRIPT DETAILS (train_model.py)")
if os.path.exists(TRAIN_SCRIPT):
    with open(TRAIN_SCRIPT, "r") as f:
        code = f.read()

    y_match = re.findall(r"^\s*y\s*=\s*.+$", code, re.MULTILINE)
    log("Target variable assignment line(s):")
    log("\n".join(y_match) if y_match else "  NOT FOUND")

    x_match = re.findall(r"^\s*X\s*=\s*.+$", code, re.MULTILINE)
    log("\nFeature matrix (X) assignment line(s):")
    log("\n".join(x_match) if x_match else "  NOT FOUND")

    tts_match = re.findall(r"train_test_split\([^\)]*\)", code)
    log("\ntrain_test_split() call(s):")
    log("\n".join(tts_match) if tts_match else "  NOT FOUND")

    cv_match = re.findall(r"KFold\([^\)]*\)|cross_val_score\([^\)]*\)|GridSearchCV\([^\)]*\)", code)
    log("\nCross-validation call(s):")
    log("\n".join(cv_match) if cv_match else "  NOT FOUND")

    model_match = re.findall(r"(LinearRegression|DecisionTreeRegressor|RandomForestRegressor|XGBRegressor|XGBoost)\([^\)]*\)", code)
    log("\nModel instantiation(s):")
    log("\n".join(model_match) if model_match else "  NOT FOUND")
else:
    log(f"NOT FOUND: {TRAIN_SCRIPT}")

# 3. Model metadata
section("3. MODEL METADATA (models/model_metadata.json)")
show_json(METADATA_JSON)

# 4. All results CSVs
section("4. RESULTS TABLES (results/*.csv)")
for path in RESULTS_CSVS:
    show_csv(path)
    log("")

# 5. All results JSONs
section("5. RESULTS SUMMARIES (results/*.json)")
for path in RESULTS_JSONS:
    show_json(path)
    log("")

with open("methodology_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n\nSaved full report to methodology_report.txt")