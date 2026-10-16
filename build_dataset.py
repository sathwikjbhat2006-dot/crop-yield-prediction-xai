"""
build_dataset.py
================
Prepares two dataset artefacts needed by train_model.py:

  dataset/crop_npk_lookup.csv   — Per-crop average N, P, K (kg/ha)
  dataset/training_data.csv     — Merged training set with features:
        Crop, Fertilizer_per_ha, Annual_Rainfall, N, P, K,
        temperature, humidity, solar_radiation, wind_speed, Yield

Source:
  Real Kaggle "Crop Yield in India" CSV located at the long nested path
  under dataset/.  Fertilizer is total kg for the field → normalised to
  kg/ha by dividing by Area.

Weather columns (temperature, humidity, solar_radiation, wind_speed) are
NOT in the Kaggle CSV.  We synthesise them from realistic per-crop
distributions so the model feature vector matches what NASA POWER delivers
at inference time.  Values are drawn from Gaussian distributions anchored
to domain knowledge (Indian agricultural conditions).

Run this script once before train_model.py.
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
RAW_CSV = os.path.join(
    DATASET_DIR,
    "Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques",
    "Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques",
    "Crop Yield Data",
    "crop_yield.csv",
)
NPK_OUT  = os.path.join(DATASET_DIR, "crop_npk_lookup.csv")
TRAIN_OUT = os.path.join(DATASET_DIR, "training_data.csv")

os.makedirs(DATASET_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1.  Per-crop N, P, K lookup  (kg / ha)
#     Sourced from UCI Crop-Recommendation dataset means
#     extended with domain-expert estimates for extra crops.
# ─────────────────────────────────────────────────────────
NPK_TABLE = {
    # Crop                  N      P      K
    "Rice":               (80,   40,   40),
    "Wheat":              (120,  60,   40),
    "Maize":              (100,  50,   40),
    "Sugarcane":          (150,  60,   100),
    "Cotton(lint)":       (120,  60,   60),
    "Groundnut":          (25,   50,   75),
    "Soyabean":           (20,   60,   40),
    "Potato":             (120,  60,   120),
    "Onion":              (100,  50,   100),
    "Banana":             (200,  60,   300),
    "Gram":               (20,   60,   20),
    "Arhar/Tur":          (20,   60,   20),
    "Bajra":              (80,   40,   40),
    "Jowar":              (80,   40,   40),
    "Ragi":               (50,   25,   25),
    "Barley":             (80,   40,   30),
    "Rapeseed &Mustard":  (80,   40,   40),
    "Sunflower":          (60,   60,   60),
    "Turmeric":           (100,  50,   100),
    "Ginger":             (75,   50,   75),
    "Garlic":             (100,  60,   80),
    "Linseed":            (40,   20,   20),
    "Castor seed":        (40,   20,   20),
    "Sesamum":            (30,   15,   15),
    "Jute":               (60,   30,   30),
    "Coconut ":           (50,   30,   200),   # trailing space matches CSV
    "Coconut":            (50,   30,   200),
    "Arecanut":           (50,   30,   100),
    "Cashewnut":          (30,   20,   150),
    "Black pepper":       (30,   20,   100),
    "Cardamom":           (30,   20,   80),
    "Moong(Green Gram)":  (20,   60,   20),
    "Urad":               (20,   60,   20),
    "Cowpea(Lobia)":      (20,   60,   20),
    "Horse-gram":         (15,   40,   20),
    "Masoor":             (20,   40,   20),
    "Khesari":            (20,   40,   20),
    "Moth":               (15,   30,   20),
    "Dry chillies":       (100,  50,   100),
    "Coriander":          (50,   25,   25),
    "Tobacco":            (60,   30,   100),
    "Tapioca":            (100,  40,   150),
    "Sweet potato":       (80,   40,   100),
    "Safflower":          (40,   20,   20),
    "Niger seed":         (20,   10,   10),
    "Guar seed":          (20,   40,   20),
    "Small millets":      (40,   20,   20),
    "Other Cereals":      (60,   30,   30),
    "Oilseeds total":     (40,   30,   30),
    "other oilseeds":     (40,   30,   30),
    "Sannhamp":           (40,   20,   20),
    "Mesta":              (40,   20,   20),
    "Peas & beans (Pulses)": (20, 50,  20),
    "Other  Rabi pulses": (20,   50,   20),
    "Other Kharif pulses":(20,   50,   20),
    "Other Summer Pulses":(20,   50,   20),
}

# Per-crop realistic weather distributions  (mean, std)
# temperature (°C),  humidity (%),  solar (MJ/m²/day),  wind (m/s)
WEATHER_PARAMS = {
    # crop                  temp          humidity       solar          wind
    "Rice":               ((28,3),      (75,8),        (17,3),        (2.5,0.8)),
    "Wheat":              ((20,4),      (55,10),       (15,3),        (3.0,1.0)),
    "Maize":              ((26,3),      (65,8),        (18,3),        (2.8,0.8)),
    "Sugarcane":          ((30,3),      (70,8),        (20,3),        (2.5,0.8)),
    "Cotton(lint)":       ((30,4),      (60,10),       (20,3),        (3.0,1.0)),
    "Groundnut":          ((29,3),      (65,8),        (19,3),        (2.5,0.8)),
    "Soyabean":           ((26,3),      (65,8),        (17,3),        (2.5,0.8)),
    "Potato":             ((18,4),      (70,8),        (14,3),        (2.0,0.8)),
    "Onion":              ((22,4),      (60,8),        (16,3),        (2.5,0.8)),
    "Banana":             ((27,2),      (75,8),        (18,3),        (2.0,0.8)),
    "Gram":               ((20,4),      (50,10),       (15,3),        (3.0,1.0)),
    "Arhar/Tur":          ((28,3),      (65,8),        (18,3),        (2.5,0.8)),
    "Bajra":              ((30,3),      (55,10),       (20,3),        (3.5,1.0)),
    "Jowar":              ((28,3),      (60,8),        (19,3),        (3.0,0.8)),
    "Ragi":               ((26,3),      (65,8),        (18,3),        (2.5,0.8)),
    "Barley":             ((18,4),      (55,10),       (14,3),        (3.0,1.0)),
    "Rapeseed &Mustard":  ((18,4),      (55,10),       (14,3),        (3.0,1.0)),
    "Sunflower":          ((26,3),      (55,8),        (19,3),        (3.0,0.8)),
    "Turmeric":           ((27,2),      (75,8),        (16,3),        (2.0,0.8)),
    "Ginger":             ((25,2),      (75,8),        (15,3),        (2.0,0.8)),
    "Garlic":             ((20,4),      (60,8),        (15,3),        (2.5,0.8)),
    "Linseed":            ((18,4),      (55,10),       (14,3),        (3.0,1.0)),
    "Castor seed":        ((30,3),      (55,8),        (20,3),        (3.0,0.8)),
    "Sesamum":            ((30,3),      (60,8),        (20,3),        (3.0,0.8)),
    "Jute":               ((30,2),      (80,5),        (17,3),        (2.0,0.8)),
    "Coconut ":           ((28,2),      (78,5),        (18,3),        (2.5,0.8)),
    "Coconut":            ((28,2),      (78,5),        (18,3),        (2.5,0.8)),
    "Arecanut":           ((27,2),      (78,5),        (16,3),        (2.0,0.8)),
    "Cashewnut":          ((28,2),      (72,5),        (18,3),        (2.5,0.8)),
    "Black pepper":       ((26,2),      (80,5),        (15,3),        (2.0,0.8)),
    "Cardamom":           ((20,3),      (80,5),        (12,3),        (2.0,0.8)),
    "Moong(Green Gram)":  ((30,3),      (60,8),        (19,3),        (2.5,0.8)),
    "Urad":               ((30,3),      (60,8),        (19,3),        (2.5,0.8)),
    "Cowpea(Lobia)":      ((28,3),      (60,8),        (18,3),        (2.5,0.8)),
    "Horse-gram":         ((28,3),      (55,8),        (18,3),        (3.0,0.8)),
    "Masoor":             ((20,4),      (55,8),        (14,3),        (3.0,0.8)),
    "Khesari":            ((20,4),      (55,8),        (14,3),        (3.0,0.8)),
    "Moth":               ((32,3),      (45,8),        (21,3),        (3.5,1.0)),
    "Dry chillies":       ((28,3),      (65,8),        (18,3),        (2.5,0.8)),
    "Coriander":          ((22,4),      (55,8),        (15,3),        (2.5,0.8)),
    "Tobacco":            ((26,3),      (60,8),        (17,3),        (2.5,0.8)),
    "Tapioca":            ((28,2),      (75,5),        (18,3),        (2.0,0.8)),
    "Sweet potato":       ((26,3),      (70,8),        (16,3),        (2.0,0.8)),
    "Safflower":          ((28,3),      (50,8),        (20,3),        (3.5,1.0)),
    "Niger seed":         ((26,3),      (55,8),        (19,3),        (3.0,0.8)),
    "Guar seed":          ((30,3),      (50,8),        (20,3),        (3.5,1.0)),
    "Small millets":      ((28,3),      (60,8),        (18,3),        (3.0,0.8)),
    "Other Cereals":      ((26,3),      (60,8),        (17,3),        (3.0,0.8)),
    "Oilseeds total":     ((28,3),      (60,8),        (18,3),        (3.0,0.8)),
    "other oilseeds":     ((28,3),      (60,8),        (18,3),        (3.0,0.8)),
    "Sannhamp":           ((28,3),      (65,8),        (17,3),        (2.5,0.8)),
    "Mesta":              ((28,3),      (65,8),        (17,3),        (2.5,0.8)),
    "Peas & beans (Pulses)":((18,4),   (60,8),        (14,3),        (2.5,0.8)),
    "Other  Rabi pulses": ((18,4),      (55,8),        (14,3),        (3.0,0.8)),
    "Other Kharif pulses":((28,3),      (60,8),        (18,3),        (2.5,0.8)),
    "Other Summer Pulses":((30,3),      (55,8),        (19,3),        (3.0,0.8)),
}
_DEFAULT_WEATHER = ((27,4),(65,10),(17,3),(2.5,0.8))

# ─────────────────────────────────────────────────────────
# 2.  Save NPK lookup table
# ─────────────────────────────────────────────────────────
npk_rows = []
for crop, (n, p, k) in NPK_TABLE.items():
    npk_rows.append({"Crop": crop.strip(), "N": n, "P": p, "K": k})
npk_df = pd.DataFrame(npk_rows)
npk_df.to_csv(NPK_OUT, index=False)
print(f"[OK] Saved N/P/K lookup -> {NPK_OUT}  ({len(npk_df)} crops)")

# ─────────────────────────────────────────────────────────
# 3.  Load & clean raw crop-yield CSV
# ─────────────────────────────────────────────────────────
print(f"\n-->  Loading raw dataset: {RAW_CSV}")
df = pd.read_csv(RAW_CSV)

# Normalise categorical names (strip whitespace)
df["Crop"] = df["Crop"].str.strip()
df["State"] = df["State"].astype(str).str.strip()
df["Season"] = df["Season"].astype(str).str.strip()

# Drop rows with zero Area (avoid divide-by-zero)
df = df[df["Area"] > 0].copy()

# Remove extreme Yield outliers (keep < 99th pct)
q99 = df["Yield"].quantile(0.99)
df = df[df["Yield"] <= q99].copy()

# Fertilizer column is total kg for field → convert to kg/ha
df["Fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]

# Cap Fertilizer_per_ha at 99th percentile to remove outliers
fert_q99 = df["Fertilizer_per_ha"].quantile(0.99)
df = df[df["Fertilizer_per_ha"] <= fert_q99].copy()

print(f"   Rows after cleaning: {len(df)}")

# ─────────────────────────────────────────────────────────
# 4.  Join N, P, K from lookup table
# ─────────────────────────────────────────────────────────
npk_map = {row["Crop"]: (row["N"], row["P"], row["K"])
           for _, row in npk_df.iterrows()}

df["N"] = df["Crop"].map(lambda c: npk_map.get(c, (60, 30, 30))[0])
df["P"] = df["Crop"].map(lambda c: npk_map.get(c, (60, 30, 30))[1])
df["K"] = df["Crop"].map(lambda c: npk_map.get(c, (60, 30, 30))[2])

# ─────────────────────────────────────────────────────────
# 5.  Synthesise weather features per crop
# ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def _weather_col(crop_series, param_idx):
    """Draw weather values row-by-row from per-crop distributions."""
    vals = []
    for crop in crop_series:
        params = WEATHER_PARAMS.get(crop, _DEFAULT_WEATHER)
        mu, sigma = params[param_idx]
        vals.append(float(rng.normal(mu, sigma)))
    return vals

df["temperature"]     = _weather_col(df["Crop"], 0)
df["humidity"]        = _weather_col(df["Crop"], 1)
df["solar_radiation"] = _weather_col(df["Crop"], 2)
df["wind_speed"]      = _weather_col(df["Crop"], 3)

# Clip to realistic ranges
df["temperature"]     = df["temperature"].clip(5, 45)
df["humidity"]        = df["humidity"].clip(20, 99)
df["solar_radiation"] = df["solar_radiation"].clip(2, 30)
df["wind_speed"]      = df["wind_speed"].clip(0.5, 10)

# ─────────────────────────────────────────────────────────
# 6.  Select final columns and save
# ─────────────────────────────────────────────────────────
FEATURES = [
    "Crop_Year",
    "Crop",
    "State",
    "Season",
    "Fertilizer_per_ha",
    "Annual_Rainfall",
    "N", "P", "K",
    "temperature",
    "humidity",
    "solar_radiation",
    "wind_speed",
    "Yield",
]
train_df = df[FEATURES].dropna().reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)

print(f"\n[OK] Saved training data -> {TRAIN_OUT}")
print(f"   Shape : {train_df.shape}")
print(f"   Crops : {train_df['Crop'].nunique()}")
print(f"\n   Feature summary:")
print(train_df.describe().to_string())
