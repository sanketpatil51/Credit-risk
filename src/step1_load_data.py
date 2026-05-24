"""
STEP 1: KAGGLE DATA LOADER
Loads raw loan application data downloaded from Kaggle.

Dataset Source:
  Title   : Loan Default Prediction — Indian NBFC/Fintech
  Kaggle  : https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior
  License : CC0 (Public Domain)

How to download:
  Option A — Kaggle API (recommended):
      pip install kaggle
      kaggle datasets download -d subhamjain/loan-prediction-based-on-customer-behavior
      unzip loan-prediction-based-on-customer-behavior.zip -d ../data/

  Option B — Manual:
      1. Go to the Kaggle URL above
      2. Click "Download" → save CSV to  data/raw_loan_data.csv
"""

import pandas as pd
import os
import time

RAW = '../data/raw_loan_data.csv'

print("=" * 60)
print("  STEP 1: LOADING KAGGLE DATASET")
print("=" * 60)
print()
print("  Source : Kaggle — Loan Default Prediction")
print("  URL    : https://www.kaggle.com/datasets/")
print("           subhamjain/loan-prediction-based-on-customer-behavior")
print()

# ── Verify file exists ───────────────────────────────────
if not os.path.exists(RAW):
    print("  ❌  raw_loan_data.csv not found at:", os.path.abspath(RAW))
    print()
    print("  Download it from Kaggle first:")
    print("    pip install kaggle")
    print("    kaggle datasets download -d subhamjain/loan-prediction-based-on-customer-behavior")
    print("    unzip *.zip -d ../data/")
    raise FileNotFoundError(f"Kaggle dataset not found: {RAW}")

# ── Load ─────────────────────────────────────────────────
t0 = time.time()
df = pd.read_csv(RAW, low_memory=False)

print(f"  ✅  Loaded : {len(df):,} rows × {df.shape[1]} columns")
print(f"  ⏱   Time  : {time.time() - t0:.1f}s")
print()

# ── Quick data preview ───────────────────────────────────
print("── Column Overview ──────────────────────────────────")
print(f"  {'Column':<35} {'Dtype':<12} {'Missing %'}")
print("  " + "-" * 55)
for col in df.columns:
    missing_pct = df[col].isna().mean() * 100
    print(f"  {col:<35} {str(df[col].dtype):<12} {missing_pct:.1f}%")

print()
print("── Basic Stats ──────────────────────────────────────")
print(f"  Total rows        : {len(df):,}")
print(f"  Total columns     : {df.shape[1]}")
print(f"  Total missing vals: {df.isnull().sum().sum():,}")
print(f"  Duplicate rows    : {df.duplicated().sum():,}")

if 'default_12m' in df.columns:
    dr = pd.to_numeric(df['default_12m'], errors='coerce').mean()
    print(f"  Default rate      : {dr:.1%}")

print()
print(f"  💾 Dataset ready → {os.path.abspath(RAW)}")
print(f"  ▶  Next step     : python step2_clean_eda.py")
