# 🏦 Fintech Credit Risk Pipeline
## End-to-End: Big Data → Cleaning → EDA → ML Model → Streamlit

---

## 📁 Project Structure
```
fintech_project/
├── data/
│   ├── raw_loan_data.csv         # Raw loan records (525,000 rows)
│   └── clean_loan_data.csv       # Cleaned, model-ready data
├── src/
│   ├── step1_load_data.py        # Load & validate raw dataset
│   ├── step2_clean_eda.py        # 14-step cleaning + full EDA
│   └── step3_model_building.py   # PD model + scorecard + monitoring
├── models/
│   ├── pd_model.pkl              # Trained Gradient Boosting model
│   ├── scaler.pkl                # StandardScaler for LR
│   └── features.json             # Feature list
├── app/
│   └── streamlit_app.py          # Full Streamlit dashboard
├── reports/
│   ├── eda_report.json           # EDA stats and distributions
│   └── model_report.json         # Model metrics and feature importance
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate the loaded data
python src/step1_load_data.py

# 3. Run cleaning + EDA (takes ~30-60s)
python src/step2_clean_eda.py

# 4. Build and evaluate model
python src/step3_model_building.py

# 5. Launch Streamlit app
streamlit run app/streamlit_app.py
```

---

## 🧹 14 Data Cleaning Steps

| Step | Technique | Finance Reason |
|------|-----------|---------------|
| 1 | Duplicate removal + PAN dedup | Same person applies multiple times |
| 2 | PAN regex validation | Format: `[A-Z]{5}[0-9]{4}[A-Z]` |
| 3 | Date parsing + stale bureau flag | Bureau >90 days old = unreliable |
| 4 | Age cap (21-65) + median impute | Lending policy limit |
| 5 | Gender standardization (10+ → 4) | Mapping dictionary |
| 6 | Employment (30+ spellings → 4) | Standard categories for WOE |
| 7 | Income Winsorization p1-p99 | Preserve rows, cap extremes |
| 8 | Loan amount validation | Must be positive |
| 9 | Interest rate range check | 5-60% valid range |
| 10 | CIBIL NTC strategy | Missing ≠ bad; sentinel -1 |
| 11 | Bureau DPD sentinels | Missing DPD ≠ zero! |
| 12 | Years employed validation | Cap at 45, remove negatives |
| 13 | Pincode validation | 6-digit India format |
| 14 | Target variable check | Only 0 or 1 allowed |

---

## 📊 EDA Sections (4 Complete Sections)

- **Section A** — Basic profiling: missing %, dtypes, numeric summary
- **Section B** — Univariate: income outliers, CIBIL dist, employment freq
- **Section C** — Bivariate: default rate vs CIBIL, employment, FOIR, state, age
- **Section D** — Distributions + WOE/IV encoding for all categorical features

---

## 🤖 Model Pipeline

- **Logistic Regression** — Interpretable scorecard base model
- **Gradient Boosting** — Production PD model (AUC ~0.84, KS ~0.56)
- **Credit Scorecard** — 300-900 score (like CIBIL)
- **PSI Monitoring** — Population Stability Index for model drift
- **Decile Analysis** — Business-standard model evaluation
- **WOE Encoding** — Weight of Evidence for categorical features

---

## 🖥️ Streamlit App (4 Pages)

1. **Overview** — KPIs, pipeline diagram, cleaning summary
2. **EDA Explorer** — Interactive charts, distributions, correlations
3. **Model Performance** — AUC, KS, Gini, decile table, feature importance
4. **Live Predictor** — Enter applicant details → instant credit decision

---

## 💡 Key Finance Domain Techniques

```python
# NTC Strategy — Most important, most misunderstood
df['ntc_flag']    = df['cibil_score'].isna().astype(int)
df['cibil_score'] = df['cibil_score'].fillna(-1)  # NOT mean!

# Winsorization — Industry standard (not dropping!)
p1, p99 = df['monthly_income'].quantile([0.01, 0.99])
df['monthly_income'] = df['monthly_income'].clip(p1, p99)

# DPD Sentinel — Missing DPD ≠ zero
df['dpd_90_count_missing'] = df['dpd_90_count'].isna().astype(int)
df['dpd_90_count'] = df['dpd_90_count'].fillna(-1)

# FOIR — Fixed Obligation to Income Ratio
df['foir'] = (df['monthly_emi_est'] / df['monthly_income']).clip(0, 0.95)

# WOE — Weight of Evidence for categorical encoding
df['cibil_band_woe'] = df['cibil_band'].map(woe_dict)
```
