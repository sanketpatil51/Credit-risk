"""
STEP 2: DATA CLEANING + FULL EDA
Complete production-grade pipeline for finance loan data.
All techniques used in real NBFC/fintech data science teams.
"""

import numpy as np
import pandas as pd
import re, os, json, warnings, time
from datetime import datetime
warnings.filterwarnings('ignore')

RAW   = '../data/raw_loan_data.csv'
CLEAN = '../data/clean_loan_data.csv'
EDA   = '../reports/eda_report.json'
PAN_REGEX = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$')

print("=" * 60)
print("  STEP 2: DATA CLEANING + EDA")
print("=" * 60)

# ════════════════════════════════════════════════════════
# 0. LOAD
# ════════════════════════════════════════════════════════
t0 = time.time()
df = pd.read_csv(RAW, low_memory=False)
raw_shape = df.shape
print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"  Missing cells: {df.isnull().sum().sum():,}")

# ════════════════════════════════════════════════════════
# EDA SECTION A — BASIC PROFILING (before cleaning)
# ════════════════════════════════════════════════════════
print("\n── EDA A: Basic Profiling ───────────────────────────")

eda = {}

# Missing value profile
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
eda['missing_pct_before'] = missing_pct[missing_pct > 0].to_dict()
print(f"  Columns with missing data: {(missing_pct > 0).sum()}")
print(f"  Highest missing: {missing_pct.idxmax()} = {missing_pct.max():.1f}%")

# Data types
eda['dtypes'] = df.dtypes.astype(str).to_dict()

# Numeric summary
num_cols = ['age','monthly_income','loan_amount','cibil_score',
            'credit_utilization','dpd_30_count','dpd_60_count','dpd_90_count',
            'bureau_inquiries_6m','existing_loans','interest_rate','years_employed']
num_summary = {}
for c in num_cols:
    s = pd.to_numeric(df[c], errors='coerce')
    num_summary[c] = {
        'mean': round(float(s.mean()), 2),
        'median': round(float(s.median()), 2),
        'std': round(float(s.std()), 2),
        'min': round(float(s.min()), 2),
        'max': round(float(s.max()), 2),
        'q1': round(float(s.quantile(0.25)), 2),
        'q3': round(float(s.quantile(0.75)), 2),
        'skew': round(float(s.skew()), 3),
        'missing_pct': round(float(s.isna().mean()*100), 2)
    }
eda['numeric_summary'] = num_summary
print(f"  Mean CIBIL score  : {num_summary['cibil_score']['mean']}")
print(f"  Median income     : {num_summary['monthly_income']['median']:,.0f}")
print(f"  Loan amount range : {num_summary['loan_amount']['min']:,.0f} – {num_summary['loan_amount']['max']:,.0f}")

# ════════════════════════════════════════════════════════
# EDA SECTION B — UNIVARIATE (before cleaning)
# ════════════════════════════════════════════════════════
print("\n── EDA B: Univariate Analysis ───────────────────────")

# Employment type frequency
emp_freq = df['employment_type'].value_counts(dropna=False).head(20).to_dict()
eda['employment_raw_freq'] = {str(k): int(v) for k,v in emp_freq.items()}
print(f"  Unique employment spellings: {df['employment_type'].nunique()}")

# Gender freq
gen_freq = df['gender'].value_counts(dropna=False).head(15).to_dict()
eda['gender_raw_freq'] = {str(k): int(v) for k,v in gen_freq.items()}

# Loan purpose
pur_freq = df['loan_purpose'].value_counts(dropna=False).to_dict()
eda['loan_purpose_freq'] = {str(k): int(v) for k,v in pur_freq.items()}

# State distribution
state_freq = df['state'].value_counts().head(15).to_dict()
eda['state_freq'] = {str(k): int(v) for k,v in state_freq.items()}

# CIBIL score distribution buckets
cibil_raw = pd.to_numeric(df['cibil_score'], errors='coerce')
eda['cibil_hist'] = {
    'below_600': int((cibil_raw < 600).sum()),
    '600_649':   int(cibil_raw.between(600,649).sum()),
    '650_699':   int(cibil_raw.between(650,699).sum()),
    '700_749':   int(cibil_raw.between(700,749).sum()),
    '750_plus':  int((cibil_raw >= 750).sum()),
    'missing':   int(cibil_raw.isna().sum()),
}
print(f"  CIBIL missing     : {eda['cibil_hist']['missing']:,} ({eda['cibil_hist']['missing']/len(df):.1%})")

# Income outlier check
inc_raw = pd.to_numeric(df['monthly_income'], errors='coerce')
eda['income_outliers'] = {
    'below_0':     int((inc_raw < 0).sum()),
    'below_10000': int((inc_raw < 10000).sum()),
    'above_500000': int((inc_raw > 500000).sum()),
}
print(f"  Income outliers   : {sum(eda['income_outliers'].values()):,}")

# Default rate raw
dr_raw = pd.to_numeric(df['default_12m'], errors='coerce').mean()
eda['default_rate_raw'] = round(float(dr_raw), 4)
print(f"  Raw default rate  : {dr_raw:.1%}")

# Duplicate count
dup_count = df.duplicated().sum()
eda['duplicate_rows'] = int(dup_count)
print(f"  Duplicate rows    : {dup_count:,}")

# ════════════════════════════════════════════════════════
# CLEANING STEP 1 — REMOVE DUPLICATES
# ════════════════════════════════════════════════════════
print("\n── CLEANING: Step 1 — Duplicate Removal ────────────")
df['_app_date_ts'] = pd.to_datetime(df['application_date'], errors='coerce')
df = df.sort_values('_app_date_ts', ascending=False, na_position='last')
df['repeat_applicant_flag'] = df.duplicated(subset=['pan_number'], keep='first').astype(int)
df = df.drop_duplicates(subset=['pan_number'], keep='first')
df = df.drop_duplicates(subset=[c for c in df.columns if not c.startswith('_')])
print(f"  After dedup: {len(df):,} rows")

# ════════════════════════════════════════════════════════
# CLEANING STEP 2 — PAN VALIDATION
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 2 — PAN Validation ───────────────")
df['pan_number'] = df['pan_number'].astype(str).str.strip().str.upper()
df['pan_valid_flag'] = df['pan_number'].apply(
    lambda x: 1 if isinstance(x, str) and PAN_REGEX.match(x) else 0
)
print(f"  Invalid PANs flagged: {(df['pan_valid_flag']==0).sum():,}")

# ════════════════════════════════════════════════════════
# CLEANING STEP 3 — DATE PARSING
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 3 — Date Parsing ─────────────────")
df['application_date']   = pd.to_datetime(df['_app_date_ts'], errors='coerce')
df['disbursement_date']  = pd.to_datetime(df['disbursement_date'], errors='coerce')
df['bureau_last_updated']= pd.to_datetime(df['bureau_last_updated'], errors='coerce')
df['app_year']    = df['application_date'].dt.year.fillna(2022).astype(int)
df['app_month']   = df['application_date'].dt.month.fillna(1).astype(int)
df['app_quarter'] = df['application_date'].dt.quarter.fillna(1).astype(int)
df['bureau_age_days'] = (df['application_date'] - df['bureau_last_updated']).dt.days
df['bureau_stale_flag'] = (df['bureau_age_days'] > 90).fillna(False).astype(int)
print(f"  Stale bureaus (>90d): {df['bureau_stale_flag'].sum():,}")

# ════════════════════════════════════════════════════════
# CLEANING STEP 4 — AGE
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 4 — Age ───────────────────────────")
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.loc[~df['age'].between(21,65), 'age'] = np.nan
df['age'] = df.groupby('state')['age'].transform(lambda x: x.fillna(x.median()))
df['age'] = df['age'].fillna(df['age'].median()).round(0).astype(int)

# ════════════════════════════════════════════════════════
# CLEANING STEP 5 — GENDER STANDARDIZATION
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 5 — Gender ───────────────────────")
gender_map = {'male':'Male','m':'Male','female':'Female','f':'Female','other':'Other'}
df['gender'] = df['gender'].astype(str).str.strip().str.lower().map(gender_map).fillna('Unknown')

# ════════════════════════════════════════════════════════
# CLEANING STEP 6 — EMPLOYMENT TYPE
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 6 — Employment Standardization ───")
emp_std = {
    'salaried':'Salaried','sal.':'Salaried','salried':'Salaried','sal':'Salaried',
    'govt salaried':'Salaried','pvt salaried':'Salaried',
    'self-employed':'Self-Employed','self employed':'Self-Employed',
    'se':'Self-Employed','s.e.':'Self-Employed','selfemployed':'Self-Employed',
    'business':'Business','businessman':'Business','businesswoman':'Business',
    'proprietor':'Business','business owner':'Business',
    'freelancer':'Freelancer','free lancer':'Freelancer',
    'freelance':'Freelancer','contract':'Freelancer','contractual':'Freelancer',
}
df['employment_type'] = (df['employment_type'].astype(str).str.strip().str.lower()
                         .map(emp_std).fillna('Salaried'))
print(f"  Unique after standardize: {df['employment_type'].nunique()}")

# ════════════════════════════════════════════════════════
# CLEANING STEP 7 — INCOME WINSORIZATION
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 7 — Income Winsorization ─────────")
df['monthly_income'] = pd.to_numeric(df['monthly_income'], errors='coerce')
df.loc[df['monthly_income'] <= 0, 'monthly_income'] = np.nan
p1  = df['monthly_income'].quantile(0.01)
p99 = df['monthly_income'].quantile(0.99)
df['monthly_income'] = df['monthly_income'].clip(p1, p99)
df['monthly_income'] = (df.groupby('employment_type')['monthly_income']
                          .transform(lambda x: x.fillna(x.median())))
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
print(f"  Income range after: ₹{p1:,.0f} – ₹{p99:,.0f}")

# ════════════════════════════════════════════════════════
# CLEANING STEP 8 — LOAN AMOUNT
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 8 — Loan Amount ──────────────────")
df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
before_loan = len(df)
df = df[df['loan_amount'].isna() | (df['loan_amount'] > 0)]
la_p1, la_p99 = df['loan_amount'].quantile([0.005, 0.995])
df['loan_amount'] = df['loan_amount'].clip(la_p1, la_p99)
df['loan_amount'] = df['loan_amount'].fillna(df['loan_amount'].median())
print(f"  Removed {before_loan - len(df):,} invalid loan rows")

# ════════════════════════════════════════════════════════
# CLEANING STEP 9 — INTEREST RATE
# ════════════════════════════════════════════════════════
df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce')
df.loc[~df['interest_rate'].between(5, 60), 'interest_rate'] = np.nan
df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())

# ════════════════════════════════════════════════════════
# CLEANING STEP 10 — CIBIL / NTC STRATEGY
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 10 — CIBIL NTC Strategy ──────────")
df['cibil_score'] = pd.to_numeric(df['cibil_score'], errors='coerce')
df.loc[~df['cibil_score'].between(300,900), 'cibil_score'] = np.nan
df['ntc_flag']    = df['cibil_score'].isna().astype(int)
df['cibil_score'] = df['cibil_score'].fillna(-1)  # sentinel
print(f"  NTC (missing CIBIL): {df['ntc_flag'].sum():,} ({df['ntc_flag'].mean():.1%})")

# ════════════════════════════════════════════════════════
# CLEANING STEP 11 — BUREAU DPD + CREDIT FEATURES
# ════════════════════════════════════════════════════════
print("── CLEANING: Step 11 — Bureau / DPD Features ────────")
dpd_caps = {'dpd_30_count':10,'dpd_60_count':8,'dpd_90_count':5,
            'bureau_inquiries_6m':20,'oldest_account_age_months':360,
            'existing_loans':15,'number_of_credit_cards':12}
for col, cap in dpd_caps.items():
    if col not in df.columns: continue
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[df[col] < 0, col] = np.nan
    df[col] = df[col].clip(0, cap)
    df[f'{col}_missing'] = df[col].isna().astype(int)
    df[col] = df[col].fillna(-1)  # unknown sentinel

df['credit_utilization'] = pd.to_numeric(df['credit_utilization'], errors='coerce')
df.loc[~df['credit_utilization'].between(0,1), 'credit_utilization'] = np.nan
df['credit_utilization'] = df['credit_utilization'].fillna(df['credit_utilization'].median())

df['total_credit_limit'] = pd.to_numeric(df['total_credit_limit'], errors='coerce')
df.loc[df['total_credit_limit'] <= 0, 'total_credit_limit'] = np.nan
df['total_credit_limit'] = df['total_credit_limit'].fillna(df['total_credit_limit'].median())

# ════════════════════════════════════════════════════════
# CLEANING STEP 12 — YEARS EMPLOYED
# ════════════════════════════════════════════════════════
df['years_employed'] = pd.to_numeric(df['years_employed'], errors='coerce')
df.loc[~df['years_employed'].between(0, 45), 'years_employed'] = np.nan
df['years_employed'] = df['years_employed'].fillna(df['years_employed'].median())

# ════════════════════════════════════════════════════════
# CLEANING STEP 13 — PINCODE
# ════════════════════════════════════════════════════════
df['pincode'] = pd.to_numeric(df['pincode'], errors='coerce')
df['pincode_valid_flag'] = df['pincode'].between(100000, 999999).fillna(False).astype(int)
df.loc[df['pincode_valid_flag']==0, 'pincode'] = np.nan

# ════════════════════════════════════════════════════════
# CLEANING STEP 14 — TARGET VARIABLE
# ════════════════════════════════════════════════════════
df['default_12m'] = pd.to_numeric(df['default_12m'], errors='coerce')
df = df[df['default_12m'].isin([0,1])]

# ════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════
print("\n── FEATURE ENGINEERING ──────────────────────────────")
# Financial ratios
df['monthly_emi_est']     = (df['loan_amount'] / df['loan_tenure_months'] * 1.10).round(0)
df['foir']                = (df['monthly_emi_est'] / df['monthly_income']).clip(0, 0.95).round(4)
df['income_to_loan_ratio']= (df['monthly_income'] * 12 / df['loan_amount']).round(4)
df['annual_income']       = (df['monthly_income'] * 12).round(0)
df['loan_to_income_ratio']= (df['loan_amount'] / df['annual_income']).clip(0, 20).round(4)

# Delinquency
df['delinquency_score']   = (df['dpd_30_count'].clip(0)*1 +
                              df['dpd_60_count'].clip(0)*2 +
                              df['dpd_90_count'].clip(0)*4)
df['any_dpd_90_flag']     = (df['dpd_90_count'] > 0).astype(int)
df['any_delinquency_flag']= ((df['dpd_30_count'] > 0) | (df['dpd_60_count'] > 0) |
                              (df['dpd_90_count'] > 0)).astype(int)

# Bureau stress
df['bureau_stress_index'] = ((df['bureau_inquiries_6m'] > 3).astype(int) +
                              (df['credit_utilization'] > 0.75).astype(int) +
                              (df['existing_loans'] > 3).astype(int) +
                              df['ntc_flag'])

# Risk flags
df['high_foir_flag']      = (df['foir'] > 0.60).astype(int)
df['low_income_flag']     = (df['monthly_income'] < 25000).astype(int)
df['high_loan_flag']      = (df['loan_amount'] > 500000).astype(int)

# CIBIL band
cibil_vals = df['cibil_score'].replace(-1, np.nan)
df['cibil_band'] = pd.cut(cibil_vals, bins=[299,599,649,699,749,900],
    labels=['1_poor','2_below_avg','3_average','4_good','5_excellent']).astype(str)
df.loc[df['ntc_flag']==1, 'cibil_band'] = '0_ntc'

# Employment stability
emp_stab = {'Salaried':3,'Business':2,'Self-Employed':1,'Freelancer':0}
df['employment_stability'] = df['employment_type'].map(emp_stab).fillna(1).astype(int)

# Loan size category
df['loan_size_cat'] = pd.cut(df['loan_amount'],
    bins=[0,100000,300000,600000,10000000],
    labels=['Small','Medium','Large','Very Large']).astype(str)

# Age bands
df['age_band'] = pd.cut(df['age'], bins=[20,30,40,50,65],
    labels=['21-30','31-40','41-50','51-65']).astype(str)

# Drop helper cols
df.drop(columns=['_app_date_ts'], errors='ignore', inplace=True)

print(f"  Feature columns now: {df.shape[1]}")

# ════════════════════════════════════════════════════════
# EDA SECTION C — BIVARIATE (after cleaning, for model insight)
# ════════════════════════════════════════════════════════
print("\n── EDA C: Bivariate Analysis (Post-Clean) ───────────")

# Default rate by CIBIL band
cibil_dr = df.groupby('cibil_band')['default_12m'].agg(['mean','count']).round(4)
eda['dr_by_cibil_band'] = {str(k): {'dr': round(float(v['mean']),4), 'count': int(v['count'])}
                            for k,v in cibil_dr.iterrows()}

# Default rate by employment
emp_dr = df.groupby('employment_type')['default_12m'].agg(['mean','count']).round(4)
eda['dr_by_employment'] = {str(k): {'dr': round(float(v['mean']),4), 'count': int(v['count'])}
                            for k,v in emp_dr.iterrows()}

# Default rate by FOIR bucket
df['foir_band'] = pd.cut(df['foir'], bins=[0,0.3,0.5,0.6,0.75,1.0],
    labels=['<30%','30-50%','50-60%','60-75%','>75%'])
foir_dr = df.groupby('foir_band', observed=True)['default_12m'].agg(['mean','count'])
eda['dr_by_foir'] = {str(k): {'dr': round(float(v['mean']),4), 'count': int(v['count'])}
                     for k,v in foir_dr.iterrows()}

# Default rate by state
state_dr = df.groupby('state')['default_12m'].mean().round(4).sort_values(ascending=False)
eda['dr_by_state'] = {str(k): round(float(v),4) for k,v in state_dr.items()}

# Default rate by age band
age_dr = df.groupby('age_band')['default_12m'].mean().round(4)
eda['dr_by_age_band'] = {str(k): round(float(v),4) for k,v in age_dr.items()}

# Default rate by loan purpose
pur_dr = df.groupby('loan_purpose')['default_12m'].mean().round(4).sort_values(ascending=False)
eda['dr_by_purpose'] = {str(k): round(float(v),4) for k,v in pur_dr.items()}

# Correlation matrix (numeric)
num_model_cols = ['age','monthly_income','loan_amount','loan_tenure_months',
                  'cibil_score','foir','delinquency_score','bureau_stress_index',
                  'credit_utilization','years_employed','default_12m']
corr = df[num_model_cols].corr().round(3)
eda['correlation_with_default'] = {c: round(float(corr.loc[c,'default_12m']),4)
                                    for c in num_model_cols if c != 'default_12m'}

# ════════════════════════════════════════════════════════
# EDA SECTION D — DISTRIBUTION STATS (post-clean)
# ════════════════════════════════════════════════════════
clean_num_summary = {}
for c in ['age','monthly_income','loan_amount','cibil_score','foir',
          'credit_utilization','delinquency_score','interest_rate','years_employed']:
    if c not in df.columns: continue
    s = pd.to_numeric(df[c].replace(-1, np.nan), errors='coerce')
    clean_num_summary[c] = {
        'mean':   round(float(s.mean()),2),
        'median': round(float(s.median()),2),
        'std':    round(float(s.std()),2),
        'min':    round(float(s.min()),2),
        'max':    round(float(s.max()),2),
        'skew':   round(float(s.skew()),3),
    }
eda['clean_numeric_summary'] = clean_num_summary

eda['final_default_rate'] = round(float(df['default_12m'].mean()), 4)
eda['final_rows']         = len(df)
eda['final_cols']         = df.shape[1]
eda['ntc_count']          = int(df['ntc_flag'].sum())
eda['stale_bureau_count'] = int(df['bureau_stale_flag'].sum())
eda['missing_pct_after']  = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

# Income distribution
inc_bins = pd.cut(df['monthly_income'],
    bins=[0,20000,40000,60000,80000,100000,150000,999999],
    labels=['<20k','20-40k','40-60k','60-80k','80-100k','100-150k','150k+'])
eda['income_dist'] = {str(k): int(v) for k,v in inc_bins.value_counts().sort_index().items()}

# CIBIL dist post-clean
cibil_post = df['cibil_score'].replace(-1, np.nan)
eda['cibil_dist_clean'] = {
    'NTC':    int(df['ntc_flag'].sum()),
    '<600':   int((cibil_post < 600).sum()),
    '600-649':int(cibil_post.between(600,649).sum()),
    '650-699':int(cibil_post.between(650,699).sum()),
    '700-749':int(cibil_post.between(700,749).sum()),
    '750+':   int((cibil_post >= 750).sum()),
}

# Loan purpose dist
eda['loan_purpose_clean_dist'] = df['loan_purpose'].value_counts().to_dict()

# ════════════════════════════════════════════════════════
# WOE CALCULATION
# ════════════════════════════════════════════════════════
print("\n── WOE Encoding ─────────────────────────────────────")
def calc_woe_iv(df, feature, target):
    temp = df.groupby(feature)[target].agg(['sum','count'])
    temp.columns = ['bads','total']
    temp['goods']     = temp['total'] - temp['bads']
    temp['pct_bad']   = (temp['bads']  / temp['bads'].sum()).replace(0, 0.0001)
    temp['pct_good']  = (temp['goods'] / temp['goods'].sum()).replace(0, 0.0001)
    temp['woe']       = np.log(temp['pct_good'] / temp['pct_bad'])
    temp['iv_contrib']= (temp['pct_good'] - temp['pct_bad']) * temp['woe']
    iv = temp['iv_contrib'].sum()
    return temp['woe'].to_dict(), round(iv, 4)

woe_cibil, iv_cibil = calc_woe_iv(df, 'cibil_band', 'default_12m')
woe_emp,   iv_emp   = calc_woe_iv(df, 'employment_type', 'default_12m')
woe_age,   iv_age   = calc_woe_iv(df, 'age_band', 'default_12m')
woe_loan,  iv_loan  = calc_woe_iv(df, 'loan_size_cat', 'default_12m')

eda['woe_iv'] = {
    'cibil_band':      {'woe':{str(k):round(float(v),4) for k,v in woe_cibil.items()}, 'iv':iv_cibil},
    'employment_type': {'woe':{str(k):round(float(v),4) for k,v in woe_emp.items()},   'iv':iv_emp},
    'age_band':        {'woe':{str(k):round(float(v),4) for k,v in woe_age.items()},   'iv':iv_age},
    'loan_size_cat':   {'woe':{str(k):round(float(v),4) for k,v in woe_loan.items()},  'iv':iv_loan},
}

# Apply WOE encoding
df['cibil_band_woe']  = df['cibil_band'].map(woe_cibil).fillna(0)
df['emp_type_woe']    = df['employment_type'].map(woe_emp).fillna(0)
df['age_band_woe']    = df['age_band'].map(woe_age).fillna(0)
df['loan_size_woe']   = df['loan_size_cat'].map(woe_loan).fillna(0)

print(f"  IV — CIBIL band : {iv_cibil} {'(Strong)' if iv_cibil>0.3 else '(Moderate)'}")
print(f"  IV — Emp type   : {iv_emp}")
print(f"  IV — Age band   : {iv_age}")

# ════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ════════════════════════════════════════════════════════
df.to_csv(CLEAN, index=False)
os.makedirs('../reports', exist_ok=True)
with open(EDA, 'w') as f:
    json.dump(eda, f, indent=2)

print(f"\n{'='*60}")
print(f"  ✅ CLEANING + EDA COMPLETE")
print(f"{'='*60}")
print(f"  Raw rows        : {raw_shape[0]:,}")
print(f"  Clean rows      : {len(df):,}")
print(f"  Feature columns : {df.shape[1]}")
print(f"  Default rate    : {df['default_12m'].mean():.1%}")
print(f"  Runtime         : {time.time()-t0:.1f}s")
print(f"  💾 Clean data  → data/clean_loan_data.csv")
print(f"  💾 EDA report  → reports/eda_report.json")
