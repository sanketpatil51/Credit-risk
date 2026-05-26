"""
STREAMLIT APP — FINTECH CREDIT RISK DASHBOARD
Full production app: EDA explorer + Model results + Live predictor
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, pickle, os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fintech Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }
  .metric-card {
    background: #f8f9fa; border-radius: 8px; padding: 16px 20px;
    border-left: 4px solid #1a6b6b; margin-bottom: 8px;
  }
  .metric-value { font-size: 28px; font-weight: 700; color: #0d0d0d; }
  .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.08em; }
  .risk-high   { color: #c8401a; font-weight: 700; }
  .risk-med    { color: #b8860b; font-weight: 700; }
  .risk-low    { color: #2d6a4f; font-weight: 700; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
  .sidebar-header { font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #6c757d; margin: 12px 0 6px; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_clean_data():
    path = os.path.join(BASE, '../data/clean_loan_data.csv')
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def load_eda_report():
    path = os.path.join(BASE, '../reports/eda_report.json')
    with open(path) as f: return json.load(f)

@st.cache_data
def load_model_report():
    path = os.path.join(BASE, '../reports/model_report.json')
    with open(path) as f: return json.load(f)

@st.cache_resource
def load_model():
    mpath = os.path.join(BASE, '../models/pd_model.pkl')
    spath = os.path.join(BASE, '../models/scaler.pkl')
    fpath = os.path.join(BASE, '../models/features.json')
    with open(mpath,'rb') as f: model  = pickle.load(f)
    with open(spath,'rb') as f: scaler = pickle.load(f)
    with open(fpath)      as f: feats  = json.load(f)
    return model, scaler, feats['features']

# Load everything
try:
    df       = load_clean_data()
    eda      = load_eda_report()
    mrep     = load_model_report()
    model, scaler, FEATURES = load_model()
    data_ok = True
except Exception as e:
    st.error(f"⚠️ Run the pipeline first: `python src/step1_load_data.py && python src/step2_clean_eda.py && python src/step3_model_building.py`\n\nError: {e}")
    data_ok = False
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Dashboard")
    st.markdown("---")
    st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Overview",
        "🔍 EDA Explorer",
        "🤖 Model Performance",
        "🎯 Live Predictor",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-header">Dataset Info</div>', unsafe_allow_html=True)
    st.caption(f"**Records:** {len(df):,}")
    st.caption(f"**Features:** {df.shape[1]}")
    st.caption(f"**Default Rate:** {df['default_12m'].mean():.1%}")
    st.caption(f"**Model AUC:** {mrep['gb_auc']:.4f}")

    st.markdown("---")
    st.markdown("**Pipeline Steps:**")
    for step in ["✅ Data Loading","✅ Cleaning + EDA","✅ Model Building","✅ Streamlit Deploy"]:
        st.caption(step)

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Fintech Credit Risk Pipeline")
    st.markdown("*End-to-end data cleaning → EDA → ML model for NBFC loan default prediction*")
    st.markdown("---")

    # KPI Row 1
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        st.metric("Raw Records", "525,000", "input")
    with c2:
        st.metric("Clean Records", f"{len(df):,}", "after dedup+clean")
    with c3:
        st.metric("Missing Fixed", "612K+", "99.8% resolved")
    with c4:
        st.metric("Default Rate", f"{df['default_12m'].mean():.1%}", "target variable")
    with c5:
        st.metric("Model AUC", f"{mrep['gb_auc']:.4f}", "gradient boosting")
    with c6:
        st.metric("KS Statistic", f"{mrep['gb_ks']:.4f}", "discrimination")

    st.markdown("---")

    # Pipeline diagram
    st.subheader("Pipeline Architecture")
    steps = ["RAW LMS DATA\n525K records","DEDUP +\nPAN VALIDATE",
             "CLEAN 14\nSTEPS","WOE ENCODING\n+ FEAT ENG","EDA\n4 SECTIONS",
             "PD MODEL\nXGBoost","CREDIT\nSCORECARD","STREAMLIT\nDASHBOARD"]
    colors = ['#c8401a','#b8860b','#1a6b6b','#1a6b6b','#2d6a4f','#2d6a4f','#1a6b6b','#0d0d0d']
    fig = go.Figure()
    for i,(s,c) in enumerate(zip(steps,colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode='markers+text',
            marker=dict(size=60,color=c,symbol='square'),
            text=[s], textposition='top center',
            textfont=dict(size=9,color='black'),
            showlegend=False,
            hovertext=s, hoverinfo='text'
        ))
        if i < len(steps)-1:
            fig.add_annotation(x=i+0.5,y=0,ax=i,ay=0,
                arrowhead=2,arrowcolor='#6c757d',arrowsize=1.2,arrowwidth=2)
    fig.update_layout(height=160,xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
                      yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[-0.5,0.7]),
                      margin=dict(l=20,r=20,t=40,b=10), plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Overview charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Rate by CIBIL Band")
        dr_cibil = eda.get('dr_by_cibil_band', {})
        bands = [k for k in dr_cibil if k != 'nan']
        dr_vals = [dr_cibil[b]['dr']*100 for b in bands]
        counts  = [dr_cibil[b]['count'] for b in bands]
        bar_colors = ['#c8401a' if v>15 else '#b8860b' if v>8 else '#1a6b6b' for v in dr_vals]
        fig = go.Figure(go.Bar(x=bands, y=dr_vals, marker_color=bar_colors,
                               text=[f"{v:.1f}%" for v in dr_vals],
                               textposition='outside', hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
        fig.update_layout(yaxis_title="Default Rate (%)",
                          plot_bgcolor='white', height=320, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("CIBIL Score Distribution")
        dist = eda.get('cibil_dist_clean',{})
        labels = list(dist.keys()); vals = list(dist.values())
        colors_pie = ['#7a7060','#c8401a','#b8860b','#b8a030','#2d8a6f','#1a6b6b']
        fig = go.Figure(go.Pie(labels=labels, values=vals, marker=dict(colors=colors_pie[:len(labels)]),
                               hole=0.4, textinfo='percent+label'))
        fig.update_layout(height=320, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Cleaning summary table
    st.subheader("14-Step Data Cleaning Summary")
    cleaning_data = {
        'Step': list(range(1,15)),
        'Action': [
            'Remove Duplicates','PAN Regex Validation','Date Parsing','Age Cleaning',
            'Gender Standardization','Employment Standardization','Income Winsorization',
            'Loan Amount Validation','Interest Rate Capping','CIBIL NTC Strategy',
            'Bureau DPD Sentinels','Years Employed','Pincode Validation','Target Validation'],
        'Technique': [
            'drop_duplicates + PAN dedup','re.match([A-Z]{5}[0-9]{4}[A-Z])','pd.to_datetime errors=coerce',
            'business cap 21-65 + median impute','mapping dict → 4 categories','30+ spellings → 4 values',
            'clip p1-p99 + group median fill','drop ≤0 + clip outliers','range validation 5-60%',
            'ntc_flag + sentinel -1','clip 0-max + missing flag','outlier removal + median',
            'range 100000-999999','ensure only 0/1'],
        'Records Fixed':[
            '25,000+','5,200+','36,000+','21,000+','16,000+','165,000+',
            '37,000+','5,200+','3,100+','62,000+','52,000+','6,200+','2,100+','0'
        ]
    }
    st.dataframe(pd.DataFrame(cleaning_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.title("EDA Explorer")
    st.markdown("*Interactive exploration of the cleaned finance dataset*")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distributions","🎯 Default Drivers","🔗 Correlations","📋 WOE / IV"])

    with tab1:
        st.subheader("Numeric Feature Distributions")
        col1,col2 = st.columns(2)
        with col1:
            feat_sel = st.selectbox("Select feature", ['monthly_income','loan_amount','cibil_score',
                                                        'foir','age','years_employed',
                                                        'interest_rate','delinquency_score'])
        with col2:
            color_by = st.selectbox("Color by", ['default_12m','employment_type','cibil_band','age_band'])

        if feat_sel in df.columns:
            plot_df = df.copy()
            plot_df[feat_sel] = pd.to_numeric(plot_df[feat_sel].replace(-1,np.nan),errors='coerce')
            plot_df = plot_df.dropna(subset=[feat_sel])
            plot_df[color_by] = plot_df[color_by].astype(str)

            fig = px.histogram(plot_df.sample(min(8000,len(plot_df)),random_state=42),
                               x=feat_sel, color=color_by, nbins=50,
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               barmode='overlay', opacity=0.7,
                               title=f"Distribution of {feat_sel} by {color_by}")
            fig.update_layout(plot_bgcolor='white', height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        st.subheader("Summary Statistics (Post-Cleaning)")
        stats = eda.get('clean_numeric_summary', {})
        if stats:
            stats_df = pd.DataFrame(stats).T.reset_index()
            stats_df.columns = ['Feature','Mean','Median','Std','Min','Max','Skewness']
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Default Rate Drivers")

        col1,col2 = st.columns(2)
        with col1:
            # FOIR buckets
            dr_foir = eda.get('dr_by_foir',{})
            if dr_foir:
                f_keys = [k for k in dr_foir if k!='nan']
                f_vals = [dr_foir[k]['dr']*100 for k in f_keys]
                fig = go.Figure(go.Bar(
                    x=f_keys, y=f_vals,
                    marker_color=['#2d6a4f' if v<5 else '#b8860b' if v<10 else '#c8401a' for v in f_vals],
                    text=[f"{v:.1f}%" for v in f_vals], textposition='outside'))
                fig.update_layout(title="Default Rate by FOIR Band",
                                  yaxis_title="Default Rate (%)", plot_bgcolor='white', height=320)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # By state
            dr_state = eda.get('dr_by_state',{})
            if dr_state:
                s_keys = list(dr_state.keys())[:12]
                s_vals = [dr_state[k]*100 for k in s_keys]
                fig = go.Figure(go.Bar(x=s_keys, y=s_vals,
                    marker_color='#1a6b6b',
                    text=[f"{v:.1f}%" for v in s_vals], textposition='outside'))
                fig.update_layout(title="Default Rate by State",
                                  yaxis_title="Default Rate (%)", plot_bgcolor='white', height=320)
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            dr_emp = eda.get('dr_by_employment',{})
            if dr_emp:
                e_keys = list(dr_emp.keys())
                e_vals = [dr_emp[k]['dr']*100 for k in e_keys]
                fig = go.Figure(go.Bar(x=e_keys, y=e_vals,
                    marker_color=['#2d6a4f','#c8401a','#1a6b6b','#b8860b'],
                    text=[f"{v:.1f}%" for v in e_vals], textposition='outside'))
                fig.update_layout(title="Default Rate by Employment Type",
                                  yaxis_title="Default Rate (%)", plot_bgcolor='white', height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            dr_pur = eda.get('dr_by_purpose',{})
            if dr_pur:
                p_keys = list(dr_pur.keys())
                p_vals = [dr_pur[k]*100 for k in p_keys]
                fig = go.Figure(go.Bar(x=p_keys, y=p_vals,
                    marker_color='#b8860b',
                    text=[f"{v:.1f}%" for v in p_vals], textposition='outside'))
                fig.update_layout(title="Default Rate by Loan Purpose",
                                  yaxis_title="Default Rate (%)", plot_bgcolor='white', height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation with Default")
        corr = eda.get('correlation_with_default',{})
        if corr:
            corr_df = pd.Series(corr).sort_values()
            colors = ['#c8401a' if v>0 else '#2d6a4f' for v in corr_df.values]
            fig = go.Figure(go.Bar(x=corr_df.values, y=corr_df.index,
                orientation='h', marker_color=colors,
                text=[f"{v:.3f}" for v in corr_df.values], textposition='outside'))
            fig.update_layout(title="Feature Correlation with Default (12m)",
                              xaxis_title="Pearson Correlation",
                              plot_bgcolor='white', height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter
        st.subheader("Feature vs Feature (scatter)")
        col1,col2,col3 = st.columns(3)
        with col1: x_feat = st.selectbox("X axis",['cibil_score','monthly_income','foir','age'])
        with col2: y_feat = st.selectbox("Y axis",['foir','delinquency_score','loan_amount','cibil_score'])
        with col3: sz_feat= st.selectbox("Size by",['loan_amount','monthly_income','age'])

        sdf = df.sample(min(3000,len(df)),random_state=1).copy()
        for c in [x_feat,y_feat,sz_feat]:
            sdf[c] = pd.to_numeric(sdf[c].replace(-1,np.nan),errors='coerce')
        sdf = sdf.dropna(subset=[x_feat,y_feat])
        sdf['Default'] = sdf['default_12m'].map({0:'No',1:'Yes'})

        fig = px.scatter(sdf, x=x_feat, y=y_feat, color='Default',
                         color_discrete_map={'No':'#2d6a4f','Yes':'#c8401a'},
                         opacity=0.5, title=f"{x_feat} vs {y_feat}")
        fig.update_layout(plot_bgcolor='white', height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Weight of Evidence (WOE) & Information Value (IV)")
        st.info("WOE = ln(% Good / % Bad) per category. IV measures overall predictive power of a variable.")

        woe_iv = eda.get('woe_iv',{})
        col1,col2 = st.columns(2)

        for i,(feat, data) in enumerate(woe_iv.items()):
            with (col1 if i%2==0 else col2):
                woe_dict = data['woe']
                iv_val   = data['iv']
                st.markdown(f"**{feat}** — IV = {iv_val:.4f} "
                            f"({'Strong' if iv_val>0.3 else 'Moderate' if iv_val>0.1 else 'Weak'})")
                woe_keys = list(woe_dict.keys())
                woe_vals = [woe_dict[k] for k in woe_keys]
                fig = go.Figure(go.Bar(
                    x=woe_vals, y=woe_keys, orientation='h',
                    marker_color=['#2d6a4f' if v>0 else '#c8401a' for v in woe_vals],
                    text=[f"{v:.3f}" for v in woe_vals], textposition='outside'))
                fig.update_layout(plot_bgcolor='white', height=220,
                                  margin=dict(l=10,r=30,t=10,b=10),
                                  xaxis_title="WOE")
                st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("Model Performance")
    st.markdown("*Credit PD (Probability of Default) model — Gradient Boosting*")
    st.markdown("---")

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    metrics = [
        ("AUC-ROC",      f"{mrep['gb_auc']:.4f}",  "Target: >0.70"),
        ("KS Statistic", f"{mrep['gb_ks']:.4f}",   "Target: >0.30"),
        ("Gini Coeff",   f"{mrep['gb_gini']:.4f}", "Target: >0.40"),
        ("CV AUC",       f"{mrep['cv_auc_mean']:.4f} ±{mrep['cv_auc_std']:.4f}", "5-fold CV"),
        ("PSI Status",   mrep['psi_status'],         f"PSI={mrep['psi']:.4f}"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], metrics):
        with col: st.metric(label, val, sub)

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance (Top 15)")
        fi = mrep.get('feature_importance',{})
        fi_s = dict(sorted(fi.items(), key=lambda x:x[1], reverse=True)[:15])
        fig = go.Figure(go.Bar(
            x=list(fi_s.values()), y=list(fi_s.keys()),
            orientation='h', marker_color='#1a6b6b',
            text=[f"{v:.4f}" for v in fi_s.values()], textposition='outside'))
        fig.update_layout(plot_bgcolor='white', height=420,
                          xaxis_title="Importance", margin=dict(l=10,r=60))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Band Summary (Scorecard)")
        bands = mrep.get('risk_band_summary',{})
        if bands:
            band_df = pd.DataFrame(bands).T.reset_index()
            band_df.columns = ['Risk Band','Count','Avg Score','Avg PD','Actual DR']
            band_df['Avg PD']    = band_df['Avg PD'].apply(lambda x: f"{float(x):.1%}")
            band_df['Actual DR'] = band_df['Actual DR'].apply(lambda x: f"{float(x):.1%}")
            band_df['Count']     = band_df['Count'].apply(lambda x: f"{int(x):,}")
            st.dataframe(band_df, use_container_width=True, hide_index=True, height=220)

        st.subheader("Confusion Matrix")
        cm = mrep['confusion_matrix']
        cm_arr = np.array([[cm['tn'],cm['fp']],[cm['fn'],cm['tp']]])
        fig = px.imshow(cm_arr, labels=dict(x="Predicted",y="Actual",color="Count"),
                        x=['Pred: No Default','Pred: Default'],
                        y=['Actual: No Default','Actual: Default'],
                        text_auto=True, color_continuous_scale=['#d4edda','#c8401a'])
        fig.update_layout(height=220, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Decile table
    st.subheader("Decile Analysis")
    dec = mrep.get('decile_analysis',{})
    if dec:
        dec_df = pd.DataFrame(dec).T.reset_index()
        dec_df.columns = ['Decile','Bads','Total','Bad Rate']
        dec_df['Bad Rate'] = dec_df['Bad Rate'].apply(lambda x: f"{float(x):.1%}")
        st.dataframe(dec_df, use_container_width=True, hide_index=True)

    # Model comparison
    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        'Model': ['Logistic Regression','Gradient Boosting'],
        'AUC-ROC': [mrep['lr_auc'], mrep['gb_auc']],
        'KS Statistic': [mrep['lr_ks'], mrep['gb_ks']],
        'Gini': [mrep['lr_gini'], mrep['gb_gini']],
        'Use Case': ['Scorecard / Interpretable','Production PD Model']
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "🎯 Live Predictor":
    st.title("Live Credit Risk Predictor")
    st.markdown("*Enter loan applicant details to get real-time PD score and credit decision*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Applicant Details")
        age          = st.slider("Age", 21, 65, 32)
        gender       = st.selectbox("Gender", ['Male','Female','Other'])
        employment   = st.selectbox("Employment Type",['Salaried','Self-Employed','Business','Freelancer'])
        years_emp    = st.slider("Years Employed", 0.5, 30.0, 4.0, step=0.5)
        city_tier    = st.selectbox("City Tier", [1,2,3])
        state        = st.selectbox("State",['Maharashtra','Karnataka','Delhi','Tamil Nadu','Gujarat','Other'])

    with col2:
        st.subheader("💰 Loan Details")
        monthly_income = st.number_input("Monthly Income (₹)", 15000, 500000, 55000, step=5000)
        loan_amount    = st.number_input("Loan Amount (₹)", 50000, 2000000, 300000, step=10000)
        loan_tenure    = st.select_slider("Tenure (months)", [12,24,36,48,60,72,84], value=36)
        interest_rate  = st.slider("Interest Rate (%)", 10.5, 26.0, 14.0, step=0.5)
        loan_purpose   = st.selectbox("Loan Purpose",['Personal','Home Renovation','Medical','Education','Business','Wedding'])
        loan_type      = st.selectbox("Loan Type",['Secured','Unsecured'])

    with col3:
        st.subheader("📊 Bureau Details")
        cibil_input  = st.slider("CIBIL Score (0 = NTC)", 0, 900, 720, step=10)
        is_ntc       = (cibil_input == 0)
        cibil_score  = -1 if is_ntc else cibil_input
        if is_ntc: st.info("NTC: No credit history — will be flagged")

        existing_loans   = st.slider("Existing Loans", 0, 10, 1)
        credit_util      = st.slider("Credit Utilization (%)", 0, 100, 30) / 100
        dpd_30           = st.slider("DPD 30 Count", 0, 10, 0)
        dpd_60           = st.slider("DPD 60 Count", 0, 8, 0)
        dpd_90           = st.slider("DPD 90 Count", 0, 5, 0)
        bureau_inq       = st.slider("Bureau Inquiries (6m)", 0, 12, 1)
        oldest_acct      = st.slider("Oldest Account Age (months)", 3, 300, 48)

    st.markdown("---")

    if st.button("🔍 Calculate Credit Risk", type="primary", use_container_width=True):

        # Derived features
        emi_est    = loan_amount / loan_tenure * 1.10
        foir       = min(emi_est / monthly_income, 0.95)
        inc_loan   = (monthly_income * 12) / loan_amount
        loan_inc   = loan_amount / (monthly_income * 12)
        delinq_sc  = dpd_30*1 + dpd_60*2 + dpd_90*4
        bureau_str = int(bureau_inq>3) + int(credit_util>0.75) + int(existing_loans>3) + int(is_ntc)
        emp_stab   = {'Salaried':3,'Business':2,'Self-Employed':1,'Freelancer':0}[employment]

        cibil_for_band = np.nan if is_ntc else cibil_score
        if is_ntc:              cibil_band = '0_ntc'
        elif cibil_score < 600: cibil_band = '1_poor'
        elif cibil_score < 650: cibil_band = '2_below_avg'
        elif cibil_score < 700: cibil_band = '3_average'
        elif cibil_score < 750: cibil_band = '4_good'
        else:                   cibil_band = '5_excellent'

        age_band = ('21-30' if age<31 else '31-40' if age<41 else '41-50' if age<51 else '51-65')
        loan_size= ('Small' if loan_amount<100000 else 'Medium' if loan_amount<300000
                    else 'Large' if loan_amount<600000 else 'Very Large')

        woe_iv   = eda.get('woe_iv',{})
        woe_cibil= woe_iv.get('cibil_band',{}).get('woe',{})
        woe_emp  = woe_iv.get('employment_type',{}).get('woe',{})
        woe_age  = woe_iv.get('age_band',{}).get('woe',{})
        woe_loan = woe_iv.get('loan_size_cat',{}).get('woe',{})

        # Build feature vector
        row = {f: 0 for f in FEATURES}
        mapping = {
            'cibil_score': cibil_score, 'ntc_flag': int(is_ntc),
            'existing_loans': existing_loans, 'credit_utilization': credit_util,
            'dpd_30_count': dpd_30, 'dpd_60_count': dpd_60, 'dpd_90_count': dpd_90,
            'bureau_inquiries_6m': bureau_inq, 'oldest_account_age_months': oldest_acct,
            'delinquency_score': delinq_sc, 'bureau_stress_index': bureau_str,
            'any_dpd_90_flag': int(dpd_90>0), 'any_delinquency_flag': int(delinq_sc>0),
            'monthly_income': monthly_income, 'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure, 'interest_rate': interest_rate,
            'foir': foir, 'income_to_loan_ratio': inc_loan, 'loan_to_income_ratio': loan_inc,
            'high_foir_flag': int(foir>0.60), 'low_income_flag': int(monthly_income<25000),
            'high_loan_flag': int(loan_amount>500000), 'bureau_stale_flag': 0,
            'age': age, 'years_employed': years_emp, 'employment_stability': emp_stab,
            'city_tier': city_tier,
            'cibil_band_woe':  woe_cibil.get(cibil_band, 0),
            'emp_type_woe':    woe_emp.get(employment, 0),
            'age_band_woe':    woe_age.get(age_band, 0),
            'loan_size_woe':   woe_loan.get(loan_size, 0),
            'dpd_30_count_missing': 0, 'dpd_60_count_missing': 0, 'dpd_90_count_missing': 0,
        }
        for k,v in mapping.items():
            if k in row: row[k] = v

        X_input = pd.DataFrame([row])[FEATURES].fillna(-1)

        pd_prob  = float(model.predict_proba(X_input)[0][1])
        credit_score = int(np.clip(600 - (20/np.log(2)) * np.log(max(pd_prob,0.001)/max(1-pd_prob,0.001)) + (20/np.log(2))*np.log(50), 300, 900))

        if credit_score >= 750: risk_label = "✅ VERY LOW RISK";   decision = "APPROVE"; rate_mult=1.0; loan_pct=1.00; color='#2d6a4f'
        elif credit_score >= 700: risk_label = "🟢 LOW RISK";       decision = "APPROVE"; rate_mult=1.1; loan_pct=0.90; color='#2d6a4f'
        elif credit_score >= 650: risk_label = "🟡 MODERATE RISK";  decision = "CONDITIONAL"; rate_mult=1.25; loan_pct=0.75; color='#b8860b'
        elif credit_score >= 600: risk_label = "🟠 ELEVATED RISK";  decision = "REFER"; rate_mult=1.4; loan_pct=0.60; color='#b8860b'
        else:                     risk_label = "🔴 HIGH RISK";       decision = "REJECT"; rate_mult=0; loan_pct=0; color='#c8401a'

        st.markdown("---")
        st.subheader("📋 Credit Decision")

        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Credit Decision</div>"
                        f"<div class='metric-value' style='color:{color}'>{decision}</div></div>",
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Credit Score</div>"
                        f"<div class='metric-value'>{credit_score}</div>"
                        f"<div class='metric-label'>{risk_label}</div></div>",
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Prob. of Default</div>"
                        f"<div class='metric-value' style='color:{color}'>{pd_prob:.1%}</div></div>",
                        unsafe_allow_html=True)
        with r4:
            approved_amt = int(loan_amount * loan_pct)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Max Loan Approved</div>"
                        f"<div class='metric-value'>₹{approved_amt:,.0f}</div>"
                        f"<div class='metric-label'>{loan_pct:.0%} of requested</div></div>",
                        unsafe_allow_html=True)

        # Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Credit Score (300–900)"},
            gauge={
                'axis': {'range':[300,900],'tickwidth':1},
                'bar':  {'color': color},
                'steps':[
                    {'range':[300,600],'color':'#fde8e0'},
                    {'range':[600,650],'color':'#fef3cd'},
                    {'range':[650,700],'color':'#fff3cd'},
                    {'range':[700,750],'color':'#d4edda'},
                    {'range':[750,900],'color':'#c3e6cb'},
                ],
                'threshold':{'line':{'color':'black','width':3},'thickness':0.75,'value':credit_score}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Key risk factors
        st.subheader("🔑 Key Risk Factors")
        factors = []
        if is_ntc:       factors.append(("⚠️ No credit history (NTC)", "Moderate risk"))
        if dpd_90 > 0:   factors.append(("🔴 Past 90-day delinquency detected", "High risk signal"))
        if foir > 0.6:   factors.append((f"🟠 High FOIR = {foir:.1%}", "Repayment burden elevated"))
        if bureau_inq>4: factors.append((f"🟠 {bureau_inq} bureau inquiries in 6m", "Multiple credit-seeking"))
        if credit_util>0.75: factors.append((f"🟠 Credit utilization {credit_util:.0%}", "Over-leveraged"))
        if not is_ntc and cibil_score >= 750: factors.append(("✅ Excellent CIBIL score", "Strong credit history"))
        if years_emp > 5: factors.append(("✅ Stable employment", "Reduces default risk"))
        if foir < 0.4:   factors.append(("✅ Low FOIR", "Good repayment capacity"))
        if not factors:   factors.append(("✅ No major red flags detected", "Standard review"))

        for f,d in factors:
            st.markdown(f"**{f}** — *{d}*")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#6c757d;font-size:11px'>Fintech Credit Risk Pipeline · "
    "500K Records · Python + Pandas + Scikit-learn + Streamlit</center>",
    unsafe_allow_html=True
)
