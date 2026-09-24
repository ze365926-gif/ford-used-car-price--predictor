"""Ford Used Car Price Predictor — Streamlit Web App"""
import sys, os
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #003087 0%, #0057e0 60%, #00a6e0 100%);
        border-radius: 16px; padding: 36px 40px; margin-bottom: 28px; color: white;
    }
    .hero h1  { font-size: 2.2rem; font-weight: 700; margin: 0 0 6px 0; }
    .hero p   { font-size: 1.05rem; margin: 0; opacity: 0.88; }

    .metric-box {
        background: white; border-radius: 12px; padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #0057e0; margin-bottom: 12px;
    }
    .metric-box .label { font-size: 0.78rem; color: #666; font-weight: 600;
                         text-transform: uppercase; letter-spacing: .06em; }
    .metric-box .value { font-size: 1.7rem; font-weight: 700; color: #003087; margin-top: 2px; }

    .result-card {
        background: linear-gradient(135deg, #003087, #0057e0);
        border-radius: 16px; padding: 32px 36px; text-align: center;
        color: white; margin-top: 20px;
        box-shadow: 0 8px 30px rgba(0,86,224,0.35);
    }
    .result-card .rlabel { font-size: 1rem; opacity: 0.85; margin-bottom: 6px; }
    .result-card .rprice { font-size: 3.4rem; font-weight: 700; letter-spacing: -1px; }
    .result-card .rnote  { font-size: 0.8rem; opacity: 0.7; margin-top: 10px; }

    .stButton>button {
        background: linear-gradient(135deg,#0057e0,#003087) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        padding: 14px 0 !important; font-size: 1.05rem !important;
        width: 100%; transition: opacity .2s;
    }
    .stButton>button:hover { opacity: 0.88 !important; }

    div[data-testid="stSidebar"] { background: #f8f9fc; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #003087;
        text-transform: uppercase; letter-spacing: .06em;
        border-bottom: 2px solid #e0e8ff; padding-bottom: 6px; margin: 20px 0 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & metrics ──────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/car_price_pipeline.joblib")

@st.cache_data
def load_metrics():
    try:
        with open("artifacts/metrics.json") as f:
            return json.load(f)
    except:
        return {"r2": "—", "mae": "—", "rmse": "—"}

@st.cache_data
def load_data():
    from data_prep import load_and_clean_data
    return load_and_clean_data()

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception:
    model_loaded = False

metrics = load_metrics()

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = sorted([
    'B-MAX','C-MAX','EcoSport','Edge','Escort','Fiesta','Focus','Fusion',
    'Galaxy','Grand C-MAX','Grand Tourneo Connect','KA','Ka+','Kuga',
    'Mondeo','Mustang','Puma','Ranger','S-MAX','Streetka',
    'Tourneo Connect','Tourneo Custom','Transit Tourneo'
])
TRANSMISSIONS = ['Manual','Automatic','Semi-Auto']
FUEL_TYPES    = ['Petrol','Diesel','Hybrid','Electric','Other']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Ford_logo_flat.svg/200px-Ford_logo_flat.svg.png", width=80)
    st.markdown("## Ford Price Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Predictor", "📊 Data Insights", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">R² Score</div>
        <div class="value">{metrics.get('r2', '—')}</div>
    </div>
    <div class="metric-box">
        <div class="label">Mean Absolute Error</div>
        <div class="value">£{metrics.get('mae', '—'):,}</div>
    </div>
    <div class="metric-box">
        <div class="label">RMSE</div>
        <div class="value">£{metrics.get('rmse', '—'):,}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Predictor":
    st.markdown("""
    <div class="hero">
        <h1>🚗 Ford Used Car Price Predictor</h1>
        <p>Estimate the UK market value of any used Ford vehicle using a trained Random Forest model.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model artifact not found. Run `python src/train.py` to generate it first.")
        st.stop()

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-title">Vehicle Specification</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            model    = st.selectbox("Model", MODELS, index=MODELS.index("Fiesta"))
            year     = st.slider("Registration Year", 1996, 2024, 2018)
            trans    = st.selectbox("Transmission", TRANSMISSIONS)
            fuel     = st.selectbox("Fuel Type", FUEL_TYPES)
        with c2:
            mileage  = st.number_input("Mileage (miles)", 1, 250000, 20000, step=1000)
            engine   = st.number_input("Engine Size (L)", 0.1, 5.0, 1.0, step=0.1)
            mpg      = st.number_input("Fuel Economy (MPG)", 15.0, 202.0, 57.7, step=0.5)
            tax      = st.number_input("Road Tax (£/yr)", 0, 600, 145, step=5)

        predict_btn = st.button("🔍 Predict Market Value", use_container_width=True)

    with col_right:
        st.markdown('<p class="section-title">Valuation Result</p>', unsafe_allow_html=True)

        if predict_btn:
            payload = pd.DataFrame([{
                "model": model, "year": year, "transmission": trans,
                "mileage": mileage, "fuelType": fuel,
                "tax": tax, "mpg": mpg, "engineSize": engine,
            }])
            with st.spinner("Calculating..."):
                price = pipeline.predict(payload)[0]

            st.markdown(f"""
            <div class="result-card">
                <div class="rlabel">Estimated Market Value</div>
                <div class="rprice">£{price:,.0f}</div>
                <div class="rnote">Based on 17,760 UK Ford listings · Random Forest · R² 0.9255</div>
            </div>
            """, unsafe_allow_html=True)

            # Price range confidence ±MAE
            mae = metrics.get("mae", 853)
            st.info(f"📐 Typical accuracy range: **£{price-mae:,.0f}** – **£{price+mae:,.0f}**  (±£{mae:,.0f} MAE)")

            # Feature mini-summary
            st.markdown("**Your Vehicle:**")
            st.markdown(f"- `{year} Ford {model}` · {trans} · {fuel}")
            st.markdown(f"- {mileage:,} miles · {engine}L · {mpg} MPG · £{tax}/yr tax")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#aaa;">
                <div style="font-size:3rem;">🚗</div>
                <div style="font-size:1rem; margin-top:10px;">Fill in the specification<br>and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.markdown('<h2 style="color:#003087;">📊 Dataset Insights</h2>', unsafe_allow_html=True)
    st.caption("Exploratory analysis of the 17,760-row cleaned Ford UK dataset.")

    df = load_data()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Listings", f"{len(df):,}")
    k2.metric("Avg Price", f"£{df['price'].mean():,.0f}")
    k3.metric("Models", df['model'].nunique())
    k4.metric("Year Range", f"{df['year'].min()}–{df['year'].max()}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "By Category", "Mileage vs Price", "Correlation"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df['price'], bins=60, kde=True, ax=ax, color="#0057e0")
        ax.set_xlabel("Price (£)"); ax.set_ylabel("Count")
        ax.set_title("Used Ford Price Distribution")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"£{x:,.0f}"))
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        feat = st.selectbox("Group by", ["transmission", "fuelType", "model"])
        fig, ax = plt.subplots(figsize=(11, 5))
        order = df.groupby(feat)['price'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=feat, y='price', order=order, ax=ax,
                    palette="Blues_r", flierprops=dict(marker='.', alpha=0.3))
        ax.set_xlabel(feat.title()); ax.set_ylabel("Price (£)")
        ax.set_title(f"Price Distribution by {feat.title()}")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"£{y:,.0f}"))
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 5))
        sample = df.sample(min(3000, len(df)), random_state=42)
        sc = ax.scatter(sample['mileage'], sample['price'],
                        c=sample['year'], cmap='RdYlBu', alpha=0.5, s=18)
        plt.colorbar(sc, ax=ax, label="Year")
        ax.set_xlabel("Mileage (miles)"); ax.set_ylabel("Price (£)")
        ax.set_title("Mileage vs Price (coloured by Year)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"£{y:,.0f}"))
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df[['year','price','mileage','tax','mpg','engineSize']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
                    ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<h2 style="color:#003087;">ℹ️ About This Project</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### Ford Used Car Price Prediction Engine
    An end-to-end ML regression project predicting UK used-Ford vehicle valuations.

    **Data fixes applied to your notebook's bugs:**
    | Bug | Fix Applied |
    |-----|-------------|
    | `year == 2060` outlier | Filtered: `year` capped at 2024 |
    | `engineSize == 0.0` entries | Filtered: only `engineSize > 0` kept |
    | `.astype(int)` truncating floats | Removed — `StandardScaler` handles numerics |
    | `pd.get_dummies` leakage | Replaced with `OneHotEncoder` inside `Pipeline` |
    | Leading spaces in model names (` Fiesta`) | `.str.strip()` applied pre-encoding |
    | 154 duplicate rows | `drop_duplicates()` applied |

    **Pipeline architecture:**
    ```
    ford.csv → clean_data() → train_test_split
                                    │
                              ColumnTransformer
                              ├─ StandardScaler  (year, mileage, tax, mpg, engineSize)
                              └─ OneHotEncoder   (model, transmission, fuelType)
                                    │
                           RandomForestRegressor
                           (n_estimators=150, max_depth=20)
    ```

    **Model metrics on holdout test set (3,552 listings):**
    """)

    m = metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score", m.get("r2","—"))
    mc2.metric("MAE", f"£{m.get('mae','—'):,}")
    mc3.metric("RMSE", f"£{m.get('rmse','—'):,}")

    st.markdown("""
    **Tech Stack:** Python · scikit-learn · Pandas · Streamlit · Matplotlib · Seaborn · Joblib

    **How to run locally:**
    ```bash
    git clone https://github.com/<your-username>/ford-used-car-price-predictor
    cd ford-used-car-price-predictor
    pip install -r requirements.txt
    python src/train.py        # trains & saves model
    streamlit run app.py       # launches web app
    ```
    """)
