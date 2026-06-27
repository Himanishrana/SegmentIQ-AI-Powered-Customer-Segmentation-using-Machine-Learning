import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0a0a0f !important;
}

[data-testid="stAppViewContainer"] {
    font-family: 'DM Mono', monospace;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Hero header ── */
.hero {
    position: relative;
    padding: 3.5rem 2rem 2.5rem;
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -80px;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(99,255,180,0.13) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(120,100,255,0.11) 0%, transparent 70%);
    pointer-events: none;
}
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    color: #63ffb4;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 800;
    color: #f0f0f5;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin-bottom: 0.9rem;
}
.hero-title span {
    color: #63ffb4;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.38);
    max-width: 480px;
    line-height: 1.7;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Card container ── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(8px);
    transition: border-color 0.25s;
}
.card:hover { border-color: rgba(99,255,180,0.18); }

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f0f0f5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #63ffb4 !important;
    box-shadow: 0 0 0 3px rgba(99,255,180,0.08) !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.5) !important;
    border-radius: 6px !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    background: #63ffb4 !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,255,180,0.2) !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] button:hover {
    background: #8fffc8 !important;
    box-shadow: 0 6px 32px rgba(99,255,180,0.35) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button:active {
    transform: translateY(0px) !important;
}

/* ── Success / result box ── */
[data-testid="stAlert"] {
    background: rgba(99,255,180,0.06) !important;
    border: 1px solid rgba(99,255,180,0.22) !important;
    border-radius: 12px !important;
    color: #63ffb4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}

/* ── Metric tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-tile {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.metric-tile .m-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-tile .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0f0f5;
}
.metric-tile .m-value.accent { color: #63ffb4; }

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(99,255,180,0.07) 0%, rgba(120,100,255,0.06) 100%);
    border: 1px solid rgba(99,255,180,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -30px; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,255,180,0.1) 0%, transparent 70%);
}
.result-cluster-num {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    color: #63ffb4;
    line-height: 1;
    letter-spacing: -0.04em;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.result-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.55);
    margin-top: 1rem;
    line-height: 1.65;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 2rem 0;
}

/* ── Input icon row ── */
.input-icon {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #63ffb4;
    margin-right: 6px;
    vertical-align: middle;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scalar = joblib.load("scalar.pkl")
    return kmeans, scalar

kmeans, scalar = load_models()

# ── Cluster descriptions (customize to your segments) ────────────────────────
CLUSTER_INFO = {
    0: ("High-Value Loyalists",    "Frequent buyers with high income & low recency. Prioritize retention perks."),
    1: ("Occasional Browsers",     "Regular web visitors but low conversion. Engage with targeted offers."),
    2: ("Bargain Seekers",         "High store visits, lower spending. Price-sensitive; respond well to discounts."),
    3: ("Dormant High Spenders",   "Historically big spenders, now inactive. Win-back campaigns recommended."),
    4: ("Emerging Regulars",       "Younger segment with growing purchase frequency. Nurture with loyalty programs."),
}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">◈ ML-Powered Analytics</div>
    <div class="hero-title">Customer<br><span>Segmentation</span></div>
    <div class="hero-sub">Input customer attributes to predict which behavioral cluster they belong to using K-Means clustering.</div>
</div>
""", unsafe_allow_html=True)

# ── Layout: two columns ───────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-label">01 — Customer Profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        total_spending = st.number_input("Total Spending (USD)", min_value=0, max_value=5000, value=1000)
        num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10)
        recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=30)
    with c2:
        income = st.number_input("Annual Income (USD)", min_value=0, max_value=200000, value=50000)
        num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10)
        num_web_visits = st.number_input("Web Visits / Month", min_value=0, max_value=50, value=3)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Live summary metrics ──
    st.markdown('<div class="section-label">02 — Input Summary</div>', unsafe_allow_html=True)
    purchase_ratio = round(num_web_purchases / (num_web_purchases + num_store_purchases + 0.01) * 100, 1)
    spend_per_visit = round(total_spending / (num_web_visits + 1), 1)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="m-label">Total Purchases</div>
            <div class="m-value accent">{num_web_purchases + num_store_purchases}</div>
        </div>
        <div class="metric-tile">
            <div class="m-label">Web Purchase %</div>
            <div class="m-value">{purchase_ratio}%</div>
        </div>
        <div class="metric-tile">
            <div class="m-label">Spend / Visit</div>
            <div class="m-value">${spend_per_visit}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("◈  Run Segmentation")

with right:
    st.markdown('<div class="section-label">03 — Segment Output</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame({
            "Age":               [age],
            "Income":            [income],
            "Total_Spending":    [total_spending],
            "NumWebPurchases":   [num_web_purchases],
            "NumStorePurchases": [num_store_purchases],
            "NumWebVisitsMonth": [num_web_visits],
            "Recency":           [recency],
        })
        input_scaled = scalar.transform(input_data)
        cluster = int(kmeans.predict(input_scaled)[0])

        seg_name, seg_desc = CLUSTER_INFO.get(cluster, ("Cluster " + str(cluster), "Segment identified successfully."))

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Cluster</div>
            <div class="result-cluster-num">{cluster}</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem;
                        color:#f0f0f5; margin-top:0.8rem; letter-spacing:0.01em;">
                {seg_name}
            </div>
            <div class="result-desc">{seg_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance bar (normalized input values)
        st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">04 — Feature Overview</div>', unsafe_allow_html=True)

        features = {
            "Age":          age / 100,
            "Income":       income / 200000,
            "Spending":     total_spending / 5000,
            "Web Purchases": num_web_purchases / 100,
            "Store Purchases": num_store_purchases / 100,
            "Web Visits":   num_web_visits / 50,
            "Recency":      recency / 365,
        }

        for fname, fval in features.items():
            pct = min(int(fval * 100), 100)
            color = "#63ffb4" if pct > 60 else ("#7864ff" if pct > 30 else "rgba(255,255,255,0.25)")
            st.markdown(f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                 letter-spacing:0.1em; color:rgba(255,255,255,0.45);
                                 text-transform:uppercase;">{fname}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                 color:rgba(255,255,255,0.3);">{pct}%</span>
                </div>
                <div style="height:4px; background:rgba(255,255,255,0.06); border-radius:99px; overflow:hidden;">
                    <div style="height:100%; width:{pct}%; background:{color};
                                border-radius:99px; transition:width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            color: rgba(255,255,255,0.2);
        ">
            <div style="font-size:2rem; margin-bottom:1rem;">◈</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                        letter-spacing:0.15em; text-transform:uppercase;">
                Fill in the profile<br>and run segmentation
            </div>
        </div>
        """, unsafe_allow_html=True)
