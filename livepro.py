"""
liveapp2.py — NetSentinel AI
Cyber Threat Detection and Prediction System
Welcome screen → Live dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import random
import altair as alt
from detect_logic import get_predictions
try:
    from Predictor import RiskPredictor
except ImportError:
    from Predictor import RiskPredictor

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NetSentinel AI",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# THEME — deep slate, steel blue accents
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}
.stApp { background-color: #0d1117; }

/* ── Welcome screen ── */
.welcome-hero {
    text-align: center;
    padding: 60px 40px 40px;
}
.welcome-title {
    font-size: 52px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 12px;
}
.welcome-subtitle {
    font-size: 18px;
    font-weight: 400;
    color: #8b949e;
    margin-bottom: 40px;
    letter-spacing: 0.3px;
}
.accent { color: #58a6ff; }
.accent2 { color: #3fb950; }
.accent3 { color: #d29922; }

.model-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: #58a6ff; }
.model-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 6px;
}
.model-card-desc {
    font-size: 13px;
    color: #8b949e;
    line-height: 1.6;
}
.model-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
}
.badge-blue  { background: #1f3a5f; color: #58a6ff; border: 1px solid #58a6ff44; }
.badge-green { background: #1a3a1f; color: #3fb950; border: 1px solid #3fb95044; }
.badge-amber { background: #3a2a1a; color: #d29922; border: 1px solid #d2992244; }
.badge-purple { background: #2a1a3a; color: #bc8cff; border: 1px solid #bc8cff44; }

.stat-row {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 32px 0;
}
.stat-pill {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 24px;
    text-align: center;
}
.stat-num {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
}
.stat-lbl {
    font-size: 12px;
    color: #8b949e;
    margin-top: 2px;
}
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 32px 0;
}

/* ── Dashboard ── */
.dash-header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 16px 0;
    margin-bottom: 24px;
}
.stMetric {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
div[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
}
div[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
h1, h2, h3 {
    color: #e6edf3 !important;
    font-weight: 600 !important;
}
.stSidebar { background: #161b22 !important; border-right: 1px solid #30363d !important; }

/* ── Status boxes ── */
.status-nominal {
    background: #0d2818;
    border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 14px 20px;
    color: #3fb950;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}
.status-warning {
    background: #2d2008;
    border: 1px solid #d29922;
    border-left: 4px solid #d29922;
    border-radius: 8px;
    padding: 14px 20px;
    color: #d29922;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}
.status-critical {
    background: #2d0808;
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 14px 20px;
    color: #f85149;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}
.status-high {
    background: #2d1a08;
    border: 1px solid #e3903a;
    border-left: 4px solid #e3903a;
    border-radius: 8px;
    padding: 14px 20px;
    color: #e3903a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}

/* ── Decision boxes ── */
.dec-box {
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 4px;
}
.dec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}
.dec-value {
    font-size: 16px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}
.dec-safe     { background: #0d2818; border: 1px solid #3fb950; color: #3fb950; }
.dec-medium   { background: #2d2008; border: 1px solid #d29922; color: #d29922; }
.dec-high     { background: #2d1a08; border: 1px solid #e3903a; color: #e3903a; }
.dec-critical { background: #2d0808; border: 1px solid #f85149; color: #f85149; }

/* ── Info panel ── */
.info-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 13px;
    color: #8b949e;
    line-height: 1.8;
}
.info-panel b { color: #c9d1d9; }

/* ── Table ── */
.stTable { background: #161b22 !important; }
thead tr th { background: #21262d !important; color: #8b949e !important; font-size: 11px !important; }
tbody tr td { color: #c9d1d9 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Button ── */
.stButton > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 40px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #388bfd !important; }

/* Hide streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PKL_PATH     = '/Users/a91959/Downloads/deeplearning project/files/ddos_ensemble_phase1_pro.pkl'
TRAFFIC_PATH = '/Users/a91959/Downloads/deeplearning project/usethis.csv'
WINDOW_SIZE  = 10

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open(PKL_PATH, 'rb') as f:
        package = pickle.load(f)
    traffic_pool = pd.read_csv(TRAFFIC_PATH)
    return package, traffic_pool

package, traffic_pool = load_assets()
benign_pool = traffic_pool.iloc[:400]
attack_pool = traffic_pool.iloc[400:]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_severity(risk):
    if risk < 30: return "LOW"
    if risk < 60: return "MEDIUM"
    if risk < 85: return "HIGH"
    return "CRITICAL"

def classify_future(pred):
    s, l, p = pred['short_avg'], pred['long_avg'], pred['short_peak']
    if p > 85 or l > 70: return "ATTACK IMMINENT", "CRITICAL"
    if s > 60:            return "HIGH RISK",       "HIGH"
    if s > 30:            return "SUSPICIOUS",      "MEDIUM"
    return "SAFE",                                   "LOW"

def dec_class(level):
    return {'CRITICAL':'dec-critical','HIGH':'dec-high','MEDIUM':'dec-medium','LOW':'dec-safe'}[level]

def action_text(level):
    return {'CRITICAL':'🚫 Block Traffic Now','HIGH':'⚡ Prepare Mitigation',
            'MEDIUM':'👁 Monitor Closely','LOW':'✅ No Action Needed'}[level]

def status_class(level):
    return {'CRITICAL':'status-critical','HIGH':'status-high',
            'MEDIUM':'status-warning','LOW':'status-nominal'}[level]

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# ═════════════════════════════════════════════
# WELCOME PAGE
# ═════════════════════════════════════════════
if st.session_state.page == 'welcome':

    st.markdown("""
<div class="welcome-hero">
    <div style="font-size:13px;font-weight:600;letter-spacing:3px;color:#58a6ff;
                text-transform:uppercase;margin-bottom:16px;opacity:0.8;">
        DEEP LEARNING · CYBERSECURITY · REAL-TIME INTELLIGENCE
    </div>
    <div class="welcome-title">
        NetSentinel <span class="accent">AI</span>
    </div>
    <div class="welcome-subtitle">
        Cyber Threat Detection and Prediction System<br>
        Powered by LSTM Temporal Modelling · Autoencoder Anomaly Detection · Ensemble Learning
    </div>
</div>
""", unsafe_allow_html=True)

    # Stats row
    st.markdown("""
<div class="stat-row">
    <div class="stat-pill">
        <div class="stat-num">99.9%</div>
        <div class="stat-lbl">Detection Accuracy</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">225K</div>
        <div class="stat-lbl">Training Samples</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">3</div>
        <div class="stat-lbl">AI Engines</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">< 50ms</div>
        <div class="stat-lbl">Detection Latency</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">1.00</div>
        <div class="stat-lbl">F1 Score</div>
    </div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

    # Model cards
    st.markdown("<h3 style='text-align:center;margin-bottom:24px;'>AI Engine Architecture</h3>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
<div class="model-card">
    <div class="model-badge badge-blue">ENGINE 01</div>
    <div class="model-card-title">Bidirectional LSTM</div>
    <div class="model-card-desc">
        2-layer bidirectional LSTM trained on 10-packet sequences.
        Detects temporal attack patterns — SYN floods, sustained bursts,
        and ramp-up signatures invisible to per-packet classifiers.
        Hidden size: 128 × 2 directions. Val accuracy: 99.92%.
    </div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="model-card">
    <div class="model-badge badge-green">ENGINE 02</div>
    <div class="model-card-title">Autoencoder Anomaly</div>
    <div class="model-card-desc">
        Deep autoencoder trained on benign traffic only.
        Flags packets with high reconstruction error as anomalous.
        Detects zero-day attack patterns never seen during training.
        Bottleneck: 16 dimensions. Threshold: 95th percentile.
    </div>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="model-card">
    <div class="model-badge badge-amber">ENGINE 03</div>
    <div class="model-card-title">Ensemble Fusion</div>
    <div class="model-card-desc">
        Weighted combination: LSTM (90%) + Autoencoder (10%).
        LSTM dominates classification decisions. AE provides
        zero-day anomaly signal. Risk score clipped to [0,1]
        with severity tiers: LOW / MEDIUM / HIGH / CRITICAL.
    </div>
</div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
<div class="model-card">
    <div class="model-badge badge-purple">ENGINE 04</div>
    <div class="model-card-title">Risk Predictor</div>
    <div class="model-card-desc">
        Three-signal consensus forecaster: EMA slope, weighted
        least squares regression, and pattern recognition.
        Short horizon: next 5 packets. Long horizon: next 30.
        Attack-aware — stays elevated during active bursts.
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Dataset and training info
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class="info-panel">
<b>Dataset</b><br>
CICIDS-2017 Friday DDoS — Canadian Institute for Cybersecurity<br>
225,711 network flow records · 78 features · 2 classes<br>
DDoS: 128,025 samples · BENIGN: 97,686 samples<br><br>
<b>Training Split</b><br>
70% training · 30% test · Stratified by label<br>
Flow-based grouping to prevent data leakage<br><br>
<b>Preprocessing</b><br>
StandardScaler normalisation · Infinite value removal<br>
Label encoding · 10-packet sliding window construction
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="info-panel">
<b>LSTM Training</b><br>
Optimizer: Adam (lr=1e-3) · Loss: CrossEntropy<br>
Scheduler: ReduceLROnPlateau · Gradient clip: 1.0<br>
Epochs: 20 · Batch size: 512 · Best val loss: 0.0040<br><br>
<b>Autoencoder Training</b><br>
Benign-only training · Loss: MSE<br>
Epochs: 25 · Bottleneck: 16 dims · Threshold: p95<br><br>
<b>Prediction Engine</b><br>
3-signal consensus: EMA + WLS + Pattern recognition<br>
Attack-aware long horizon · Early warning at 25–50% zone
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Start button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("▶  Launch Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("""
<div style="text-align:center;margin-top:24px;font-size:12px;color:#484f58;">
    CICIDS-2017 · PyTorch 2.x · Streamlit · Built for research and demonstration purposes
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# DASHBOARD PAGE
# ═════════════════════════════════════════════
elif st.session_state.page == 'dashboard':

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ NetSentinel AI")
        st.markdown("---")
        sim_speed = st.select_slider(
            "Simulation Speed",
            options=[0.15, 0.1, 0.05, 0.02],
            value=0.1,
            format_func=lambda x: {0.15:"Slow",0.1:"Normal",0.05:"Fast",0.02:"Max"}[x]
        )
        attack_mode = st.selectbox(
            "Traffic Scenario",
            ["Normal Traffic", "Mixed (30% Attack)", "Mixed (60% Attack)", "Heavy Attack (80%)"],
            index=0
        )
        attack_ratio = {"Normal Traffic": 0.0, "Mixed (30% Attack)": 0.30,
                        "Mixed (60% Attack)": 0.60, "Heavy Attack (80%)": 0.80}[attack_mode]

        st.markdown("---")
        st.markdown("**Engine Status**")
        st.success("✅ LSTM Temporal")
        st.success("✅ Autoencoder")
        st.success("✅ Risk Predictor")
        st.success("✅ Decision Engine")
        st.info("Ensemble: LSTM 90% + AE 10%")

        st.markdown("---")
        st.markdown("**Severity Scale**")
        st.markdown("🟢 LOW  · 0–30%")
        st.markdown("🟡 MEDIUM  · 30–60%")
        st.markdown("🟠 HIGH  · 60–85%")
        st.markdown("🔴 CRITICAL  · 85–100%")

        st.markdown("---")
        if st.button("← Back to Overview"):
            st.session_state.page = 'welcome'
            st.rerun()

    # Header
    st.markdown("## 🛡️ NetSentinel AI — Live Monitoring")
    st.caption(f"Scenario: **{attack_mode}** · LSTM + Autoencoder Ensemble · Risk Forecasting Active")

    # Row 1: core metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    packet_ph   = m1.empty()
    threat_ph   = m2.empty()
    health_ph   = m3.empty()
    risk_ph     = m4.empty()
    severity_ph = m5.empty()

    st.markdown("---")
    alert_ph = st.empty()
    st.markdown("---")

    # Row 2: prediction metrics
    st.markdown("#### 🔮 Risk Prediction Engine")
    p1, p2, p3, p4 = st.columns(4)
    short_ph = p1.empty()
    long_ph  = p2.empty()
    peak_ph  = p3.empty()
    burst_ph = p4.empty()

    st.markdown("---")

    # Row 3: Decision Engine
    st.markdown("#### 🧠 AI Decision Engine")
    d1, d2, d3, d4 = st.columns(4)
    dec_short_ph  = d1.empty()
    dec_long_ph   = d2.empty()
    dec_peak_ph   = d3.empty()
    dec_action_ph = d4.empty()

    st.markdown("---")

    # Row 4: Accuracy tracker
    st.markdown("#### 📊 Prediction Accuracy Tracker")
    a1, a2, a3, a4, a5 = st.columns(5)
    acc_total_ph  = a1.empty()
    acc_right_ph  = a2.empty()
    acc_pct_ph    = a3.empty()
    acc_fp_ph     = a4.empty()
    acc_fn_ph     = a5.empty()

    st.markdown("---")

    # Row 5: Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### 📈 Real-Time Risk + Short Forecast")
        short_chart_ph = st.empty()
    with ch2:
        st.markdown("#### 📉 Long Horizon Forecast (30 packets)")
        long_chart_ph = st.empty()

    st.markdown("---")

    # Row 6: Distribution + Severity
    dc, sc_col = st.columns(2)
    with dc:
        st.markdown("#### 📊 Threat Distribution")
        dist_ph = st.empty()
    with sc_col:
        st.markdown("#### 🎯 Severity Breakdown")
        sv = st.columns(4)
        sev_ph = {
            "LOW":      sv[0].empty(),
            "MEDIUM":   sv[1].empty(),
            "HIGH":     sv[2].empty(),
            "CRITICAL": sv[3].empty(),
        }

    st.markdown("---")

    # Row 7: explainer
    with st.expander("🔍 How the Prediction Engine Works"):
        st.markdown("""
<div class="info-panel">
<b>Short Horizon (next 5 packets) — Three-Signal Consensus</b><br>
Signal 1: EMA slope — exponential moving average of last 8 risk scores, weighted toward recency.<br>
Signal 2: WLS slope — weighted least squares regression on last 15 scores, exponential weights.<br>
Signal 3: Pattern recognition — identifies sustained attack (high mean, low variance) or benign patterns.<br>
Consensus: if signals 1 and 2 agree on direction, prediction blends 70% trend + 30% pattern.
If they disagree, it blends 40% trend + 60% pattern (more conservative, fewer false positives).<br><br>

<b>Long Horizon (next 30 packets) — Attack-Aware Smoothing</b><br>
If attack is active: prediction decays very slowly (factor 0.97) toward a floor of 65%.<br>
If benign: prediction reverts quickly (factor 0.78) toward the 10-packet historical mean.<br>
This is why long horizon shows ATTACK LIKELY during active bursts instead of always STABLE.<br><br>

<b>Accuracy Tracking</b><br>
Each prediction is queued with a target packet number. When that packet arrives, the actual
risk is compared. Prediction labelled ATTACK if avg ≥ 50%, SAFE otherwise.
Correct = labels match. FP = predicted ATTACK but was SAFE. FN = predicted SAFE but was ATTACK.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📟 Live Security Log")
    log_ph = st.empty()

    # ── Simulation loop ──────────────────────────────────────
    if st.sidebar.button("▶ Activate Sentinel", use_container_width=True):
        risk_history    = []
        log_data        = []
        total_p         = 0
        total_t         = 0
        threat_counts   = {"BENIGN": 0, "DDoS": 0}
        sev_counts      = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        packet_buffer   = []
        predictor       = RiskPredictor()

        for i in range(1000):
            total_p += 1

            # Sample based on scenario
            if attack_ratio > 0 and random.random() < attack_ratio:
                packet_data = attack_pool.sample(1)
            else:
                packet_data = benign_pool.sample(1)

            # Rolling window
            packet_buffer.append(packet_data)
            if len(packet_buffer) > WINDOW_SIZE:
                packet_buffer.pop(0)

            window_df     = pd.concat(packet_buffer, ignore_index=True)
            labels, risks = get_predictions(window_df, package)
            risk_val      = float(risks[-1]) * 100
            label         = labels[-1]
            severity      = get_severity(risk_val)

            predictor.update(risk_val, total_p)
            pred    = predictor.summary()
            warning = pred['early_warning']
            burst   = pred['burst_status']
            acc     = pred['accuracy']
            status_text, level = classify_future(pred)

            risk_history.append(risk_val)
            threat_counts[label if label == "BENIGN" else "DDoS"] += 1
            sev_counts[severity] += 1
            if label != "BENIGN":
                total_t += 1

            avg_risk = sum(risk_history[-10:]) / len(risk_history[-10:])

            # ── Core metrics ──────────────────────────────────
            packet_ph.metric("Packets Analyzed",  f"{total_p:,}")
            threat_ph.metric("Threats Detected",  f"{total_t:,}", delta=str(total_t), delta_color="inverse")
            health_ph.metric("System Health",     f"{100 - avg_risk:.1f}%")
            risk_ph.metric("Current Risk",        f"{risk_val:.1f}%")
            severity_ph.metric("Severity",        severity)

            # ── Alert banner ──────────────────────────────────
            if level == "CRITICAL" and (burst['active'] or pred['short_peak'] > 85):
                burst_info = f"{burst['packets_in']} packets in burst" if burst['active'] \
                             else f"{pred['short_peak']}% peak predicted"
                alert_ph.markdown(
                    f'<div class="status-critical">🚨 AI DECISION: BLOCK INCOMING TRAFFIC NOW — {burst_info}</div>',
                    unsafe_allow_html=True)
            elif burst['active']:
                alert_ph.markdown(
                    f'<div class="status-critical">🔴 BURST ATTACK IN PROGRESS — '
                    f'{burst["packets_in"]} packets — '
                    f'~{burst["remaining_est"]:,} estimated remaining ({burst["pct_complete"]}% complete)</div>',
                    unsafe_allow_html=True)
            elif warning['active']:
                alert_ph.markdown(
                    f'<div class="status-warning">⚠ PRE-ATTACK WARNING — '
                    f'Risk rising +{warning["slope"]}%/packet — '
                    f'Attack predicted in ~{warning["eta_packets"]} packets — '
                    f'Confidence: {warning["confidence"]}%</div>',
                    unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="status-nominal">✓ Network Nominal — No threats predicted in next 30 packets</div>',
                    unsafe_allow_html=True)

            # ── Prediction metrics ────────────────────────────
            short_ph.metric("Avg Risk · Next 5",   f"{pred['short_avg']}%", delta=f"{pred['short_avg']-risk_val:+.1f}%")
            long_ph.metric("Avg Risk · Next 30",   f"{pred['long_avg']}%",  delta=f"{pred['long_avg']-risk_val:+.1f}%")
            peak_ph.metric("Peak · Next 5 pkts",   f"{pred['short_peak']}%")
            burst_ph.metric("Burst Size",          f"{burst.get('packets_in',0):,}" if burst['active'] else "—")

            # ── Decision Engine ───────────────────────────────
            dc_cls = dec_class(level)
            dec_short_ph.markdown(f'<div class="dec-box {dc_cls}"><div class="dec-label">Next 5 Packets</div><div class="dec-value">{status_text}</div></div>', unsafe_allow_html=True)

            long_lbl = "Attack Likely" if pred['long_avg'] > 60 else "Stable"
            long_dc  = dec_class("CRITICAL" if pred['long_avg'] > 60 else "LOW")
            dec_long_ph.markdown(f'<div class="dec-box {long_dc}"><div class="dec-label">Next 30 Packets</div><div class="dec-value">{long_lbl}</div></div>', unsafe_allow_html=True)

            peak_lv  = "CRITICAL" if pred['short_peak'] > 85 else "HIGH" if pred['short_peak'] > 60 else "MEDIUM" if pred['short_peak'] > 30 else "LOW"
            peak_dc  = dec_class(peak_lv)
            dec_peak_ph.markdown(f'<div class="dec-box {peak_dc}"><div class="dec-label">Peak Risk</div><div class="dec-value">{pred["short_peak"]}%</div></div>', unsafe_allow_html=True)

            dec_action_ph.markdown(f'<div class="dec-box {dc_cls}"><div class="dec-label">Recommended Action</div><div class="dec-value">{action_text(level)}</div></div>', unsafe_allow_html=True)

            # ── Accuracy tracker ──────────────────────────────
            acc_total_ph.metric("Predictions",   f"{acc['total']}")
            acc_right_ph.metric("Correct",       f"{acc['correct']}")
            acc_pct_ph.metric("Accuracy",        f"{acc['accuracy']}%")
            acc_fp_ph.metric("False Positives",  f"{acc['false_positives']}")
            acc_fn_ph.metric("False Negatives",  f"{acc['false_negatives']}")

            # ── Short forecast chart ──────────────────────────
            hist_s  = risk_history[-20:]
            fore_s  = pred['short_horizon'].tolist()
            short_df = pd.DataFrame({
                'x':    list(range(len(hist_s))) + list(range(len(hist_s), len(hist_s)+len(fore_s))),
                'risk': hist_s + fore_s,
                'type': ['Actual']*len(hist_s) + ['Forecast']*len(fore_s)
            })
            short_chart_ph.altair_chart(
                alt.Chart(short_df).mark_line(point=True).encode(
                    x=alt.X('x:Q', title='Packet'),
                    y=alt.Y('risk:Q', title='Risk %', scale=alt.Scale(domain=[0,100])),
                    color=alt.Color('type:N', scale=alt.Scale(
                        domain=['Actual','Forecast'], range=['#58a6ff','#3fb950'])),
                    strokeDash=alt.condition(alt.datum.type=='Forecast', alt.value([5,5]), alt.value([0]))
                ).properties(height=200),
                use_container_width=True
            )

            # ── Long forecast chart ───────────────────────────
            long_fore = pred['long_horizon'].tolist()
            long_df   = pd.DataFrame({'x': list(range(len(long_fore))), 'risk': long_fore})
            band_df   = pd.DataFrame({'threshold': [50]})
            long_chart_ph.altair_chart(
                alt.Chart(long_df).mark_line(point=False, strokeDash=[4,4]).encode(
                    x=alt.X('x:Q', title='Packets Ahead'),
                    y=alt.Y('risk:Q', title='Predicted Risk %', scale=alt.Scale(domain=[0,100])),
                    color=alt.value('#bc8cff')
                ).properties(height=200) +
                alt.Chart(band_df).mark_rule(color='#f85149', strokeDash=[4,4], opacity=0.5).encode(y='threshold:Q'),
                use_container_width=True
            )

            # ── Threat distribution ───────────────────────────
            dist_ph.altair_chart(
                alt.Chart(pd.DataFrame({
                    "Classification": list(threat_counts.keys()),
                    "Count":          list(threat_counts.values())
                })).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x='Classification', y='Count',
                    color=alt.Color('Classification', scale=alt.Scale(
                        domain=['BENIGN','DDoS'], range=['#3fb950','#f85149']),
                        legend=None)
                ).properties(height=200),
                use_container_width=True
            )

            # ── Severity breakdown ────────────────────────────
            sev_ph["LOW"].metric("🟢 Low",          sev_counts["LOW"])
            sev_ph["MEDIUM"].metric("🟡 Medium",    sev_counts["MEDIUM"])
            sev_ph["HIGH"].metric("🟠 High",        sev_counts["HIGH"])
            sev_ph["CRITICAL"].metric("🔴 Critical", sev_counts["CRITICAL"])

            # ── Terminal log ──────────────────────────────────
            try:
                port = int(packet_data['Destination Port'].values[0])
            except Exception:
                port = 0

            if warning['active']:
                pred_tag = f"⚠ ETA {warning['eta_packets']}p"
            elif burst['active']:
                pred_tag = f"Burst {burst['packets_in']}pkt"
            else:
                pred_tag = f"→ {pred['short_avg']}%"

            log_data.insert(0, {
                "Time":      time.strftime("%H:%M:%S"),
                "Status":    "🔴" if label != "BENIGN" else "🟢",
                "Port":      port,
                "Risk":      f"{risk_val:.1f}%",
                "Severity":  severity,
                "Decision":  status_text,
                "Action":    "BLOCK" if label != "BENIGN" else "PASS",
            })
            log_ph.table(pd.DataFrame(log_data[:10]))

            time.sleep(sim_speed)