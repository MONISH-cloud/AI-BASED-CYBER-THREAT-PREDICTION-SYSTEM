import streamlit as st
import pandas as pd
import pickle
from detect_logic import get_predictions

pd.set_option("styler.render.max_elements", 1000000)

st.set_page_config(page_title="AI Predictive Firewall", layout="wide")

PKL_PATH = '/Users/a91959/Downloads/deeplearning project/files/ddos_ensemble_phase1.pkl'

@st.cache_resource
def load_model():
    with open(PKL_PATH, 'rb') as f:
        return pickle.load(f)

package = load_model()

st.title("🛡️ AI Network Threat Predictor")
st.markdown("Upload network traffic to analyze real-time risk levels.")

uploaded_file = st.file_uploader("Upload CSV Traffic Log", type="csv")

if uploaded_file:
    with st.spinner('Analyzing full dataset...'):
        data = pd.read_csv(uploaded_file)
        labels, risks = get_predictions(data, package)
        data['Prediction'] = labels
        data['Risk_Score'] = [f"{r*100:.1f}%" for r in risks]
        avg_risk = risks.mean() * 100
        total_packets = len(data)
        threats = (labels != 'BENIGN').sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Packets Analyzed", f"{total_packets:,}")
    col2.metric("Threats Detected", f"{threats:,}", delta=f"{threats}", delta_color="inverse")

    if avg_risk > 70:
        col3.error(f"SYSTEM RISK: {avg_risk:.1f}% (CRITICAL)")
    elif avg_risk > 30:
        col3.warning(f"SYSTEM RISK: {avg_risk:.1f}% (SUSPICIOUS)")
    else:
        col3.success(f"SYSTEM RISK: {avg_risk:.1f}% (SAFE)")

    st.markdown("---")
    st.subheader("Traffic Analysis Log (Top 5,000 Rows)")

    def color_threats(val):
        return 'color: red; font-weight: bold' if val != 'BENIGN' else 'color: green'

    display_df = data[['Prediction', 'Risk_Score']].head(5000)
    st.dataframe(
        display_df.style.map(color_threats, subset=['Prediction']),
        use_container_width=True
    )

    if total_packets > 5000:
        st.info(f"💡 Note: Showing first 5,000 rows. Full analysis of {total_packets:,} rows completed.")