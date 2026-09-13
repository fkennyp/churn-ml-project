import requests
import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================================
# Config
# =====================================================
API_URL = "http://127.0.0.1:8000/predict"
REF_DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "churn_clean.csv"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# =====================================================
# Load dropdown options directly from processed data
# (to ensure options are always valid based on the dataset)
# =====================================================
@st.cache_data
def load_options():
    df = pd.read_csv(REF_DATA)
    return (
        sorted(df["Contract"].unique()),
        sorted(df["InternetService"].unique()),
        sorted(df["PaymentMethod"].unique()),
    )

CONTRACT_OPTS, INTERNET_OPTS, PAYMENT_OPTS = load_options()

# =====================================================
# Form UI
# =====================================================
st.title("📉 Churn Prediction App")
st.caption("This frontend calls FastAPI — the model is not loaded here.")

with st.form("customer_form"):
    st.subheader("Customer Data")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=75.0, step=0.5)
    with col2:
        contract = st.selectbox("Contract", CONTRACT_OPTS)
        internet = st.selectbox("Internet Service", INTERNET_OPTS)
    payment = st.selectbox("Payment Method", PAYMENT_OPTS)

    submitted = st.form_submit_button("🔮 Predict Churn")

# =====================================================
# Call API when button is pressed
# =====================================================
if submitted:
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
    }
    try:
        with st.spinner("Asking the model..."):
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
        result = resp.json()

        if result["prediction"] == 1:
            st.error(result["message"])
        else:
            st.success(result["message"])

        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        st.progress(result["churn_probability"])

        with st.expander("View raw JSON response from API"):
            st.json(result)

    except requests.ConnectionError:
        st.error("🚨 API unreachable! Make sure the uvicorn server (port 8000) is still running.")
    except Exception as e:
        st.error(f"Error: {e}")