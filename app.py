
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_tuned.joblib"

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# --------------------------------------------------
# Load the trained pipeline
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("📡 Telecom Customer Churn Prediction")
st.markdown(
    "### Customer Retention Intelligence Dashboard"
)
st.write(
    "Enter customer details below to estimate whether the customer "
    "is likely to leave the telecom service."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

# --------------------------------------------------
# Customer input form
# --------------------------------------------------
with st.form("customer_prediction_form"):
    st.subheader("1. Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    with col2:
        tenure_months = st.number_input(
            "Tenure (months)", min_value=0, max_value=120, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

    with col3:
        contract_type = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.divider()
    st.subheader("2. Services and Support")

    col4, col5, col6 = st.columns(3)

    with col4:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col5:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        support_calls = st.number_input(
            "Support Calls", min_value=0, max_value=100, value=1
        )

    with col6:
        late_payments = st.number_input(
            "Late Payments", min_value=0, max_value=100, value=0
        )
        customer_satisfaction = st.slider(
            "Customer Satisfaction", min_value=1, max_value=5, value=3
        )

    st.divider()
    st.subheader("3. Billing and Usage")

    col7, col8, col9 = st.columns(3)

    with col7:
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=10000.0,
            value=70.0, step=5.0
        )

    with col8:
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=1000000.0,
            value=840.0, step=10.0
        )

    with col9:
        avg_monthly_gb = st.number_input(
            "Average Monthly Data Usage (GB)",
            min_value=0.0, max_value=100000.0,
            value=50.0, step=5.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True,
        type="primary",
    )

# --------------------------------------------------
# Make prediction
# --------------------------------------------------
if submitted:
    if tenure_months == 0 and total_charges > 0:
        st.warning(
            "A customer with zero months of tenure normally should not "
            "have positive total charges. Please verify these inputs."
        )
        st.stop()

    calculated_monthly_charge = (
        total_charges / tenure_months if tenure_months > 0 else 0.0
    )

    customer_risk_signals = support_calls + late_payments

    customer = pd.DataFrame([{
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "contract_type": contract_type,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "avg_monthly_gb": avg_monthly_gb,
        "support_calls": support_calls,
        "late_payments": late_payments,
        "customer_satisfaction": customer_satisfaction,
        "customer_risk_signals": customer_risk_signals,
        "calculated_monthly_charge": calculated_monthly_charge,
    }])

    try:
        prediction = int(model.predict(customer)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(
                    model.predict_proba(customer)[0][classes.index(1)]
                )

        st.divider()
        st.subheader("Prediction Results")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.error("⚠️ Prediction: Customer Likely to Churn")
                st.write(
                    "Consider reviewing the customer's experience and "
                    "offering appropriate retention support."
                )
            else:
                st.success("✅ Prediction: Customer Likely to Stay")
                st.write(
                    "Continue monitoring satisfaction and service usage."
                )

        with probability_col:
            if probability is not None:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.1%}",
                )
                st.progress(probability)
            else:
                st.info("This model does not provide probability estimates.")

        st.caption(
            "This is a machine-learning estimate, not a guarantee of "
            "future customer behavior. Predictions depend on data quality "
            "and how representative the training dataset is."
        )

    except Exception as error:
        st.error(
            "Prediction failed. Please verify the inputs and saved model."
        )
        st.exception(error)

st.divider()
st.caption("Telecom Customer Churn Prediction | Machine Learning Project")
