import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}. Ensure 'churn_model.pkl' and 'scaler.pkl' are in the same folder.")


MAPPINGS = {
    "gender": {"Female": 0, "Male": 1},
    "region": {"City": 0, "Town": 1, "Village": 2},
    "membership": {
        "Basic Membership": 0, 
        "Gold Membership": 1, 
        "No Membership": 2, 
        "Platinum Membership": 3, 
        "Premium Membership": 4, 
        "Silver Membership": 5
    },
    "referral": {"No": 0, "Yes": 1},
    "offer_type": {"Credit/Debit Card Offers": 0, "Gift Vouchers/Coupons": 1, "Without Offer": 2},
    "operation_channel": {"Both": 0, "Desktop": 1, "Smartphone": 2},
    "internet": {"Fiber Optic": 0, "Mobile Data": 1, "Wi-Fi": 2},
    "used_special_discount": {"No": 0, "Yes": 1},
    "offer_application_preference": {"No": 0, "Yes": 1},
    "past_complaint": {"No": 0, "Yes": 1},
    "complaint": {"No": 0, "Yes": 1},
    "age_group": {"Young (18-30)": 0, "Adult (31-45)": 1, "Middle-Aged (46-60)": 2, "Senior (60+)": 3},
    "feedback": {
        "Poor Product Quality": 0,
        "Too many ads": 1,
        "As expected": 2,
        "User Friendly": 3,
        "No reason specified": 4,
        "Product quality is good": 5,
        "Reasonable Price": 6,
        "Quality Customer Care": 7,
        "Helpful Staff": 8
    }
}


st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom Title Header
st.title("📊 Telecom Customer Churn Prediction & Analytics System")
st.markdown("Developed by : Saif Eldeen Wesam Elsayed")

# Create Project Tabs to contain everything about your project
tab1, tab2, tab3 = st.tabs(["🏠 Project Overview", "🔮 Churn Risk Predictor", "💡 Strategic Business Insights"])


with tab1:
    st.header("📋 Business Understanding & Goals")
    st.write(
        "Customer churn is one of the biggest challenges faced by telecommunications companies. "
        "Losing customers leads to reduced revenue and increased acquisition costs. This application "
        "analyzes customer behavior and leverages Machine Learning to predict whether a customer is likely to churn."
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔍 Project Research Questions")
        st.markdown("""
        - **Q1:** What factors influence customer churn?
        - **Q2:** Does contract or membership type affect churn rates?
        - **Q3:** Do transaction values and financial traits impact customer retention?
        - **Q4:** Which features are the most important predictors of churn?
        - **Q5:** Can machine learning accurately flag at-risk customers?
        """)
        
    with col_b:
        st.subheader("⚙️ System Pipeline Architecture")
        st.info("""
        1. **Data Collection & Cleaning**: Handled missing values, encoded categorical features.
        2. **Feature Engineering**: Extracted date traits, created active status thresholds.
        3. **Data Scaling**: Uniformly scaled via Standard Scaler transformation.
        4. **Model Prediction**: Optimization of hyper-parameters via machine learning ensemble models.
        """)

with tab2:
    st.header("🔮 Interactive Inference Engine")
    st.write("Fill out the customer profiles using the fields below to compute a real-time risk classification.")
    
    # Form layout for input fields
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            ui_age_group = st.selectbox("Age Bracket Group", options=list(MAPPINGS["age_group"].keys()))
            ui_gender = st.selectbox("Gender Orientation", options=list(MAPPINGS["gender"].keys()))
            ui_region = st.selectbox("Geographic Region Category", options=list(MAPPINGS["region"].keys()))
            
            st.markdown("---")
            st.subheader("📅 Customer Tenure Info")
            join_year = st.number_input("Join Year", min_value=2000, max_value=2026, value=2022)
            join_month = st.slider("Join Month", min_value=1, max_value=12, value=6)
            join_day = st.slider("Join Day", min_value=1, max_value=31, value=15)

        with col2:
            st.subheader("💳 Membership & Activity")
            ui_membership = st.selectbox("Account Membership Category", options=list(MAPPINGS["membership"].keys()))
            ui_referral = st.selectbox("Joined via Referral Program?", options=list(MAPPINGS["referral"].keys()))
            ui_offer_type = st.selectbox("Preferred Marketing Offer Type", options=list(MAPPINGS["offer_type"].keys()))
            ui_operation_channel = st.selectbox("Primary Operations Device/Channel", options=list(MAPPINGS["operation_channel"].keys()))
            ui_internet = st.selectbox("Internet Network Connectivity type", options=list(MAPPINGS["internet"].keys()))
            
            st.markdown("---")
            st.subheader("📊 Numeric Engagement Metrics")
            time_spent = st.number_input("Average Time Spent (Minutes)", min_value=0.0, value=120.0)
            transaction_value = st.number_input("Average Transaction Value ($)", min_value=0.0, value=550.0)
            login_frequency = st.number_input("Average Login Frequency (Days)", min_value=0.0, value=12.0)
            wallet_points = st.number_input("Current Wallet Reward Points Balance", min_value=0.0, value=1200.0)
            last_login_days = st.number_input("Days Since Last System Login", min_value=-999, value=4)

        with col3:
            st.subheader("📣 Satisfaction & Support Details")
            ui_used_special_discount = st.selectbox("Utilized Special Promotion Discount?", options=list(MAPPINGS["used_special_discount"].keys()))
            ui_offer_application_pref = st.selectbox("Prefers Targeted Promotional Alerts?", options=list(MAPPINGS["offer_application_preference"].keys()))
            ui_past_complaint = st.selectbox("Has Registered a Complaint Globally?", options=list(MAPPINGS["past_complaint"].keys()))
            ui_complaint = st.selectbox("Has an Unresolved Active Support Case?", options=list(MAPPINGS["complaint"].keys()))
            ui_feedback = st.selectbox("Direct Customer Review Sentiment Category", options=list(MAPPINGS["feedback"].keys()))

        submit_btn = st.form_submit_button("Run Analytics Inference Profile", type="primary")

    if submit_btn:

        mapped_data = {
            "age": age,
            "gender": MAPPINGS["gender"][ui_gender],
            "region": MAPPINGS["region"][ui_region],
            "membership": MAPPINGS["membership"][ui_membership],
            "referral": MAPPINGS["referral"][ui_referral],
            "offer_type": MAPPINGS["offer_type"][ui_offer_type],
            "operation_channel": MAPPINGS["operation_channel"][ui_operation_channel],
            "internet": MAPPINGS["internet"][ui_internet],
            "last_login_days": last_login_days,
            "time_spent": time_spent,
            "transaction_value": transaction_value,
            "login_frequency": login_frequency,
            "wallet_points": wallet_points,
            "used_special_discount": MAPPINGS["used_special_discount"][ui_used_special_discount],
            "offer_application_preference": MAPPINGS["offer_application_preference"][ui_offer_application_pref],
            "past_complaint": MAPPINGS["past_complaint"][ui_past_complaint],
            "complaint": MAPPINGS["complaint"][ui_complaint],
            "feedback": MAPPINGS["feedback"][ui_feedback],
            "join_year": join_year,
            "join_month": join_month,
            "join_day": join_day,
            "age_group": MAPPINGS["age_group"][ui_age_group],
            "active_customer": 1 if time_spent > 100 else 0  
        }

        input_df = pd.DataFrame([mapped_data])

        try:
            # Transform and Predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            # Extract probabilities
            stay_probability = probability[0]
            churn_probability = probability[1]

            st.markdown("---")
            st.subheader("🎯 Customer Churn Prediction Report")


            if churn_probability >= 0.80:
                st.error("🚨 Very High Risk Customer")
                st.warning("Immediate retention action is recommended.")
            elif churn_probability >= 0.60:
                st.warning("⚠️ High Risk Customer")
                st.info("Consider loyalty rewards, discounts, and proactive outreach.")
            elif churn_probability >= 0.40:
                st.warning("🟡 Moderate Risk Customer")
                st.info("Customer should be monitored closely.")
            else:
                st.success("🟢 Low Risk Customer")
                st.info("Customer appears engaged and likely to remain.")


            st.markdown("### 📊 Probability Breakdown")
            metric1, metric2 = st.columns(2)
            with metric1:
                st.metric(
                    "Retention Probability",
                    f"{stay_probability:.2%}"
                )
            with metric2:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}"
                )

            st.markdown("### 📌 Business Recommendation")
            if prediction == 1:
                st.error(
                    f"""
                    This customer has a churn probability of {churn_probability:.2%}.

                    Recommended Actions:
                    • Loyalty rewards
                    • Personalized offers
                    • Customer support outreach
                    • Complaint resolution
                    """
                )
            else:
                st.success(
                    f"""
                    This customer has a retention probability of {stay_probability:.2%}.

                    Recommended Actions:
                    • Maintain engagement
                    • Continue loyalty programs
                    • Monitor future activity
                    """
                )

        except Exception as err:
            st.error(f"Execution Error occurred: {err}")


with tab3:
    st.header("💡 Churn Mitigation Strategies")
    st.write("Based on key factors impacting attrition rates, apply the following strategies to your accounts:")
    
    col_ins1, col_ins2, col_ins3 = st.columns(3)
    with col_ins1:
        st.subheader("🎁 Reward Interventions")
        st.markdown(
            "**Wallet Point Thresholds:** Customers falling below specific loyalty point balances show elevated risk factors. "
            "Implement high-value promotional automated voucher issuance to boost spending triggers."
        )
    with col_ins2:
        st.subheader("☎️ Support Pipeline Resolution")
        st.markdown(
            "**Active Complaint Status:** Open issues remain a critical driver for quick account termination. "
            "Flag profiles with active grievances for priority escalation queues to boost customer lifetime value."
        )
    with col_ins3:
        st.subheader("⭐ Premium Category Protection")
        st.markdown(
            "**Basic Membership Risk:** Standard non-tiered members present higher baseline attrition compared to premium variants. "
            "Provide discounted trials to upgrade tiers and lower continuous cancellation risks."
        )