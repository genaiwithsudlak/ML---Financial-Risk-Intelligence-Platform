import streamlit as st
import requests
import json
import pandas as pd

# API URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial Risk Intelligence", layout="wide")

st.title("🛡️ Financial Risk Intelligence Platform")
st.markdown("Real-time Fraud Detection & Loan Default Prediction")

# Tabs for each dataset
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💳 Credit Card Fraud", 
    "💻 IEEE-CIS Fraud", 
    "📱 PaySim Fraud", 
    "🏦 Give Me Some Credit", 
    "🏠 Home Credit Risk", 
    "🤝 Lending Club"
])

def predict(endpoint, payload):
    try:
        response = requests.post(f"{API_URL}/predict/{endpoint}", json=payload)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}. Is the API running?")
        return None

# --- Tab 1: Credit Card Fraud ---
with tab1:
    st.subheader("Credit Card Transaction Risk")
    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Transaction Amount ($)", value=100.0)
            age = st.number_input("Cardholder Age", value=30, min_value=18, max_value=100)
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12)
            category = st.selectbox("Category", ["grocery_pos", "gas_transport", "online_shopping", "luxury", "misc_net"])
        
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            payload = {
                "amt": amt, "age": age, "hour": hour, "category": category,
                "merchant": "M_ID_Demo", "day_of_week": 1, 
                "trans_date_trans_time": "2023-01-01 12:00:00",
                "cc_num": 123456789, "dob": "1990-01-01"
            }
            res = predict("credit_card", payload)
            if res:
                st.metric("Risk Score", f"{res['risk_score']:.4f}")
                if res['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Legitimate Transaction")

# --- Tab 2: IEEE-CIS Fraud ---
with tab2:
    st.subheader("Online Transaction Risk (IEEE-CIS)")
    with st.form("ieee_form"):
        col1, col2 = st.columns(2)
        with col1:
            tx_amt = st.number_input("Transaction Amount", value=50.0)
            product = st.selectbox("Product Code", ["W", "H", "C", "R", "S"])
        with col2:
            email = st.text_input("Email Domain", "gmail.com")
            card1 = st.number_input("Card Signature 1", value=10000)
            
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            payload = {
                "TransactionAmt": tx_amt, "ProductCD": product,
                "P_emaildomain": email, "card1": card1,
                "card2": 321.0, "addr1": 123.0
            }
            res = predict("ieee", payload)
            if res:
                st.metric("Risk Score", f"{res['risk_score']:.4f}")
                if res['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Legitimate")

# --- Tab 3: PaySim Fraud ---
with tab3:
    st.subheader("Mobile Money Transfer Risk")
    with st.form("paysim_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount", value=1000.0)
            tx_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"])
        with col2:
            old_bal = st.number_input("Sender Old Balance", value=1000.0)
            new_bal = st.number_input("Sender New Balance", value=0.0)
            
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            payload = {
                "step": 1, "type": tx_type, "amount": amount,
                "oldbalanceOrg": old_bal, "newbalanceOrig": new_bal,
                "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                "nameOrig": "C1", "nameDest": "C2"
            }
            res = predict("paysim", payload)
            if res:
                st.metric("Risk Score", f"{res['risk_score']:.4f}")
                if res['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Legitimate")

# --- Tab 4: Give Me Some Credit ---
with tab4:
    st.subheader("Borrower Delinquency Probability")
    with st.form("gmsc_form"):
        col1, col2 = st.columns(2)
        with col1:
            revol_util = st.slider("Revolving Utilization", 0.0, 2.0, 0.3)
            age = st.number_input("Age", 20, 100, 40)
            income = st.number_input("Monthly Income", value=5000.0)
        with col2:
            debt_ratio = st.number_input("Debt Ratio", value=0.3)
            past_due = st.number_input("Times 30-59 Days Past Due", 0, 10, 0)
            
        submitted = st.form_submit_button("Predict Default")
        if submitted:
            payload = {
                "RevolvingUtilizationOfUnsecuredLines": revol_util,
                "age": age, "MonthlyIncome": income, "DebtRatio": debt_ratio,
                "NumberOfTime30_59DaysPastDueNotWorse": past_due,
                "NumberOfOpenCreditLinesAndLoans": 5, "NumberOfDependents": 1.0
            }
            res = predict("gmsc", payload)
            if res:
                st.metric("Default Probability", f"{res['default_probability']:.2%}")
                if res['prediction'] == 1:
                    st.warning("⚠️ HIGH DEFAULT RISK")
                else:
                    st.success("✅ Low Risk")

# --- Tab 5: Home Credit ---
with tab5:
    st.subheader("Home Loan Application Risk")
    with st.form("homecredit_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Total Income", value=50000.0)
            credit = st.number_input("Credit Amount", value=100000.0)
        with col2:
            annuity = st.number_input("Annuity", value=5000.0)
            contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
            
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            payload = {
                "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity, "NAME_CONTRACT_TYPE": contract,
                "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -2000, "SK_ID_CURR": 999
            }
            res = predict("homecredit", payload)
            if res:
                st.metric("Default Probability", f"{res['default_probability']:.2%}")
                if res['prediction'] == 1:
                    st.warning("⚠️ HIGH RISK")
                else:
                    st.success("✅ Low Risk")

# --- Tab 6: Lending Club ---
with tab6:
    st.subheader("P2P Loan Default Risk")
    with st.form("lendingclub_form"):
        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input("Loan Amount", value=10000.0)
            int_rate = st.number_input("Interest Rate (%)", value=10.0)
            annual_inc = st.number_input("Annual Income", value=60000.0)
        with col2:
            dti = st.number_input("DTI", value=15.0)
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
            term = st.selectbox("Term", [" 36 months", " 60 months"])
            
        submitted = st.form_submit_button("Predict Default")
        if submitted:
            payload = {
                "loan_amnt": loan_amnt, "int_rate": int_rate,
                "annual_inc": annual_inc, "dti": dti,
                "grade": grade, "term": term, "emp_length": "10+ years"
            }
            res = predict("lendingclub", payload)
            if res:
                st.metric("Default Probability", f"{res['default_probability']:.2%}")
                if res['prediction'] == 1:
                    st.warning("⚠️ HIGH DEFAULT RISK")
                else:
                    st.success("✅ Low Risk")
