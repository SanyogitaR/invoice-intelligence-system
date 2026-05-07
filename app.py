 import streamlit as st
import pandas as pd
import numpy as np
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# --- Page Configuration ---
st.set_page_config(
    page_title="Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    selected_module = st.radio(
        "Navigation",
        ["Freight Cost Prediction", "Invoice Risk Analysis"],
        help="Select the AI module you wish to use."
    )
    st.markdown("---")
    st.info("**Business Impact:**\n- Faster Operations\n- Reduced Fraud\n- Precise Budgeting")

# --- Header Section ---
st.title("🏢 Vendor Invoice Intelligence Portal")
st.markdown("AI-driven insights for modern logistics and finance teams.")

# --- MODULE 1: Freight Cost Prediction ---
if selected_module == "Freight Cost Prediction":
    st.subheader("📦 Forecast Freight Expenditure")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input("Item Quantity", min_value=1, value=1200)
        with col2:
            dollars = st.number_input("Invoice Total ($)", min_value=1.0, value=18500.0)
        
        if st.button("Calculate Expected Freight"):
            input_data = {"Quantity": [quantity], "Dollars": [dollars]}
            # Running inference
            prediction = predict_freight_cost(input_data)['Predicted_Freight']
            
            st.success("Analysis Complete")
            st.metric(label="Estimated Freight Cost", value=f"${prediction[0]:,.2f}")

# --- MODULE 2: Invoice Risk Analysis ---
else:
    st.subheader("🚩 Automated Risk Screening")
    
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inv_qty = st.number_input("Invoice Quantity", value=50)
            freight = st.number_input("Actual Freight Paid", value=1.73)
        with c2:
            inv_dlrs = st.number_input("Invoice Dollars", value=352.95)
            total_item_qty = st.number_input("Total Item Qty", value=162)
        with c3:
            total_item_dlrs = st.number_input("Total Item Dollars", value=2476.0)
            
        submit = st.form_submit_button("Evaluate Risk Profile")

    if submit:
        input_data = {
            "invoice_quantity": [inv_qty], "invoice_dollars": [inv_dlrs],
            "Freight": [freight], "total_item_quantity": [total_item_qty],
            "total_item_dollars": [total_item_dlrs]
        }
        
        # Adding dummy columns if your model requires features not in the UI
        # (Change these based on your specific train.py requirements)
        input_data.update({
            "days_po_to_invoice": [5], "total_brands": [1], "avg_receiving_delay": [0]
        })
        
        result = predict_invoice_flag(input_data)['Predicted_Flag']
        is_flagged = bool(result[0])

        if is_flagged:
            st.error("🚨 **High Risk Detected:** This invoice requires MANUAL APPROVAL.")
        else:
            st.success("✅ **Low Risk:** Invoice cleared for automated payment.")