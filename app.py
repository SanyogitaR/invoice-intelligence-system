import streamlit as st
import pandas as pd
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag 

# --- UI CONFIG ---
st.set_page_config(page_title="Vendor Intelligence Portal", layout="wide")

# --- CUSTOM CSS FOR "PERSIAN BLUE & DARK" THEME ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header Styling */
    h1 {
        color: #1E90FF !important; 
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0A192F !important; /* Deep Persian Blue Night */
        border-right: 1px solid #1E90FF;
    }

    /* Button Glow Effect */
    div.stButton > button:first-child {
        background-color: #1E90FF;
        color: white;
        border-radius: 10px;
        border: none;
        box-shadow: 0 0 15px rgba(30, 144, 255, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-weight: bold;
        height: 3em;
    }
    
    div.stButton > button:hover {
        box-shadow: 0 0 25px rgba(30, 144, 255, 0.8);
        transform: translateY(-2px);
        background-color: #2b65ec;
        color: white;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True) # Fixed the argument name here

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    app_mode = st.radio("Navigation", ["Freight Prediction", "Risk Analysis"])
    st.markdown("---")
    st.info("""
    **Intelligence Metrics:**
    * 🚀 Logistics Optimization
    * 🛡️ Fraud Guard AI
    * 📊 Financial Precision
    """)

# --- MAIN HEADER ---
st.title("🏢 Vendor Intelligence Portal")
st.markdown("#### *AI-powered decision support for finance and logistics.*")
st.divider()

# --- APP LOGIC ---

if app_mode == "Freight Prediction":
    st.subheader("📦 Forecast Freight Expenditure")
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Item Quantity", min_value=1, value=10)
    with col2:
        dollars = st.number_input("Invoice Total ($)", min_value=1.0, value=500.0)

    if st.button("✨ Calculate Expected Freight"):
        input_dict = {"Quantity": [quantity], "Dollars": [dollars]}
        result_df = predict_freight_cost(input_dict)
        
        if result_df is not None:
            prediction = result_df['Predicted_Freight'].iloc[0]
            st.metric(label="Calculated Freight Cost", value=f"${prediction:,.2f}")
            st.balloons()

elif app_mode == "Risk Analysis":
    st.subheader("🚩 Invoice Risk Profiler")
    col1, col2 = st.columns(2)
    with col1:
        inv_qty = st.number_input("Invoice Quantity", min_value=1, value=50)
        inv_dlrs = st.number_input("Invoice Dollars ($)", min_value=1.0, value=1200.0)
        freight = st.number_input("Freight Component ($)", min_value=0.0, value=45.0)
    
    with col2:
        item_qty = st.number_input("Total Item Quantity", min_value=1, value=50)
        item_dlrs = st.number_input("Total Item Dollars ($)", min_value=1.0, value=1195.0)
    
    st.markdown("---")
    if st.button("🔍 Run AI Risk Assessment"):
        input_risk = {
            "invoice_quantity": [inv_qty],
            "invoice_dollars": [inv_dlrs],
            "Freight": [freight],
            "total_item_quantity": [item_qty],
            "total_item_dollars": [item_dlrs]
        }
        
        risk_result = predict_invoice_flag(input_risk)
        
        if risk_result is not None:
            is_flagged = risk_result['Predicted_Flag'].iloc[0]
            
            if is_flagged == 1:
                st.error("🚨 **HIGH RISK**: Manual review required.")
            else:
                st.success("✅ **LOW RISK**: Invoice cleared for processing.")
