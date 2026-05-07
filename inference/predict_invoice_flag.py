import joblib
import pandas as pd
import numpy as np
import os

# --- ABSOLUTE PATHS ---
MODEL_PATH = r"D:\data science\inventory invoice intellignece ml\invoice_flagging\models\classification\predict_flag_invoice.pkl"
SCALER_PATH = r"D:\data science\inventory invoice intellignece ml\invoice_flagging\models\scaler.pkl"

def load_artifacts():
    """Helper to load the model and scaler once."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model missing at {MODEL_PATH}")
        return None, None
    if not os.path.exists(SCALER_PATH):
        print(f"❌ ERROR: Scaler missing at {SCALER_PATH}")
        return None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading .pkl files: {e}")
        return None, None

# Load them globally so they stay in memory
MODEL, SCALER = load_artifacts()

def predict_invoice_flag(input_data):
    # Check if artifacts were loaded correctly
    if MODEL is None or SCALER is None:
        return None
    
    # Create DataFrame
    input_df = pd.DataFrame(input_data)
    
    try:
        # 1. Feature Selection (Match the 5 features from your GridSearchCV)
        features = ['invoice_quantity', 'invoice_dollars', 'Freight', 
                    'total_item_quantity', 'total_item_dollars']
        
        # Filter and reorder columns
        input_ready = input_df[features]
        
        # 2. Scale the data
        input_scaled = SCALER.transform(input_ready)

        # 3. Predict
        predictions = MODEL.predict(input_scaled)
        input_df['Predicted_Flag'] = np.array(predictions).flatten().astype(int)
        
        return input_df

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        # Helpful debug: check if column names match
        print(f"DEBUG: Your input columns: {list(input_df.columns)}")
        return None

if __name__ == "__main__":
    # Test data - exactly 5 features
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [1200.0],
        "Freight": [45.0],
        "total_item_quantity": [50],
        "total_item_dollars": [1195.0]
    }

    print("\n--- Running Local Terminal Test ---")
    result = predict_invoice_flag(sample_data)
    
    if result is not None:
        print("✅ SUCCESS!")
        print(result[['invoice_dollars', 'total_item_dollars', 'Predicted_Flag']])
        
        flag_val = result['Predicted_Flag'].iloc[0]
        status = "🚩 HIGH RISK" if flag_val == 1 else "✅ LOW RISK"
        print(f"\nResult: {status}")
    else:
        print("❌ Test failed. See errors above.")