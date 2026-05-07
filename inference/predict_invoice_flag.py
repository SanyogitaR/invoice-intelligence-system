import joblib
import pandas as pd
import os

# This path goes UP one level, into invoice_flagging, into models, 
# and then into your classification subfolder.
MODEL_PATH = "../invoice_flagging/models/classification/predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    model = load_model()
    if model is None:
        return None
    
    input_df = pd.DataFrame(input_data)
    # Using the exact logic from your screenshot
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    # Example data matching your Classification model features
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [1200.0],
        "Freight": [45.0],
        "days_po_to_invoice": [5],
        "total_brands": [2],
        "total_item_quantity": [50],
        "total_item_dollars": [1195.0],
        "avg_receiving_delay": [2.5]
    }

    prediction = predict_invoice_flag(sample_data)
    if prediction is not None:
        print("\n--- Inference Results ---")
        print(prediction)