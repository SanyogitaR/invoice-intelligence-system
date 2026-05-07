import joblib
import pandas as pd
import os

# Updated path to look into the invoice_flagging/models folder
MODEL_PATH = "../invoice_flagging/models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
        
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    """
    model = load_model()
    if model is None:
        return None
        
    input_df = pd.DataFrame(input_data)
    # Perform prediction and round the result
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    # Example inference run (local testing)
    # Ensure "Dollars" matches the feature name used during training
    sample_data = {
        "Dollars": [18500, 9000]
    }

    prediction = predict_freight_cost(sample_data)
    
    if prediction is not None:
        print("\n--- Freight Prediction Results ---")
        print(prediction)