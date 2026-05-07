import joblib
import pandas as pd
import numpy as np
import os

# Path relative to the project root
MODEL_PATH = "freight_cost_prediction/models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        # Using a direct load to handle the version warning gracefully
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_freight_cost(input_data):
    model = load_model()
    if model is None:
        return None
        
    # 1. Create DataFrame from input
    input_df = pd.DataFrame(input_data)
    
    try:
        # 2. MATCH THE MODEL'S EXPECTATION
        # The terminal confirmed the model was trained ONLY on ['Dollars']
        # We use .values to avoid any hidden name/space issues
        data_values = input_df[['Dollars']].values
        
        # 3. Perform prediction
        predictions = model.predict(data_values)
        
        # 4. Add result back to the dataframe
        # Flatten the prediction in case it comes back as a nested array
        input_df['Predicted_Freight'] = np.array(predictions).flatten().round(2)
        return input_df

    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

if __name__ == "__main__":
    # Test it locally with the single feature logic
    sample_data = {"Quantity": [100], "Dollars": [5000]}
    result = predict_freight_cost(sample_data)
    if result is not None:
        print("\n--- Success! Prediction Match Found ---")
        print(result)