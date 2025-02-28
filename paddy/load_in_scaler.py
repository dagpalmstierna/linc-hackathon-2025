import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Define your stock and paths
stock = "STOCK1"
scaler_filename = f"{stock}_scaler.pkl"
model_filename = f"{stock}_lstm_model.keras"

# Load the scaler and model
scaler = joblib.load(scaler_filename)
model = load_model(model_filename)

# Suppose you have new data in a DataFrame new_df with the same feature columns:
# new_df = pd.DataFrame(...)

# Ensure new_df only contains the features used during training:
features = [col for col in new_df.columns if col not in ['gmtTime', 'symbol', 'target']]
X_new = new_df[features].values

# Scale the new data using the saved scaler
X_new_scaled = scaler.transform(X_new)

# Prepare sequence data for prediction (ensure you have enough timesteps)
sequence_length = 10
if len(X_new_scaled) >= sequence_length:
    X_seq = X_new_scaled[-sequence_length:]
    X_seq = np.expand_dims(X_seq, axis=0)  # reshape to (1, sequence_length, num_features)
    
    # Make prediction
    prediction = model.predict(X_seq)
    print(f"Prediction: {prediction[0][0]:.4f}")
else:
    print("Not enough data for prediction.")
