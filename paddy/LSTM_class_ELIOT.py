import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

class LSTMModelTrader:
    def __init__(self, stock, model_path, sequence_length=10, threshold=0.5):
        """
        Parameters:
        - stock: the ticker for the stock (e.g., "STOCK1")
        - model_path: path to the saved Keras model file (e.g., "STOCK1_lstm_model.keras")
        - sequence_length: number of timesteps the model expects as input
        - threshold: probability threshold above which to issue a buy signal (otherwise sell)
        """
        self.stock = stock
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.model = load_model(model_path)
        # NOTE: In a production environment, you should save and load your scaler parameters.
        self.scaler = StandardScaler()  

    def __call__(self, historical_data):
        """
        Process historical data for self.stock, prepare the input sequence,
        and return a trading action based on the model's prediction.
        The function should return a list of tuples, e.g., [("buy", self.stock)]
        or [("sell", self.stock)], or an empty list if there isn't enough data.
        """
        # Filter historical data for the specific stock
        stock_data = [d for d in historical_data if d.get("symbol") == self.stock]
        if len(stock_data) < self.sequence_length:
            return []
        
        # Convert the list of dictionaries to a DataFrame for easier manipulation.
        df = pd.DataFrame(stock_data)
        if "gmtTime" in df.columns:
            df["gmtTime"] = pd.to_datetime(df["gmtTime"])
            df = df.sort_values("gmtTime")
        
        # Determine feature columns. Assumes that during training you removed ['gmtTime', 'symbol', 'target'].
        feature_cols = [col for col in df.columns if col not in ["gmtTime", "symbol", "target"]]
        if not feature_cols:
            return []
        
        # Use the last sequence_length rows for prediction
        df_seq = df.iloc[-self.sequence_length:][feature_cols]
        X_seq = df_seq.values.astype(np.float32)
        
        # Scale the features.
        # NOTE: In production, you should use a scaler fitted on your training data.
        X_seq_scaled = self.scaler.fit_transform(X_seq)
        
        # Reshape to match LSTM model input: (1, sequence_length, num_features)
        X_seq_scaled = np.expand_dims(X_seq_scaled, axis=0)
        
        # Predict using the loaded model.
        pred = self.model.predict(X_seq_scaled)
        
        # Determine action based on the prediction probability.
        # For example, if the predicted probability is above the threshold, signal a "buy",
        # otherwise signal a "sell". You can customize this logic as needed.
        action = "buy" if pred[0][0] > self.threshold else "sell"
        print(f"LSTM prediction for {self.stock}: {pred[0][0]:.4f}, action: {action}")
        return [(action, self.stock)]
