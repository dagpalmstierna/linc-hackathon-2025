import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import hackathon_linc as lh
import math
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class LSTMModelManager:
    def __init__(self, model_files):
        self.models = {}
        self.scalers = {}
        self.sequence_length = 10  # Use last 10 hours as input
        self.features = None
        self.target = 'target'
        self.load_models(model_files)

    def load_models(self, model_files):
        for model_file in model_files:
            stock_name = model_file.split('_')[0]
            model_path = os.path.join(MODELS_DIR, f"{model_file}_lstm_model.keras")
            print(model_path)
            try:
                self.models[stock_name] = self.custom_load_model(model_path)
            except Exception as e:
                print(f"Error loading model for {stock_name}: {e}")

            scaler_path = os.path.join(MODELS_DIR, f"{model_file}_scaler.pkl")
            self.scalers[stock_name] = joblib.load(scaler_path)
            print(f"Loaded model for {stock_name} from {model_path}")

    def custom_load_model(self, model_path):
        with open(model_path, 'r') as f:
            config = f.read()
        config = config.replace('"batch_shape":', '"batch_input_shape":')
        model = model_from_json(config)
        model.load_weights(model_path.replace('.json', '.keras'))
        return model

    def preprocess_data(self, df):
        scaler = self.scalers[df['symbol'].iloc[0]]
        if self.features is None:
            self.features = [col for col in df.columns if col not in ['gmtTime', 'symbol', 'target']]

        cols_to_round = [col for col in df.columns if col not in ["gmtTime", "symbol"]]
        df[cols_to_round] = df[cols_to_round].round(2)

        df['hour'] = df['gmtTime'].dt.hour
        df['day_of_week'] = df['gmtTime'].dt.dayofweek

        df['askMedian_rolling_mean_3h'] = df['askMedian'].rolling(window=3, min_periods=1).mean()
        df['bidMedian_rolling_mean_3h'] = df['bidMedian'].rolling(window=3, min_periods=1).mean()
        df['askMedian_rolling_std_3h'] = df['askMedian'].rolling(window=3, min_periods=1).std()
        df['bidMedian_rolling_std_3h'] = df['bidMedian'].rolling(window=3, min_periods=1).std()

        df['askMedian_pct_change'] = df['askMedian'].pct_change()
        df['bidMedian_pct_change'] = df['bidMedian'].pct_change()

        df['spread_ratio'] = df['spreadMedian'] / (df['askMedian'] + df['bidMedian'])

        df['askVolume_relative'] = df['askVolume'] / df['askVolume'].rolling(window=5, min_periods=1).mean()
        df['bidVolume_relative'] = df['bidVolume'] / df['bidVolume'].rolling(window=5, min_periods=1).mean()
        df['volume_imbalance'] = (df['askVolume'] - df['bidVolume']) / (df['askVolume'] + df['bidVolume'])

        for lag in range(1, 25):
            df[f'askMedian_lag_{lag}'] = df['askMedian'].shift(lag)
            df[f'bidMedian_lag_{lag}'] = df['bidMedian'].shift(lag)
            df[f'spreadMedian_lag_{lag}'] = df['spreadMedian'].shift(lag)

        df = df.dropna()

        X = df[self.features]
        X = scaler.transform(X)

        X_seq = []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])

        X_seq = np.array(X_seq)
        return X_seq

    def predict(self, historical_data):
        df = pd.DataFrame(historical_data)
        df.index = pd.to_datetime(df["gmtTime"])

        predictions = {}
        for symbol in df["symbol"].unique():
            df_stock = df[df["symbol"] == symbol].copy()
            X_seq = self.preprocess_data(df_stock)

            if symbol in self.models:
                model = self.models[symbol]
                predictions[symbol] = model.predict(X_seq)
                print(f"Predictions for {symbol}: {predictions[symbol]}")
            else:
                print(f"No model found for {symbol}")

        return predictions


# Example usage
if __name__ == "__main__":
    model_files = ['STOCK20', 'STOCK1', 'STOCK18', 'STOCK21', 'STOCK16', 'STOCK11', 'STOCK12',
                   'STOCK17', 'STOCK6', 'STOCK9', 'STOCK2', 'STOCK3', 'STOCK14', 'STOCK10',
                   'STOCK8', 'STOCK13', 'STOCK5', 'STOCK4', 'STOCK19', 'INDEX1']

    lh.init('YOUR_API_KEY')
    historical_data = lh.get_historical_data(100)

    manager = LSTMModelManager(model_files)
    predictions = manager.predict(historical_data)
