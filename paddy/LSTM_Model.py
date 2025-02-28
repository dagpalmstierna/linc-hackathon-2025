import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import hackathon_linc as lh
import math
import joblib


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
            model_path = f"/home/paddy/Documents/linc_hackathon/linc_hackathon_2025/paddy/{model_file}_lstm_model.keras"
            print(model_path)
            try:
                self.models[stock_name] = self.custom_load_model(model_path)
            except Exception as e:
                print(f"Error loading model for {stock_name}: {e}")

            self.scalers[stock_name] = joblib.load(f"/home/paddy/Documents/linc_hackathon/linc_hackathon_2025/paddy/{model_file}_scaler.pkl")
            print(f"Loaded model for {stock_name} from {model_path}")
    
    def custom_load_model(self, model_path):
        with open(model_path, 'r') as f:
            config = f.read()
        config = config.replace('"batch_shape":', '"batch_input_shape":')
        model = model_from_json(config)
        model.load_weights(model_path.replace('.json', '.keras'))
        return model

    def preprocess_data(self, df):
        scaler = self.scalars[df['symbol'].iloc[0]]
        if self.features is None:
            self.features = [col for col in df.columns if col not in ['gmtTime', 'symbol', 'target']]

        cols_to_round = [col for col in df_stock.columns if col not in ["gmtTime", "symbol"]]
        df_stock[cols_to_round] = df_stock[cols_to_round].round(2)

        # Time-based features
        df_stock['hour'] = df_stock['gmtTime'].dt.hour
        df_stock['day_of_week'] = df_stock['gmtTime'].dt.dayofweek

        # Rolling statistics
        df_stock['askMedian_rolling_mean_3h'] = df_stock['askMedian'].rolling(window=3, min_periods=1).mean()
        df_stock['bidMedian_rolling_mean_3h'] = df_stock['bidMedian'].rolling(window=3, min_periods=1).mean()
        df_stock['askMedian_rolling_std_3h'] = df_stock['askMedian'].rolling(window=3, min_periods=1).std()
        df_stock['bidMedian_rolling_std_3h'] = df_stock['bidMedian'].rolling(window=3, min_periods=1).std()

        # Percentage changes
        df_stock['askMedian_pct_change'] = df_stock['askMedian'].pct_change()
        df_stock['bidMedian_pct_change'] = df_stock['bidMedian'].pct_change()

        # Spread-related features
        df_stock['spread_ratio'] = df_stock['spreadMedian'] / (df_stock['askMedian'] + df_stock['bidMedian'])
        # df_stock['spread_pct_change'] = df_stock['spreadMedian'].pct_change()

        # Volume-related features
        df_stock['askVolume_relative'] = df_stock['askVolume'] / df_stock['askVolume'].rolling(window=5, min_periods=1).mean()
        df_stock['bidVolume_relative'] = df_stock['bidVolume'] / df_stock['bidVolume'].rolling(window=5, min_periods=1).mean()
        df_stock['volume_imbalance'] = (df_stock['askVolume'] - df_stock['bidVolume']) / (df_stock['askVolume'] + df_stock['bidVolume'])

        # Lagged features (e.g., previous hour's values)
        for lag in range(1, 25):  # Add lags for the last 3 hours
            df_stock[f'askMedian_lag_{lag}'] = df_stock['askMedian'].shift(lag)
            df_stock[f'bidMedian_lag_{lag}'] = df_stock['bidMedian'].shift(lag)
            df_stock[f'spreadMedian_lag_{lag}'] = df_stock['spreadMedian'].shift(lag)

        # Target variable: Direction of price movement (1 if bidMedian increases next hour, 0 otherwise)
        # df_stock['target'] = (df_stock['bidMedian'].shift(-20) > df_stock['bidMedian']).astype(int)

        # Drop rows with missing values (due to lags and rolling features)
        df_stock = df_stock.dropna()

        df = df_stock.copy()    
        
        X = scaler.transform(X)
        X = df[self.features]

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
            X_seq= self.preprocess_data(df_stock)

            if symbol in self.models:
                model = self.models[symbol]
                predictions[symbol] = model.predict(X_seq)
                print(f"Predictions for {symbol}: {predictions[symbol]}")
            else:
                print(f"No model found for {symbol}")

        return predictions



# Example usage
model_files = ['STOCK20', 'STOCK1', 'STOCK18', 'STOCK21', 'STOCK16', 'STOCK11', 'STOCK12',
               'STOCK17', 'STOCK6', 'STOCK9', 'STOCK2', 'STOCK3', 'STOCK14', 'STOCK10',
               'STOCK8', 'STOCK13', 'STOCK5', 'STOCK4', 'STOCK19', 'INDEX1']

lh.init('5d5249da-7e7e-438b-a392-692f71364fd0')
historical_data = lh.get_historical_data(100)

manager = LSTMModelManager(model_files)

def example_how_function_called(historical_data):
    predictions = manager.predict(historical_data)
    return predictions