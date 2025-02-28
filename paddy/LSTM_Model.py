import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

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
            model_path = f"{model_file}_lstm_model.keras"
            self.models[stock_name] = load_model(model_path)
            print(f"Loaded model for {stock_name} from {model_path}")

    def preprocess_data(self, df):
        if self.features is None:
            self.features = [col for col in df.columns if col not in ['gmtTime', 'symbol', 'target']]
        scaler = StandardScaler()
        X = scaler.fit_transform(df[self.features])
        y = df[self.target].values

        X_seq = []
        y_seq = []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        return X_seq, y_seq, scaler

    def predict(self, historical_data):
        df = pd.DataFrame(historical_data)
        df.index = pd.to_datetime(df["gmtTime"])

        predictions = {}
        for symbol in df["symbol"].unique():
            df_stock = df[df["symbol"] == symbol].copy()
            X_seq, y_seq, scaler = self.preprocess_data(df_stock)
            self.scalers[symbol] = scaler

            if symbol in self.models:
                model = self.models[symbol]
                predictions[symbol] = model.predict(X_seq)
            else:
                print(f"No model found for {symbol}")

        return predictions

# Example usage
model_files = ['STOCK20', 'STOCK1', 'STOCK18', 'STOCK21', 'STOCK16', 'STOCK11', 'STOCK12',
               'STOCK17', 'STOCK6', 'STOCK9', 'STOCK2', 'STOCK3', 'STOCK14', 'STOCK10',
               'STOCK8', 'STOCK13', 'STOCK5', 'STOCK4', 'STOCK19', 'INDEX1']

manager = LSTMModelManager(model_files)

def example_how_function_called(historical_data):
    predictions = manager.predict(historical_data)
    return predictions