import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("stockPrices_hourly.csv")

df["gmtTime"] = pd.to_datetime(df["gmtTime"])
df['hour'] = df['gmtTime'].dt.hour
df['day_of_week'] = df['gmtTime'].dt.dayofweek

stock_dfs = {}

for symbol in df["symbol"].unique():
    df_stock = df[df["symbol"] == symbol].copy()  
    
    cols_to_round = [col for col in df_stock.columns if col not in ["gmtTime"]]
    
    df_stock[cols_to_round] = df_stock[cols_to_round].round(2)

    stock_dfs[symbol] = df_stock

df = stock_dfs['STOCK1']

short_window = 50
long_window = 200

df['short_mavg'] = df['askMedian'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['askMedian'].rolling(window=long_window, min_periods=1).mean()

df['signal'] = 0  # Default: no signal
df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0)  # Buy signal
df['position'] = df['signal'].diff()  # Buy (1) or Sell (-1) signal


plt.figure(figsize=(14, 8))
plt.plot(df.index, df['askMedian'], label='Price (askMedian)', color='blue', alpha=0.5)
plt.plot(df.index, df['short_mavg'], label=f'{short_window}-Period Moving Average', color='red', alpha=0.7)
plt.plot(df.index, df['long_mavg'], label=f'{long_window}-Period Moving Average', color='green', alpha=0.7)

plt.plot(df[df['position'] == 1].index, 
         df['short_mavg'][df['position'] == 1], 
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

plt.plot(df[df['position'] == -1].index, 
         df['short_mavg'][df['position'] == -1], 
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Moving Average Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.show()


