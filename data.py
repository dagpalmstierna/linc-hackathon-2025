import pandas as pd

df = pd.read_csv("stockPrices_hourly.csv")

df_stock1 = df[df["symbol"] == "STOCK1"]
df_stock2 = df[df["symbol"] == "STOCK2"]
df_stock3 = df[df["symbol"] == "STOCK3"]
df_stock4 = df[df["symbol"] == "STOCK4"]
df_stock5 = df[df["symbol"] == "STOCK5"]
df_stock6 = df[df["symbol"] == "STOCK6"]
df_stock7 = df[df["symbol"] == "STOCK7"]
df_stock8 = df[df["symbol"] == "STOCK8"]
df_stock9 = df[df["symbol"] == "STOCK9"]
df_stock10 = df[df["symbol"] == "STOCK10"]


'''df['gmtTime'] = pd.to_datetime(df['gmtTime'])

symbols = df['symbol'].unique()

stock_dfs = {}

for symbol in symbols:
    df_stock = df[df["symbol"] == symbol].copy()  
    
    df_stock['hour'] = df_stock['gmtTime'].dt.hour
    df_stock['day_of_week'] = df_stock['gmtTime'].dt.dayofweek

    df_stock['askMedian_rolling_mean_3h'] = df_stock['askMedian'].rolling(window=3, min_periods=1).mean()
    df_stock['bidMedian_rolling_std_3h'] = df_stock['bidMedian'].rolling(window=3, min_periods=1).std()

    df_stock['askMedian_pct_change'] = df_stock['askMedian'].pct_change()
    df_stock['bidMedian_pct_change'] = df_stock['bidMedian'].pct_change()

    df_stock['spread_ratio'] = df_stock['spreadMedian'] / (df_stock['askMedian'] + df_stock['bidMedian'])

    df_stock['askVolume_relative'] = df_stock['askVolume'] / df_stock['askVolume'].rolling(window=5, min_periods=1).mean()
    df_stock['bidVolume_relative'] = df_stock['bidVolume'] / df_stock['bidVolume'].rolling(window=5, min_periods=1).mean()

    
    stock_dfs[symbol] = df_stock.dropna()

print(stock_dfs['STOCK1'].head())'''
