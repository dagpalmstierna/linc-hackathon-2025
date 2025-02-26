import pandas as pd
import talib as ta
import hackathon_linc as lh
import numpy as np

df = pd.read_csv("stockPrices_hourly.csv")
df["gmtTime"] = pd.to_datetime(df["gmtTime"])

stock_dfs = {}

for symbol in df["symbol"].unique():
    df_stock = df[df["symbol"] == symbol].copy()
    cols_to_round = [col for col in df_stock.columns if col not in ["gmtTime", "symbol"]]
    df_stock[cols_to_round] = df_stock[cols_to_round].round(2)
    stock_dfs[symbol] = df_stock

def momentum_strategy(capital=100000, ticker="STOCK1", short_window=2000, long_window=8000, betsize=10):

    start_capital = capital
    df = stock_dfs[ticker]

    df['price'] = (df['askMedian'] + df['bidMedian']) / 2
    df['rsi'] = ta.RSI(df['price'], timeperiod=14)
    
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist  

    
    df.dropna(subset=["rsi", "macd_line", "macd_signal"], inplace=True)

    shares = 0
    last_price = 0 

    for index, row in df.iterrows():
        price = row["price"]
        rsi_value = row["rsi"]
        
        macd_val = row["macd_line"]
        signal_val = row["macd_signal"]
    
        last_price = price  

        if rsi_value < 30 and macd_val > signal_val:
            possible_shares = int(capital * 0.3 // price)
            if possible_shares > 0:
                capital -= possible_shares * price  
                shares += possible_shares  
        
        elif rsi_value > 70 and shares > 0 and macd_val < signal_val:
            shares_to_sell = int(shares * 0.3)  
            capital += shares_to_sell * price  
            shares -= shares_to_sell  

    if shares > 0:
        capital += shares * last_price
        shares = 0

    profit_loss = capital - start_capital
    print(f"{ticker} | Start: {start_capital}, End: {capital}, P/L: {profit_loss}, Shares: {shares}")

    return capital

for key in stock_dfs.keys():
    momentum_strategy(ticker=key)

