import pandas as pd
import talib as ta
import hackathon_linc as lh

df = pd.read_csv("stockPrices_hourly.csv")
df["gmtTime"] = pd.to_datetime(df["gmtTime"])

stock_dfs = {}

for symbol in df["symbol"].unique():
    
    df_stock = df[df["symbol"] == symbol].copy() 
    cols_to_round = [col for col in df_stock.columns if col not in ["gmtTime", "symbol"]]
    df_stock[cols_to_round] = df_stock[cols_to_round].round(2)
    stock_dfs[symbol] = df_stock


def rsi_strategy(capital=100000, ticker="STOCK1"):

    start_capital = capital
    df_stock = stock_dfs[ticker].copy()
    df_stock['price'] = (df_stock['askMedian'] + df_stock['bidMedian']) / 2
    
    df_stock['rsi'] = ta.RSI(df_stock['price'], timeperiod=14)

   
    df_stock.dropna(subset=['rsi'], inplace=True)
    position = False
    shares = 0
    
    for idx, row in df_stock.iterrows():
        price = row["price"]
        rsi_value = row["rsi"]

        # BUY if RSI < 30 and not in position
        if not position and rsi_value < 30:
            possible_shares = int(capital // price)
            if possible_shares > 0:
                shares = possible_shares
                capital -= shares * price
                position = True

        # SELL if RSI > 70 and in position
        elif position and rsi_value > 70:
            capital += shares * price
            shares = 0
            position = False

    # 4) If still in position at the end, sell at last known price
    if position and shares > 0:
        last_price = df_stock.iloc[-1]["price"]
        capital += shares * last_price
        shares = 0
        position = False

    profit_loss = capital - start_capital
    print(f"{ticker} | Start: {start_capital}, End: {capital}, P/L: {profit_loss}")
    return capital

for key in stock_dfs.keys():
    rsi_strategy(ticker=key)
