import pandas as pd
import talib as ta
import hackathon_linc as lh
import numpy as np
    
def momentum_strategy(historical_data, capital=100000, ticker="STOCK1"):
    
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2
    
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist  
   
    df["rsi"] = ta.RSI(df["price"], timeperiod=14)

    current_date = df.iloc[-1]
    
    rsi_value = current_date["rsi"]
    
    macd_val = current_date["macd_line"]
    
    signal_val = current_date["macd_signal"]
    
    if rsi_value < 30 and macd_val > signal_val:
        return 'buy'
    elif rsi_value > 70 and macd_val < signal_val:
        return 'sell'
    else: 
        return "hold"
    
print(momentum_strategy()) 