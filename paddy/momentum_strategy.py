import pandas as pd
import talib as ta
import hackathon_linc as lh
import numpy as np
    
#får in portfölj och ska 
def momentum_strategy(historical_data):
    
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

    df["rsi"] = ta.RSI(df["price"], timeperiod=14)
    
    
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist  
   
    stock_dfs = {}

    for symbol in df["symbol"].unique():
        df_stock = df[df["symbol"] == symbol].copy()  
        stock_dfs[symbol] = df_stock
    
    report = {}
    
    for key in stock_dfs.keys():

        current_date = stock_dfs[key].iloc[-1]
        rsi_value = current_date["rsi"]
        macd_val = current_date["macd_line"]
        signal_val = current_date["macd_signal"]

        if  macd_val > signal_val and rsi_value < 49:
            report[key] = ['buy', np.abs(rsi_value-50)]

        elif macd_val < signal_val and rsi_value > 51:
            report[key] = ['sell', np.abs(rsi_value-50)]

       
    return recommendation(report)

def rsi_strategy(historical_data):
    
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

    df["rsi"] = ta.RSI(df["price"], timeperiod=14)
    
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    stock_dfs = {}

    for symbol in df["symbol"].unique():
        df_stock = df[df["symbol"] == symbol].copy()  
        stock_dfs[symbol] = df_stock
    
    report = {}
    
    for key in stock_dfs.keys():

        current_date = stock_dfs[key].iloc[-1]
        rsi_value = current_date["rsi"]
        
        if rsi_value < 45:
            report[key] = ['buy', np.abs(rsi_value-50)]

        elif rsi_value > 55:
            report[key] = ['sell', np.abs(rsi_value-50)]
    return recommendation(report)

def recommendation(report):
    if report == {}:
        return []
    
    sorted_items = sorted(report.items(), key=lambda item: item[1][1], reverse=True)
    sorted_list = [[value[0], key] for key, value in sorted_items]
    
    if sorted_list[0][0] == "sell":     
        for item in sorted_list:
            if item[0] == "buy":
                return [sorted_list[0], item]
        return [sorted_list[0]]
    return [sorted_list[0]]

