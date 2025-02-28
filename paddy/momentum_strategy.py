import pandas as pd
import talib as ta
import hackathon_linc as lh
import numpy as np

#bollinger bands kanske
#tweeka lokback, lower_pct, upper_pct etc. 

def dynamic_rsi_thresholds(rsi_series, lower_pct=25, upper_pct=75, lookback=100):
    """
    Computes dynamic RSI thresholds using the specified percentiles over a lookback period.

    Parameters:
    -----------
    rsi_series : pd.Series
        The RSI values for a particular stock (indexed by datetime).
    lower_pct : int
        Lower percentile for the RSI threshold.
    upper_pct : int
        Upper percentile for the RSI threshold.
    lookback : int
        Number of latest data points to consider for computing thresholds 
        (e.g., last 200 hours).

    Returns:
    --------
    lower_threshold, upper_threshold : float, float
    """
    # Drop NaNs that might appear from RSI calculation.
    valid_rsi = rsi_series.dropna()
    
    # If we don't have enough data, just return default or safe values.
    if len(valid_rsi) < lookback:
        windowed_rsi = valid_rsi
    else:
        # Take the last 'lookback' points
        windowed_rsi = valid_rsi.iloc[-lookback:]
    
    # Compute the dynamic thresholds
    lower_threshold = np.percentile(windowed_rsi, lower_pct)
    upper_threshold = np.percentile(windowed_rsi, upper_pct)
    
    return lower_threshold, upper_threshold


def momentum_strategy(historical_data):
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

    df["rsi"] = ta.RSI(df["price"], timeperiod=14)
    

    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=400, # 24, 52, 18
        slowperiod=1600,
        signalperiod=72
    )

    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist  

    stock_dfs = {}
    for symbol in df["symbol"].unique():
        df_stock = df[df["symbol"] == symbol].copy()
        stock_dfs[symbol] = df_stock
    
    recommendations = {}
    
    for symbol, df_stock in stock_dfs.items():
        # 1) Compute dynamic RSI thresholds
        lower_threshold, upper_threshold = dynamic_rsi_thresholds(
            df_stock["rsi"], 
            lower_pct=25,  
            upper_pct=85, 
            lookback=100
        )

        latest_row = df_stock.iloc[-1]
        latest_rsi = latest_row["rsi"]
        macd_val   = latest_row["macd_line"]
        signal_val = latest_row["macd_signal"]
        mid_point = (upper_threshold + lower_threshold) / 2
        
        if macd_val > signal_val and latest_rsi < lower_threshold:
            # 'Buy' signal
            recommendations[symbol] = ['buy', np.abs(latest_rsi - mid_point)]
        elif macd_val < signal_val and latest_rsi > upper_threshold:
            # 'Sell' signal
        
            recommendations[symbol] = ['sell', np.abs(latest_rsi - mid_point)]
        
    return recommendation(recommendations)

# def rsi_strategy(historical_data):
    
#     df = pd.DataFrame(historical_data)
#     df.index = pd.to_datetime(df["gmtTime"])
#     df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

#     df["rsi"] = ta.RSI(df["price"], timeperiod=14)
    
#     macd_line, macd_signal, macd_hist = ta.MACD(
#         df["price"],
#         fastperiod=12,
#         slowperiod=26,
#         signalperiod=9
#     )

#     stock_dfs = {}

#     for symbol in df["symbol"].unique():
#         df_stock = df[df["symbol"] == symbol].copy()  
#         stock_dfs[symbol] = df_stock
    
#     report = {}
    
#     for key in stock_dfs.keys():

#         current_date = stock_dfs[key].iloc[-1]
#         rsi_value = current_date["rsi"]
        
#         if rsi_value < 45:
#             report[key] = ['buy', np.abs(rsi_value-50)]

#         elif rsi_value > 55:
#             report[key] = ['sell', np.abs(rsi_value-50)]
#     return recommendation(report)

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


def backtest_dynamic_rsi(historical_data, rsi_lookback=14, dyna_lookback=100,
                         lower_pct=25, upper_pct=85):
    """
    Example backtest that uses:
    1) MACD for direction.
    2) A rolling dynamic RSI threshold for overbought/oversold.
    
    - rsi_lookback: Period for TA-Lib's RSI.
    - dyna_lookback: Number of bars to look back for computing dynamic thresholds.
    - lower_pct, upper_pct: RSI percentiles to define oversold/overbought levels.
    """
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    # Mid-price
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

    # Indicators
    df["rsi"] = ta.RSI(df["price"], timeperiod=rsi_lookback)
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=400, #hourly #try slower: 24, 52, 18
        slowperiod=1600,
        signalperiod=72
    )
    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist

    # Split by symbol
    stock_dfs = {}
    for symbol in df["symbol"].unique():
        df_stock = df[df["symbol"] == symbol].copy()
        df_stock.sort_index(inplace=True)
        stock_dfs[symbol] = df_stock

    report = {}

    for symbol, df_stock in stock_dfs.items():
        position = 0    # 0 = not in a position, 1 = long
        trades = []

        # We'll iterate through each bar in chronological order
        # and compute a dynamic threshold from the *prior* bars.
        for i in range(len(df_stock)):
            if i < dyna_lookback:
                # Skip until we have enough bars to do the rolling window
                # You could also do something smaller or just skip
                continue
            
            # Slice the RSI up to (but not including) the current bar
            past_rsi = df_stock["rsi"].iloc[i-dyna_lookback:i]
            lower_thr, upper_thr = dynamic_rsi_thresholds(
                past_rsi,
                lower_pct=lower_pct,
                upper_pct=upper_pct,
                lookback=dyna_lookback
            )

            # Current bar's data
            row = df_stock.iloc[i]
            time = row.name  # since df_stock is indexed by datetime
            rsi_value = row["rsi"]
            macd_val  = row["macd_line"]
            signal_val= row["macd_signal"]
            price     = row["price"]

            # Skip if RSI or MACD is NaN
            if pd.isna(rsi_value) or pd.isna(macd_val) or pd.isna(signal_val):
                continue

            # Buy condition: not in position, MACD bullish, RSI < dynamic lower
            if position == 0 and macd_val > signal_val and rsi_value < lower_thr:
                position = 1
                trades.append({"time": time, "action": "buy", "price": price})

            # Sell condition: in position, MACD bearish, RSI > dynamic upper
            elif position == 1 and macd_val < signal_val and rsi_value > upper_thr:
                position = 0
                trades.append({"time": time, "action": "sell", "price": price})

        # If we end in a long position, exit at final bar
        if position == 1:
            last_price = df_stock.iloc[-1]["price"]
            last_time  = df_stock.index[-1]
            trades.append({"time": last_time, "action": "sell", "price": last_price})
            position = 0
        
        # Calculate PnL from the trade list
        pnl = 0.0
        trade_count = 0
        for t in range(0, len(trades)-1, 2):
            buy_trade  = trades[t]
            sell_trade = trades[t+1]
            trade_pnl  = sell_trade["price"] - buy_trade["price"]
            pnl += trade_pnl
            trade_count += 1
        
        report[symbol] = {
            "stock": symbol,
            "pnl": pnl,
            "trade_count": trade_count,
        }
    
    return report

# lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')
# historical_data = lh.get_historical_data(365)
# kv = backtest_dynamic_rsi(historical_data=historical_data)
# for k,v in kv.items():
#     print(f"{k}: {v}")