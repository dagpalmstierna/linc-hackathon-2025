import pandas as pd
import talib as ta
import hackathon_linc as lh
import numpy as np


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
    valid_rsi = rsi_series.dropna()

    if len(valid_rsi) < lookback:
        windowed_rsi = valid_rsi
    else:
        windowed_rsi = valid_rsi.iloc[-lookback:]

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
        fastperiod=400,
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
            recommendations[symbol] = ['buy', np.abs(latest_rsi - mid_point)]
        elif macd_val < signal_val and latest_rsi > upper_threshold:
            recommendations[symbol] = ['sell', np.abs(latest_rsi - mid_point)]

    return recommendation(recommendations)


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
    Backtests a combined MACD + dynamic RSI strategy.

    - rsi_lookback: Period for TA-Lib's RSI.
    - dyna_lookback: Number of bars to look back for computing dynamic thresholds.
    - lower_pct, upper_pct: RSI percentiles to define oversold/overbought levels.
    """
    df = pd.DataFrame(historical_data)
    df.index = pd.to_datetime(df["gmtTime"])
    df["price"] = (df["askMedian"] + df["bidMedian"]) / 2

    df["rsi"] = ta.RSI(df["price"], timeperiod=rsi_lookback)
    macd_line, macd_signal, macd_hist = ta.MACD(
        df["price"],
        fastperiod=400,
        slowperiod=1600,
        signalperiod=72
    )
    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist

    stock_dfs = {}
    for symbol in df["symbol"].unique():
        df_stock = df[df["symbol"] == symbol].copy()
        df_stock.sort_index(inplace=True)
        stock_dfs[symbol] = df_stock

    report = {}

    for symbol, df_stock in stock_dfs.items():
        position = 0
        trades = []

        for i in range(len(df_stock)):
            if i < dyna_lookback:
                continue

            past_rsi = df_stock["rsi"].iloc[i-dyna_lookback:i]
            lower_thr, upper_thr = dynamic_rsi_thresholds(
                past_rsi,
                lower_pct=lower_pct,
                upper_pct=upper_pct,
                lookback=dyna_lookback
            )

            row = df_stock.iloc[i]
            time = row.name
            rsi_value = row["rsi"]
            macd_val  = row["macd_line"]
            signal_val= row["macd_signal"]
            price     = row["price"]

            if pd.isna(rsi_value) or pd.isna(macd_val) or pd.isna(signal_val):
                continue

            if position == 0 and macd_val > signal_val and rsi_value < lower_thr:
                position = 1
                trades.append({"time": time, "action": "buy", "price": price})

            elif position == 1 and macd_val < signal_val and rsi_value > upper_thr:
                position = 0
                trades.append({"time": time, "action": "sell", "price": price})

        if position == 1:
            last_price = df_stock.iloc[-1]["price"]
            last_time  = df_stock.index[-1]
            trades.append({"time": last_time, "action": "sell", "price": last_price})
            position = 0

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
