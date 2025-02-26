from paddy.API_polling_working import DataPoller
import pandas as pd
import numpy as np



def calculate_moving_averages(prices: list, short_window: int = 10, long_window: int = 50):

    prices_series = pd.Series(prices)

    short_ma = prices_series.rolling(window=short_window).mean()
    long_ma = prices_series.rolling(window=long_window).mean()

    return short_ma, long_ma

def execute_moving_average_strategy(current_prices: list, short_window: int = 10, long_window: int = 50):

    short_ma, long_ma = calculate_moving_averages(current_prices, short_window, long_window)
    if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
        return "buy"
    elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
        return "sell"
    else:
        return "hold"
