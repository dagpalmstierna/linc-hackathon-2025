import time
import multiprocessing
from typing import Dict, Any, List
import hackathon_linc as lh
import random
import requests
import pandas as pd
import platform
from momentum_strategy import momentum_strategy, rsi_strategy
import traceback
# lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class Strategy:
    def __init__(self, func_poll_interval: tuple, strategy, data_source, order_manager=None, starting_balance=100000):
        self.polling_interval = func_poll_interval
        self.running = multiprocessing.Value('b', True)  # Shared flag to control processes
        self.process = multiprocessing.Process(target=self.run_strat)
        self.lock = multiprocessing.Lock()
        self.cache = multiprocessing.Manager().dict()
        self.strategy_func = strategy
        self.data_source = data_source
        time.sleep(1)
        all_tickers = self.get_all_tickers()
        self.portfolio = {ticker: 0 for ticker in all_tickers}
        self.order_manager = order_manager
        self.balance = starting_balance
    
    @staticmethod
    def get_portfolio() -> Dict[str, int]:
        url =  'https://hackathonlincapp.azurewebsites.net/api'+ '/account/portfolio'
        body = {"api_key": "265d0b0b-7e97-44a7-9576-47789e8712b2" } # hard code this to avoid shared resources
        response = requests.get(url, json=body)
        return response.json()
    
    @staticmethod
    def get_all_tickers() -> List[str]:

        ticker_url = 'https://hackathonlincapp.azurewebsites.net/api'+'/symbols'
        response = requests.get(ticker_url)
        response_json = response.json()

        return response_json
    
    @staticmethod
    def get_current_price(ticker: str = None) -> dict:
        gstock_url = 'https://hackathonlincapp.azurewebsites.net/api' + '/data/stocks'
        params = {'ticker': ticker} if ticker else {}
        response = requests.get(gstock_url, params=params)
        return response.json()

    def run_strat(self):
        # func = getattr(lh, func_name)
        interval = self.polling_interval
        while self.running.value:
            start_time = time.monotonic()
            try:
                print('PORTFOLIO', self.portfolio)
                historical_data = self.data_source.get_cached_data()
                strat_response = self.strategy_func(historical_data)
                print(strat_response)
                if len(strat_response) > 0:
                    current_prices = {ticker:{'ask': 50, 'bid': 50} for ticker in self.portfolio.keys()} # need to make calculate this from historical data
                    order_to_execute = self.order_manager(strat_response, self.balance, self.portfolio, current_prices)

                    if order_to_execute != None:
                        if order_to_execute[0] == 'buy':
                            self.buy(order_to_execute[1], order_to_execute[2])
                        if order_to_execute[0] == 'sell':
                            self.sell(order_to_execute[1], order_to_execute[2])
                 
                # print(result_historical)
                # print(f'Polled', flush=True)
            except Exception as e:
                print(f'Error polling: {e}')
                traceback.print_exc()
            
            elapsed = time.monotonic() - start_time
            # print(f"Elapsed time: {elapsed}")
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))
    
    def buy(self, ticker, amount, price=None, days_to_cancel=1):
        params = {'type': 'buy',
              'ticker': ticker,
              'amount': amount,
              'days_to_cancel': days_to_cancel}
        body = {'api_key': "265d0b0b-7e97-44a7-9576-47789e8712b2"}

        if price is not None:
            params['price'] = price

        url_s = 'https://hackathonlincapp.azurewebsites.net/api' + '/order/'

        response = requests.post(url_s, params=params, json=body)
        response_json = response.json()
        print(response_json)
        if 'order_status' in response_json and response_json['order_status'] == 'completed':
            price = response_json['price']
            amount = response_json['amount']
            self.balance -= amount * price
            self.portfolio[ticker] += amount
            print(f"Bought {amount} of {ticker} at {price}")
            return response_json['order_status']
        return None
    
    def sell(self, ticker, amount, price=None, days_to_cancel=1):
        params = {'type': 'sell',
              'ticker': ticker,
              'amount': amount,
              'days_to_cancel': days_to_cancel}
        body = {'api_key': "265d0b0b-7e97-44a7-9576-47789e8712b2"}

        if price is not None:
            params['price'] = price

        url_s = 'https://hackathonlincapp.azurewebsites.net/api' + '/order/'

        response = requests.post(url_s, params=params, json=body)
        response_json = response.json()
        print(response_json)

        if 'order_status' in response_json and response_json['order_status'] == 'completed':
            price = response_json['price']
            amount = response_json['amount']
            self.balance += amount * price
            self.portfolio[ticker] -= amount
            print(f"Sold {amount} of {ticker} at {price}")
            return response_json['order_status']
        return None

    def start(self):
        if self.running.value:
            self.process.start()
    
    def pause(self):
        if self.running_value:
            self.running_value = False
    
    def resume(self):
        if not self.running_value:
            self.running_value = True

    def stop(self):
        if self.running.value:
            self.running.value = False  # Signal processes to stop
            self.process.join()  # Wait for processes to finish

    def get_cached_data(self):
        with self.lock:  # Lock only when reading the cache
            return dict(self.cache)


class DataCollection:
    def __init__(self, func_poll_interval: tuple):
        self.polling_interval = func_poll_interval
        self.running = multiprocessing.Value('b', True)  # Shared flag to control processes
        self.process = multiprocessing.Process(target=self.run_data_collect)
        self.lock = multiprocessing.Lock()
        self.cache = multiprocessing.Manager().list()
    
    @staticmethod
    def get_historical_data(days_back: int, ticker: str = None) -> dict:
        if days_back < 0 or days_back > 365:
            raise ValueError("""
            You have entered a negative value for days back, it must be psotive.
            """)

        params = {'days_back': days_back}
        if ticker:
            params['ticker'] = ticker
        body = {"api_key": "265d0b0b-7e97-44a7-9576-47789e8712b2" } # hard code this to avoid shared resources
        response = requests.get('https://hackathonlincapp.azurewebsites.net/api' + '/data', params=params, json=body)

        return response.json()
    
    def process_data(self, historical_data):
        
        # time.sleep(100)
        df = pd.DataFrame(historical_data)
        # print(df)
        df["gmtTime"] = pd.to_datetime(df["gmtTime"])

        # Dictionary to store processed data for each stock
        stock_dfs = {}

        # Feature engineering for each stock
        for symbol in df["symbol"].unique():
            df_stock = df[df["symbol"] == symbol].copy()

            # Round numerical columns
            cols_to_round = [col for col in df_stock.columns if col not in ["gmtTime", "symbol"]]
            df_stock[cols_to_round] = df_stock[cols_to_round].round(2)

            # Time-based features
            df_stock['hour'] = df_stock['gmtTime'].dt.hour
            df_stock['day_of_week'] = df_stock['gmtTime'].dt.dayofweek

            # Rolling statistics
            df_stock['askMedian_rolling_mean_3h'] = df_stock['askMedian'].rolling(window=3, min_periods=1).mean()
            df_stock['bidMedian_rolling_mean_3h'] = df_stock['bidMedian'].rolling(window=3, min_periods=1).mean()
            df_stock['askMedian_rolling_std_3h'] = df_stock['askMedian'].rolling(window=3, min_periods=1).std()
            df_stock['bidMedian_rolling_std_3h'] = df_stock['bidMedian'].rolling(window=3, min_periods=1).std()

            # Percentage changes
            df_stock['askMedian_pct_change'] = df_stock['askMedian'].pct_change()
            df_stock['bidMedian_pct_change'] = df_stock['bidMedian'].pct_change()

            # Spread-related features
            df_stock['spread_ratio'] = df_stock['spreadMedian'] / (df_stock['askMedian'] + df_stock['bidMedian'])
            # df_stock['spread_pct_change'] = df_stock['spreadMedian'].pct_change()

            # Volume-related features
            df_stock['askVolume_relative'] = df_stock['askVolume'] / df_stock['askVolume'].rolling(window=5, min_periods=1).mean()
            df_stock['bidVolume_relative'] = df_stock['bidVolume'] / df_stock['bidVolume'].rolling(window=5, min_periods=1).mean()
            df_stock['volume_imbalance'] = (df_stock['askVolume'] - df_stock['bidVolume']) / (df_stock['askVolume'] + df_stock['bidVolume'])

            # Lagged features (e.g., previous hour's values)
            for lag in range(1, 25):  # Add lags for the last 3 hours
                df_stock[f'askMedian_lag_{lag}'] = df_stock['askMedian'].shift(lag)
                df_stock[f'bidMedian_lag_{lag}'] = df_stock['bidMedian'].shift(lag)
                df_stock[f'spreadMedian_lag_{lag}'] = df_stock['spreadMedian'].shift(lag)

            # Target variable: Direction of price movement (1 if bidMedian increases next hour, 0 otherwise)
            df_stock['target'] = (df_stock['bidMedian'].shift(-5) > df_stock['bidMedian']).astype(int)

            # Drop rows with missing values (due to lags and rolling features)
            df_stock = df_stock.dropna()

            # Store processed dataframe
            stock_dfs[symbol] = df_stock
            # print('FROM processing',df_stock)
        return stock_dfs

    def run_data_collect(self):
        # func = getattr(lh, func_name)
        print("HERE")
        interval = self.polling_interval
        while self.running.value:
            start_time = time.monotonic()
            try:
                historical_data = self.get_historical_data(100)
               # print(historical_data)
                historical_processed = historical_data #self.process_data(historical_data)
                with self.lock:  # Lock only when updating the cache
                    # self.cache.clear()  # Clear the dictionary
                    # self.cache.update(historical_processed)
                    self.cache[:] = []  # Clear the list
                    self.cache[:] = historical_processed
                 
                # print(result_historical)
                print(f'Polled', flush=True)
            except Exception as e:
                print(f'Error polling: {e}')
            
            elapsed = time.monotonic() - start_time
            print(f"Elapsed time: {elapsed}")
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))

    def start(self):
        if self.running.value:
            self.process.start()
    
    def pause(self):
        if self.running_value:
            self.running_value = False
    
    def resume(self):
        if not self.running_value:
            self.running_value = True

    def stop(self):
        if self.running.value:
            self.running.value = False  # Signal processes to stop
            self.process.join()  # Wait for processes to finish

    def get_cached_data(self):
        with self.lock:  # Lock only when reading the cache
            return list(self.cache)
            # return dict(self.cache)

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork")

    def test(historical_data):
        buy_sell = random.choice(['buy', 'sell'])
        stock = random.choice(['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5', 'STOCK6', 'STOCK7', 'STOCK8', 'STOCK9', 'STOCK10'])
        return random.choice([[(buy_sell, stock)], [], [], [], []])
    

    def order_manager_example(strategy_response, balance, portfolio, current_price):
        for action, ticker in strategy_response:
            if action == 'sell' and portfolio[ticker] > 0:
                amount_stock_we_have = portfolio[ticker]
                return ('sell', ticker, amount_stock_we_have)
            elif action == 'buy':
                print("TRIED BUYING")
                curr_stock_price = current_price[ticker]['ask']
                amount = int((balance*0.01) // curr_stock_price)
                # print('BUY AMOUNT', amount)
                return ('buy', ticker, amount)
            elif action == 'sell':
                print('SELL but had no stock')
        return None
    
    
    data_collect = DataCollection(0.7)
    data_collect.start()
    time.sleep(1)

    # strat = Strategy(1, test, data_collect, order_manager_example)
    # strat.start()

    strat2 = Strategy(1, rsi_strategy, data_collect, order_manager_example)
    strat2.start()
    

    # time.sleep(0.75)

    # strat2 = Strategy(0.5, test, data_collect)
    # strat2.start()

    # time.sleep(1)

    try:
        while True:
            # print(cached_data)
            # cached_data = data_collect.get_cached_data()
            # print(cached_data)
            print("~~~~~~ONE SECOND~~~~~~")
            time.sleep(1)
    except KeyboardInterrupt:
        data_collect.stop()
       # strat.stop()
        strat2.stop()
