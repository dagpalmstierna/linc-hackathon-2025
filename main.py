import time
import multiprocessing
from typing import Dict, Any, List
import hackathon_linc as lh
import random
import requests
import pandas as pd
import platform
import traceback
from strategies.momentum_strategy import momentum_strategy


class Strategy:
    def __init__(self, func_poll_interval: tuple, strategy, data_source, order_manager=None, starting_balance=100000):
        self.polling_interval = func_poll_interval
        self.running = multiprocessing.Value('b', True)
        self.process = multiprocessing.Process(target=self.run_strat)
        self.lock = multiprocessing.Lock()
        self.cache = multiprocessing.Manager().dict()
        self.strategy_func = strategy
        self.data_source = data_source
        time.sleep(1)
        all_tickers = self.get_all_tickers()
        self.portfolio = {ticker: 0 for ticker in all_tickers}
        self.open_positions = {}
        self.order_manager = order_manager
        self.balance = starting_balance

    @staticmethod
    def get_portfolio() -> Dict[str, int]:
        url = 'https://hackathonlincapp.azurewebsites.net/api' + '/account/portfolio'
        body = {"api_key": "5d5249da-7e7e-438b-a392-692f71364fd0"}
        response = requests.get(url, json=body)
        return response.json()

    @staticmethod
    def get_all_tickers() -> List[str]:
        ticker_url = 'https://hackathonlincapp.azurewebsites.net/api' + '/symbols'
        response = requests.get(ticker_url)
        return response.json()

    @staticmethod
    def get_current_price(ticker: str = None) -> dict:
        gstock_url = 'https://hackathonlincapp.azurewebsites.net/api' + '/data/stocks'
        params = {'ticker': ticker} if ticker else {}
        response = requests.get(gstock_url, params=params)
        return response.json()

    def run_strat(self):
        interval = self.polling_interval
        while self.running.value:
            start_time = time.monotonic()
            try:
                historical_data = self.data_source.get_cached_data()
                current_prices = {stock_info["symbol"]: {'ask': stock_info["askMedian"], "bid": stock_info["bidMedian"]}
                                  for stock_info in historical_data[:-len(self.portfolio.keys())]}
                self.check_stops(current_prices)
                strat_response = self.strategy_func(historical_data)
                print('STRAT RESPONSE', strat_response)

                if len(strat_response) > 0:
                    order_to_execute = self.order_manager(strat_response, self.balance, self.portfolio, current_prices)

                    if order_to_execute is not None:
                        if order_to_execute[0] == 'buy':
                            self.buy(order_to_execute[1], order_to_execute[2])
                        if order_to_execute[0] == 'sell':
                            self.sell(order_to_execute[1], order_to_execute[2])

            except Exception as e:
                print(f'Error polling: {e}')
                traceback.print_exc()

            elapsed = time.monotonic() - start_time
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))

    def buy(self, ticker, amount, price=None, days_to_cancel=1):
        amount = max(amount, 1)
        params = {'type': 'buy', 'ticker': ticker, 'amount': amount, 'days_to_cancel': days_to_cancel}
        body = {'api_key': "5d5249da-7e7e-438b-a392-692f71364fd0"}

        if price is not None:
            params['price'] = price

        url_s = 'https://hackathonlincapp.azurewebsites.net/api' + '/order/'
        print("TRIED BUYING", ticker)
        response = requests.post(url_s, params=params, json=body)
        response_json = response.json()
        print(response_json)

        if 'order_status' in response_json and response_json['order_status'] == 'completed':
            price = response_json['price']
            amount = response_json['amount']
            self.balance -= amount * price
            self.portfolio[ticker] += amount

            if ticker in self.open_positions:
                pos = self.open_positions[ticker]
                pos["cost"] += amount * price
                pos["shares"] += amount
            else:
                self.open_positions[ticker] = {
                    "shares": amount,
                    "cost": amount * price,
                }

            print(f"Bought {amount} of {ticker} at {price}")
            return response_json['order_status']
        return None

    def sell(self, ticker, amount, price=None, days_to_cancel=1):
        params = {'type': 'sell', 'ticker': ticker, 'amount': amount, 'days_to_cancel': days_to_cancel}
        body = {'api_key': "5d5249da-7e7e-438b-a392-692f71364fd0"}

        if price is not None:
            params['price'] = price

        url_s = 'https://hackathonlincapp.azurewebsites.net/api' + '/order/'
        response = requests.post(url_s, params=params, json=body)
        response_json = response.json()
        print("sell response", response_json)

        if 'order_status' in response_json and response_json['order_status'] == 'completed':
            price = response_json['price']
            amount = response_json['amount']
            self.balance += amount * price
            self.portfolio[ticker] -= amount
            del self.open_positions[ticker]
            print(f"Sold {amount} of {ticker} at {price}")
            return response_json['order_status']
        return None

    def check_stops(self, price_data):
        for ticker, pos in self.open_positions.items():
            shares = pos["shares"]
            if shares <= 0:
                continue
            current_price = price_data[ticker]['bid']
            current_return = (current_price * pos["shares"] / pos["cost"]) * 100

            if current_return is not None and (current_return < -12.5 or current_return > 12.5):
                print(f"STOP LOSS triggered for {ticker}! Price = {current_price:.2f}")
                self_resp = self.sell(ticker, shares)
                print(self_resp)

    def start(self):
        if self.running.value:
            self.process.start()

    def stop(self):
        if self.running.value:
            self.running.value = False
            self.process.join()

    def get_cached_data(self):
        with self.lock:
            return dict(self.cache)


class DataCollection:
    def __init__(self, func_poll_interval: tuple):
        self.polling_interval = func_poll_interval
        self.running = multiprocessing.Value('b', True)
        self.process = multiprocessing.Process(target=self.run_data_collect)
        self.lock = multiprocessing.Lock()
        self.cache = multiprocessing.Manager().list()

    @staticmethod
    def get_historical_data(days_back: int, ticker: str = None) -> dict:
        if days_back < 0 or days_back > 365:
            raise ValueError("days_back must be between 0 and 365.")

        params = {'days_back': days_back}
        if ticker:
            params['ticker'] = ticker
        body = {"api_key": "5d5249da-7e7e-438b-a392-692f71364fd0"}
        response = requests.get('https://hackathonlincapp.azurewebsites.net/api' + '/data', params=params, json=body)
        return response.json()

    def run_data_collect(self):
        print("Data collection started")
        interval = self.polling_interval
        while self.running.value:
            start_time = time.monotonic()
            try:
                historical_data = self.get_historical_data(100)
                with self.lock:
                    self.cache[:] = []
                    self.cache[:] = historical_data
                print('Polled', flush=True)
            except Exception as e:
                print(f'Error polling: {e}')

            elapsed = time.monotonic() - start_time
            print(f"Elapsed time: {elapsed}")
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))

    def start(self):
        if self.running.value:
            self.process.start()

    def stop(self):
        if self.running.value:
            self.running.value = False
            self.process.join()

    def get_cached_data(self):
        with self.lock:
            return list(self.cache)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork")

    def order_manager(strategy_response, balance, portfolio, current_price):
        for action, ticker in strategy_response:
            if action == 'sell' and portfolio[ticker] > 0:
                amount_stock_we_have = portfolio[ticker]
                return ('sell', ticker, amount_stock_we_have)
            elif action == 'buy':
                curr_stock_price = current_price[ticker]['ask']
                amount = int(1000 // curr_stock_price)
                return ('buy', ticker, amount)
            elif action == 'sell':
                print('SELL but had no stock')
        return None

    data_collect = DataCollection(0.7)
    data_collect.start()
    time.sleep(1)

    strat = Strategy(1, momentum_strategy, data_collect, order_manager, starting_balance=10000)
    strat.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        data_collect.stop()
        strat.stop()
