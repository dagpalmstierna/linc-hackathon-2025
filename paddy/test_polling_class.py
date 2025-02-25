import hackathon_linc as lh
import pandas as pd
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading


lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')


class DataPoller:
    def __init__(self, polling_intervals):
        self.polling_intervals = polling_intervals
        self.cache = {}
        self.running = False
        self.paused = threading.Event()
        self.paused.set()  # Start in the unpaused state
        self.executor = None

    def poll_data(self, func, func_name, interval):
        while self.running:
            self.paused.wait()  # Block if paused
            start_time = time.time()
            result = func()
            self.cache[func_name] = result
            elapsed_time = time.time() - start_time
            print('~~~~~CALLED: ' + func_name)
            # print(f"{func_name}: {result} (Time taken: {elapsed_time:.2f} seconds)")
            self.paused.wait(interval)

    def start_polling(self):
        if not self.running:
            self.running = True
            self.executor = ThreadPoolExecutor(max_workers=len(self.polling_intervals))
            for func_name, interval in self.polling_intervals.items():
                func = getattr(lh, func_name)
                self.executor.submit(self.poll_data, func, func_name, interval)

    def stop_polling(self):
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

    def pause_polling(self):
        self.paused.clear()  # Block the polling threads

    def resume_polling(self):
        self.paused.set()  # Unblock the polling threads

    def get_cached_data(self, func_name):
        return self.cache.get(func_name)

def strategy_function(data_poller):
    
    # Access cached data
    all_orders = data_poller.get_cached_data("get_all_orders")
    completed_orders = data_poller.get_cached_data("get_completed_orders")
    pending_orders = data_poller.get_cached_data("get_pending_orders")
    stoploss_orders = data_poller.get_cached_data("get_stoploss_orders")
    balance = data_poller.get_cached_data("get_balance")
    portfolio = data_poller.get_cached_data("get_portfolio")
    all_tickers = data_poller.get_cached_data("get_all_tickers")
    current_price = data_poller.get_cached_data("get_current_price")

    # Implement your strategy logic here
    print(f"Strategy accessing data: {all_orders}, {completed_orders}, {pending_orders}, {stoploss_orders}, {balance}, {portfolio}, {all_tickers}, {current_price}")

    time.sleep(1)

# Example usage
if __name__ == "__main__":
    polling_intervals = {
        "get_all_orders": 5,
        "get_completed_orders": 10,
        "get_pending_orders": 15,
        "get_stoploss_orders": 5,
        "get_balance": 10,
        "get_portfolio": 15,
        "get_all_tickers": 2,
        "get_current_price": 0.001,
    }

    data_poller = DataPoller(polling_intervals)

    try:
        data_poller.start_polling()
        time.sleep(1)
        print(1)
        print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
        time.sleep(2)
        data_poller.pause_polling()
        time.sleep(2)
        strategy_function(data_poller)
        time.sleep(5)
        strategy_function(data_poller)
        print("RESUMING")
        data_poller.resume_polling()
        print("sleeping main thread")
        time.sleep(10)
        # print(2)
        # print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
        # time.sleep(3)
        # print(3)
        # print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
        # time.sleep(4)
        # print(4)
        # print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
        # time.sleep(5)
        # print(5)
        # print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
        # time.sleep(6)
        # print(6)
        # print("FROM main" + str(data_poller.get_cached_data("get_current_price")))
    except KeyboardInterrupt:
        data_poller.stop_polling()


