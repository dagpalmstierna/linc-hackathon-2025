import time
import multiprocessing
from typing import Dict, Any
import hackathon_linc as lh
import random

lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class DataPoller:
    def __init__(self, func_poll_interval: tuple):
        self.function_name = func_poll_interval[0]
        self.polling_interval = func_poll_interval[1]
        self.process = multiprocessing.Process(target=self._poll_data)
        self.running = multiprocessing.Value('b', True)  # Shared flag to control processes
        self.cache = func_poll_interval[2]
        self.lock = multiprocessing.Lock()

    def _poll_data(self):
        # func = getattr(lh, func_name)
        func_name = self.function_name
        interval = self.polling_interval
        while self.running.value:
            start_time = time.monotonic()
            
            try:
                result = getattr(lh, func_name)()  # Blocking API call
                # result = {}
                # result = random.random()  # Simulate API call
                with self.lock:  # Lock only when updating the cache
                    if isinstance(self.cache, multiprocessing.managers.DictProxy):
                        self.cache.clear()  # Clear the dictionary
                        self.cache.update({"value": result})  # Update with new data
                    elif isinstance(self.cache, multiprocessing.managers.ListProxy):
                        self.cache[:] = []  # Clear the list
                        self.cache.append(result)  # Update with new data
                    else:
                        self.cache.value = result  
                print(f'Polled {func_name}', flush=True)
            except Exception as e:
                print(f'Error polling {func_name}: {e}')
            
            elapsed = time.monotonic() - start_time
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))
            # with self.lock:
            #     time.sleep(max(0.01, sleep_time))

    def start_polling(self):
        if self.running.value:
            self.process.start()

    def stop_polling(self):
        if self.running.value:
            self.running.value = False  # Signal processes to stop
            self.process.join()  # Wait for processes to finish

    def get_cached_data(self):
        with self.lock:  # Lock only when reading the cache
            if isinstance(self.cache, multiprocessing.managers.DictProxy):
                return dict(self.cache)  # Return a copy of the dictionary
            elif isinstance(self.cache, multiprocessing.managers.ListProxy):
                return list(self.cache)  # Return a copy of the list
            else:
                return self.cache.value  # Return the value from the shared Value

if __name__ == "__main__":
    polling_intervals_completed_orders = ("get_completed_orders", 8, multiprocessing.Manager().dict())
    data_poller_completed_orders = DataPoller(polling_intervals_completed_orders)
    data_poller_completed_orders.start_polling()
    time.sleep(1)

    polling_intervals_pending_orders = ("get_pending_orders", 15, multiprocessing.Manager().dict())
    data_poller_pending_orders = DataPoller(polling_intervals_pending_orders)
    data_poller_pending_orders.start_polling()
    time.sleep(1)

    polling_intervals_stoploss_orders = ("get_stoploss_orders", 5, multiprocessing.Manager().dict())
    data_poller_stoploss_orders = DataPoller(polling_intervals_stoploss_orders)
    data_poller_stoploss_orders.start_polling()
    time.sleep(1)

    # polling_intervals_balance = ("get_balance", 11, multiprocessing.Value('d', 0.0))
    # data_poller_balance = DataPoller(polling_intervals_balance)
    # data_poller_balance.start_polling()
    time.sleep(1)

    # polling_intervals_all_tickers = ("get_all_tickers", 5, multiprocessing.Manager().list())
    # data_poller_all_tickers = DataPoller(polling_intervals_all_tickers)
    # data_poller_all_tickers.start_polling()
    # time.sleep(1)

    polling_intervals_all_orders = ("get_all_orders", 12, multiprocessing.Manager().dict())
    data_poller_all_orders = DataPoller(polling_intervals_all_orders)
    data_poller_all_orders.start_polling()
    time.sleep(1)

    polling_intervals_get_portfolio = ("get_portfolio", 0.5, multiprocessing.Manager().dict())
    data_poller_get_portfolio = DataPoller(polling_intervals_get_portfolio)
    data_poller_get_portfolio.start_polling()
    time.sleep(1.2)

    polling_intervals_get_current_price = ("get_current_price", 0.4, multiprocessing.Manager().dict())
    data_poller_get_current_price = DataPoller(polling_intervals_get_current_price)
    data_poller_get_current_price.start_polling()
    time.sleep(1)

    # polling_intervals_get_current_price = ("get_historical_data", 0.2, multiprocessing.Manager().dict())
    # data_poller_get_current_price = DataPoller(polling_intervals_get_current_price)
    # data_poller_get_current_price.start_polling()
    # time.sleep(1)

    

# This is where we would integrate with our main trading logic and user input etc
    try:
        while True:
            cached_data =  data_poller_get_current_price.get_cached_data()
            cached_data =  data_poller_get_portfolio.get_cached_data()
            # print(cached_data)
            print("QUARTER SECOND")
            time.sleep(0.25)
    except KeyboardInterrupt:
        data_poller_completed_orders.stop_polling()
        data_poller_pending_orders.stop_polling()
        data_poller_stoploss_orders.stop_polling()
        data_poller_balance.stop_polling()
        data_poller_all_tickers.stop_polling()
        data_poller_all_orders.stop_polling()
        data_poller_get_portfolio.stop_polling()
        data_poller_get_current_price.stop_polling()