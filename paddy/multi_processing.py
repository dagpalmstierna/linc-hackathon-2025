import time
import multiprocessing
from typing import Dict, Any
import hackathon_linc as lh
import random

lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class DataPoller:
    def __init__(self, polling_intervals: Dict[str, float]):
        self.polling_intervals = polling_intervals
        self.processes: Dict[str, multiprocessing.Process] = {}
        self.running = multiprocessing.Value('b', True)  # Shared flag to control processes

        # Initialize independent cache and lock for each function
        self.cache = {}  # Stores shared cache variables
        self.locks = {}  # Stores shared locks
        for func_name in polling_intervals.keys():
            # Use multiprocessing.Value for shared cache (supports numbers)
            self.cache[func_name] = multiprocessing.Value('d', 0.0)  # 'd' for double
            # Use multiprocessing.Lock for shared lock
            self.locks[func_name] = multiprocessing.Lock()

    def _poll_data(self, func_name: str, interval: float):
        func = getattr(lh, func_name)
        cache_var = self.cache[func_name]
        lock = self.locks[func_name]

        while self.running.value:
            start_time = time.monotonic()
            
            try:
                result = func()  # Blocking API call
                result = random.random()  # Simulate API call
                with lock:  # Lock only when updating the cache
                    cache_var.value = result  # Update shared cache
                print(f'Polled {func_name}')
            except Exception as e:
                print(f'Error polling {func_name}: {e}')
            
            elapsed = time.monotonic() - start_time
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.15, sleep_time))

    def start_polling(self):
        if self.running.value:
            for func_name, interval in self.polling_intervals.items():
                process = multiprocessing.Process(
                    target=self._poll_data,
                    args=(func_name, interval))
                self.processes[func_name] = process
                process.start()

    def stop_polling(self):
        if self.running.value:
            self.running.value = False  # Signal processes to stop
            for process in self.processes.values():
                process.join()  # Wait for processes to finish

    def get_cached_data(self, func_name: str) -> Any:
        cache_var = self.cache[func_name]
        lock = self.locks[func_name]
        with lock:  # Lock only when reading the cache
            return cache_var.value  # Return the cached value

def strategy_function(data_poller: DataPoller):
    # Access cached data
    data = {func: data_poller.get_cached_data(func) for func in data_poller.polling_intervals}
    print("Strategy data:")

if __name__ == "__main__":
    polling_intervals_slow = {
        "get_completed_orders": 10,
        "get_pending_orders": 15,
        "get_stoploss_orders": 5,
        "get_balance": 10,
        "get_all_tickers": 0.5,
    }

    polling_intervals_fast = {
        "get_all_orders": 10,
        "get_portfolio": 10,
        "get_current_price": 0.5,
    }

    data_poller_slow = DataPoller(polling_intervals_slow)
    data_poller_slow.start_polling()

    data_poller_fast = DataPoller(polling_intervals_fast)
    data_poller_fast.start_polling()

    try:
        while True:
            func_name = random.choice(list(polling_intervals_slow.keys()))
            cached_data = data_poller_slow.get_cached_data(func_name)
            print(f"SLOW: Randomly accessed cache for {func_name}")

            func_name = random.choice(list(polling_intervals_fast.keys()))
            cached_data = data_poller_fast.get_cached_data(func_name)
            print(f"FAST: Randomly accessed cache for {func_name}")
            time.sleep(1)
    except KeyboardInterrupt:
        data_poller_slow.stop_polling()
        data_poller_fast.stop_polling()