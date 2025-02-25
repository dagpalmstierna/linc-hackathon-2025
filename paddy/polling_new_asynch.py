import time
import threading
from typing import Dict, Any
import hackathon_linc as lh

lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class DataPoller:
    def __init__(self, polling_intervals: Dict[str, float]):
        self.polling_intervals = polling_intervals
        self.cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False

    def _poll_data(self, func_name: str, interval: float):
        func = getattr(lh, func_name)
        while self.running:
            start_time = time.monotonic()
            
            try:
                result = func()  # Call the blocking API function
                print(f'Polled {func_name}')
                with self.cache_lock:
                    self.cache[func_name] = result
                # print(f'Polled {func_name}: {result}')
                # print(f'Polled {func_name}')
            except Exception as e:
                print(f'Error polling {func_name}: {e}')
            
            elapsed = time.monotonic() - start_time
            sleep_time = max(interval - elapsed, 0)
            time.sleep(sleep_time)

    def start_polling(self):
        if not self.running:
            self.running = True
            for func_name, interval in self.polling_intervals.items():
                thread = threading.Thread(
                    target=self._poll_data,
                    args=(func_name, interval),
                    daemon=True
                )
                self.threads[func_name] = thread
                thread.start()

    def stop_polling(self):
        if self.running:
            self.running = False
            for thread in self.threads.values():
                thread.join()

    def get_cached_data(self, func_name: str) -> Any:
        with self.cache_lock:
            return self.cache.get(func_name)

def strategy_function(data_poller: DataPoller):
    # Access cached data
    data = {func: data_poller.get_cached_data(func) for func in data_poller.polling_intervals}
    print("Strategy data:", data)

if __name__ == "__main__":
    polling_intervals = {
        "get_all_orders": 5,
        "get_completed_orders": 10,
        "get_pending_orders": 15,
        "get_stoploss_orders": 5,
        "get_balance": 10,
        "get_portfolio": 15,
        "get_all_tickers": 2,
        "get_current_price": 0.2,
    }

    data_poller = DataPoller(polling_intervals)
    data_poller.start_polling()

    try:
        while True:
            # strategy_function(data_poller)
            time.sleep(1)
    except KeyboardInterrupt:
        data_poller.stop_polling()