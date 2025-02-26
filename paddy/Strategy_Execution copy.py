import time
import multiprocessing
from typing import Dict, Any
import hackathon_linc as lh
import random
import requests

# lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class StrategyExecution:
    def __init__(self, func_poll_interval: tuple, strategy):
        self.polling_interval = func_poll_interval
        self.running = multiprocessing.Value('b', True)  # Shared flag to control processes
        self.strategy = strategy
        self.process = multiprocessing.Process(target=self.run_strat)
    
    @staticmethod
    def get_portfolio() -> Dict[str, int]:
        url =  'https://hackathonlincapp.azurewebsites.net/api'+ '/account/portfolio'
        body = {"api_key": "265d0b0b-7e97-44a7-9576-47789e8712b2" } # hard code this to avoid shared resources
        # response = requests.get(url, json=body)
        # response = {}
        return {}#response.json()
    
    @staticmethod
    def get_current_price(ticker: str = None) -> dict:
        gstock_url = 'https://hackathonlincapp.azurewebsites.net/api' + '/data/stocks'
        params = {'ticker': ticker} if ticker else {}
        # response = requests.get(gstock_url, params=params)
        return {} #response.json()

    def run_strat(self):
        # func = getattr(lh, func_name)
        interval = self.polling_interval
        while True:
            start_time = time.monotonic()
            try:
                with multiprocessing.Pool(processes=3) as pool:
                    result_portfolio = pool.apply_async(self.get_portfolio)
                    result_price = pool.apply_async(self.get_current_price)
                    result_historical = pool.apply_async(self.get_historical_data, (25,))
                    
                    result = {
                        "portfolio": result_portfolio.get(),
                        "price": result_price.get(),
                        "historical": result_historical.get()
                    }

                    print(result['portfolio'])
                print(f'Polled {self.strategy}', flush=True)
            except Exception as e:
                print(f'Error polling {self.strategy}: {e}')
            
            elapsed = time.monotonic() - start_time
            print(f"Elapsed time: {elapsed}")
            sleep_time = max(interval - elapsed, 0)
            time.sleep(max(0.01, sleep_time))
                

    def buy(self):
        lh.buy()

    def sell(self):
        lh.sell()

    def start(self):
        if self.running.value:
            self.process.start()

    def stop(self):
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
    def test(historical_data):
        return {'buy':('STOCK1', 10), 'sell':('STOCK2', 5)}
    
    strat_test = StrategyExecution(0.5, test)
    strat_test.start()
    time.sleep(1)


    # time.sleep(1)

    

# This is where we would integrate with our main trading logic and user input etc
    try:
        while True:
            # print(cached_data)
            print("QUARTER SECOND")
            time.sleep(0.25)
    except KeyboardInterrupt:
        strat_test.stop_polling()