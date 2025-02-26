import time
import multiprocessing
import threading
from typing import Dict, Any
import hackathon_linc as lh
from API_polling_working import DataPoller
import pandas as pd

lh.init('265d0b0b-7e97-44a7-9576-47789e8712b2')

class TradingController:
    def __init__(self):
        # Initialize DataPoller instances
        self.data_pollers = {
            "completed_orders": DataPoller(("get_completed_orders", 8, multiprocessing.Manager().dict())),
            "pending_orders": DataPoller(("get_pending_orders", 15, multiprocessing.Manager().dict())),
            "stoploss_orders": DataPoller(("get_stoploss_orders", 5, multiprocessing.Manager().dict())),
            "balance": DataPoller(("get_balance", 11, multiprocessing.Value('d', 0.0))),
            "all_tickers": DataPoller(("get_all_tickers", 5, multiprocessing.Manager().list())),
            "all_orders": DataPoller(("get_all_orders", 12, multiprocessing.Manager().dict())),
            "portfolio": DataPoller(("get_portfolio", 0.1, multiprocessing.Manager().dict())),
            "current_price": DataPoller(("get_current_price", 0.1, multiprocessing.Manager().dict())),
        }

        # Start all pollers
        for poller in self.data_pollers.values():
            poller.start_polling()

        # Flag to control the main loop
        self.running = multiprocessing.Value('b', True)

    def run(self):
        # Start a separate thread for user input
        user_input_thread = threading.Thread(target=self.handle_user_input, daemon=True)
        user_input_thread.start()

        # Main loop
        while self.running.value:
            # Execute trading logic
            time.sleep(1)
            self.execute_trading_logic()

            # Sleep to avoid busy-waiting
            

    def execute_trading_logic(self):
        # Access cached data
        current_price = pd.DataFrame(self.data_pollers["current_price"].get_cached_data()['value']['data']) 
        
        balance = self.data_pollers["balance"].get_cached_data()
        portfolio = self.data_pollers["portfolio"].get_cached_data()

        print(f"Display: {current_price}")

        # Example trading logic
        # if current_price != None and balance != None and portfolio != None:
        #     print(f"Display: {current_price}")
        #     print(f"Balance: {balance}")
        #     print(f"Portfolio: {portfolio}")

            # # Place a bet if conditions are met
            # if current_price["value"] > 100:  # Example condition
            #     self.place_bet(amount=10, asset="BTC")

    def place_bet(self, amount: float, asset: str):
        # Simulate placing a bet
        print(f"Placing a bet: {amount} on {asset}")
        # Call the API to place the bet
        # lh.place_order(asset, amount)

    def handle_user_input(self):
        while self.running.value:
            # Simulate user input (e.g., from a command-line interface)
            command = input("Enter command (start/stop/exit): ").strip().lower()

            if command == "start":
                print("Starting all pollers...")
                for poller in self.data_pollers.values():
                    poller.start_polling()
            elif command == "stop":
                print("Stopping all pollers...")
                for poller in self.data_pollers.values():
                    poller.stop_polling()
            elif command == "exit":
                print("Exiting...")
                self.running.value = False
            else:
                print("Invalid command. Use 'start', 'stop', or 'exit'.")

    def stop(self):
        # Stop all pollers
        for poller in self.data_pollers.values():
            poller.stop_polling()
        self.running.value = False

if __name__ == "__main__":
    controller = TradingController()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.stop()