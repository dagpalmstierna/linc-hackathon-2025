import time
from io import TextIOWrapper
from threading import Event, Thread
import multiprocessing
import hackathon_linc as lh

import sys
import os

# Add the parent directory (where both 'eliot' and 'paddy' are located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import the module from paddy
from paddy.trading_controller import TradingController


def index_live(capital: float, should_stop: Event, controller: TradingController):
    """
    Live index strategy:
      - Buys every available stock with equal capital allocation.
      - Waits until a stop signal is received.
      - Sells all holdings and logs the net result.
    """
    start_capital = capital
    with open("./logs/index_logs_live.txt", "a") as logs:
        logs.write("Starting live index strategy\n")
        portfolio, capital_left = buy_all(capital, logs, controller)
        logs.write(f"Portfolio holdings after buys: {portfolio}\n")
        
        # Hold positions until a stop signal is received.
        while not should_stop.is_set():
            time.sleep(1)
        
        capital_net = sell_all(capital_left, logs, controller, portfolio)
        result = capital_net - start_capital
        logs.write(f"Net result: {result}\n\n")

def buy_all(capital: float, logs: TextIOWrapper, controller: TradingController):
    """
    Buys all available stocks from the API with equal allocation.
    It obtains the list of tickers and current prices from the live data pollers.
    """
    # Retrieve the list of available tickers.
    tickers = controller.data_pollers["all_tickers"].get_cached_data()
    if not tickers:
        logs.write("No tickers available from API.\n")
        return {}, capital

    # Get current prices.
    current_price_data = controller.data_pollers["current_price"].get_cached_data()
    # The current price poller returns a dict with a "value" key.
    if "value" in current_price_data:
        current_price_data = current_price_data["value"]

    n_stocks = len(tickers)
    allocation = capital / n_stocks
    portfolio = {}
    for ticker in tickers:
        price = current_price_data.get(ticker)
        if price is None:
            logs.write(f"Price for ticker {ticker} not available, skipping.\n")
            continue
        shares = int(allocation // price)
        if shares <= 0:
            logs.write(f"Not enough allocation for {ticker} at price {price}, skipping.\n")
            continue

        # Place a buy order using the new API call (lh.buy).
        # The returned dict is expected to contain "amount" and "price".
        buy_result = lh.buy(ticker, shares)
        executed_shares = buy_result.get("amount", shares)
        executed_price = buy_result.get("price", price)
        portfolio[ticker] = executed_shares
        cost = executed_shares * executed_price
        logs.write(f"Bought {executed_shares} of {ticker} at {executed_price}\n")
        capital -= cost

    logs.write(f"Capital remaining after buys: {capital}\n")
    return portfolio, capital

def sell_all(capital: float, logs: TextIOWrapper, controller: TradingController, portfolio: dict):
    """
    Sells all stocks in the portfolio using the API.
    """
    logs.write("Selling all stocks...\n")
    for ticker, shares in portfolio.items():
        sell_result = lh.sell(ticker, shares)
        executed_shares = sell_result.get("amount", shares)
        executed_price = sell_result.get("price", 0)
        logs.write(f"Sold {executed_shares} of {ticker} at {executed_price}\n")
        capital += executed_shares * executed_price
    return capital

def main():
    # Instantiate the new API controller which automatically starts the data pollers.
    controller = TradingController()
    
    # Create an Event to signal when to stop the strategy.
    stop_event = Event()
    
    # Run the live index strategy in a separate thread.
    strategy_thread = Thread(target=index_live, args=(1000000, stop_event, controller))
    strategy_thread.start()
    
    try:
        # Let the strategy run live for a set period (e.g. 60 seconds) or until an external signal.
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        strategy_thread.join()
        controller.stop()  # Ensure all data pollers are stopped.

if __name__ == "__main__":
    main()
