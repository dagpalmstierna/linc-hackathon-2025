import time
import multiprocessing
import sys
import os
import requests
from threading import Event

import hackathon_linc as lh

# Add the parent directory (where both 'eliot' and 'paddy' are located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the new API wrapper classes from paddy.
from paddy.Strategy_Execution import Strategy, DataCollection


def create_index_strategy_func(shared_portfolio, starting_capital=1000000):
    """
    Returns a strategy function that:
      - On its first invocation, extracts available tickers from historical data,
        allocates equal capital per ticker, and buys shares via lh.buy.
      - Stores the portfolio (ticker: {shares, buy_price}) in a shared dict.
      - On subsequent calls, simply holds the positions.
    """
    def strategy_func(historical_data):
        # historical_data is expected to be a dict with key "value" containing a list of stock records.
        if not shared_portfolio:  # If nothing has been bought yet
            records = historical_data.get("value", [])
            tickers = set()
            # Assume each record has a 'ticker' key.
            for record in records:
                ticker = record.get("ticker")
                if ticker:
                    tickers.add(ticker)
            tickers = list(tickers)
            if not tickers:
                print("No tickers found in historical data.")
                return {"action": "none", "message": "No tickers found."}
            allocation = starting_capital / len(tickers)
            for ticker in tickers:
                # Retrieve the current price via the API wrapper’s static method.
                price_data = Strategy.get_current_price(ticker)
                price = price_data.get("price")
                if price is None:
                    print(f"Price for {ticker} not available, skipping.")
                    continue
                shares = int(allocation // price)
                if shares <= 0:
                    print(f"Not enough allocation for {ticker} at price {price}, skipping.")
                    continue
                # Place a buy order.
                buy_result = lh.buy(ticker, shares)
                executed_shares = buy_result.get("amount", shares)
                executed_price = buy_result.get("price", price)
                shared_portfolio[ticker] = {"shares": executed_shares, "buy_price": executed_price}
                print(f"Bought {executed_shares} of {ticker} at {executed_price}")
            return {"action": "buy", "portfolio": dict(shared_portfolio)}
        else:
            # Already bought – hold positions.
            return {"action": "hold", "portfolio": dict(shared_portfolio)}
    return strategy_func

def sell_all(shared_portfolio):
    """
    Iterates over the shared portfolio and sells every holding using lh.sell.
    Returns the total capital realized from sales.
    """
    capital = 0
    for ticker, data in shared_portfolio.items():
        shares = data.get("shares", 0)
        sell_result = lh.sell(ticker, shares)
        executed_shares = sell_result.get("amount", shares)
        executed_price = sell_result.get("price", 0)
        capital += executed_shares * executed_price
        print(f"Sold {executed_shares} of {ticker} at {executed_price}")
    return capital

# --------------------------
# Simple Test Functions
# --------------------------
def test_create_index_strategy_func():
    print("Running test_create_index_strategy_func...")

    # Save original functions to restore later
    original_get_current_price = Strategy.get_current_price
    original_buy = lh.buy

    # Monkey-patch external API calls
    Strategy.get_current_price = lambda ticker: {"price": 10}
    lh.buy = lambda ticker, shares: {"amount": shares, "price": 10}

    shared_portfolio = {}
    strategy = create_index_strategy_func(shared_portfolio, starting_capital=100)
    historical_data = {"value": [{"ticker": "AAPL"}, {"ticker": "GOOG"}]}

    # First call should trigger buy orders
    result = strategy(historical_data)
    print("First call result:", result)

    # Second call should hold existing positions
    result_hold = strategy(historical_data)
    print("Second call result:", result_hold)

    # Restore original functions
    Strategy.get_current_price = original_get_current_price
    lh.buy = original_buy

def test_sell_all():
    print("Running test_sell_all...")

    # Save original sell function to restore later
    original_sell = lh.sell
    lh.sell = lambda ticker, shares: {"amount": shares, "price": 12}

    shared_portfolio = {
        "AAPL": {"shares": 5, "buy_price": 10},
        "GOOG": {"shares": 5, "buy_price": 10}
    }
    capital = sell_all(shared_portfolio)
    print("Total capital realized:", capital)

    # Restore original function
    lh.sell = original_sell

# Simple test
if __name__ == "__main__":
    test_create_index_strategy_func()
    print("-" * 40)
    test_sell_all()
    