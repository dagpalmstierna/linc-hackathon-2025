import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
This is a backtesting script for an index fund strategy based on historical stock 
price data given from LINC.

The index strategy buys an equal amount (in dollars) of each stock in the CSV file and 
tracks the portfolio's value over time.

Find the CSV file with the index backtest results in index_portfolio.csv
'''

def load_csv_to_df(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['gmtTime'])
    print("CSV Columns:", df.columns)
    return df

def create_index_fund(df, capital=1000000):
    # Ticker column
    ticker_col = df.columns[-1]
    
    # Get the first measurement per ticker based on gmtTime
    initial_prices = df.sort_values('gmtTime').groupby(ticker_col).first().reset_index()
    
    tickers = initial_prices[ticker_col].unique()
    n_stocks = len(tickers)
    allocation = capital / n_stocks  # equal allocation for each stock
    
    portfolio = {}
    total_spent = 0
    for idx, row in initial_prices.iterrows():
        ticker = row[ticker_col]
        ask_price = row['askMedian']
        shares = int(allocation // ask_price)  # integer division
        portfolio[ticker] = shares
        cost = shares * ask_price
        total_spent += cost
    leftover = capital - total_spent
    print(f"Capital left after buys: ${leftover:,.2f}")
    return portfolio


def compute_portfolio_value_from_csv(portfolio, df):
    ticker_col = df.columns[-1]
    
    # Pivot the CSV so that each ticker's ask price is in its own column
    pivot_df = df.pivot(index='gmtTime', columns=ticker_col, values='askMedian')
    
    # Ensure the DataFrame is sorted by time and fill missing values if needed
    pivot_df = pivot_df.sort_index().fillna(method='ffill')
    
    # Compute the portfolio value at each timestamp
    pivot_df['portfolio_value'] = sum(pivot_df[ticker] * shares for ticker, shares in portfolio.items())
    
    # Compute hourly change in portfolio value
    pivot_df['hourly_change'] = pivot_df['portfolio_value'].diff()
    
    # Return the full DataFrame including individual stock series
    return pivot_df

def plot_portfolio(portfolio_history, portfolio):
    """
    Plot the portfolio's value over time and the hourly income/loss changes in one window,
    and plot the value of each individual holding (ask price * shares) in a separate window.
    
    Parameters:
        portfolio_history (pd.DataFrame): A DataFrame with a datetime index.
                                          It should include columns for individual stocks,
                                          'portfolio_value', and 'hourly_change'.
        portfolio (dict): A dictionary of stock tickers and their share counts.
    """
    # First figure with two subplots for portfolio value and hourly change
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot portfolio value over time
    axs[0].plot(portfolio_history.index, portfolio_history['portfolio_value'])
    axs[0].set_title('Portfolio Value Over Time')
    axs[0].set_ylabel('Portfolio Value ($)')
    
    # Plot hourly income/loss changes
    axs[1].bar(portfolio_history.index, portfolio_history['hourly_change'], color='skyblue')
    axs[1].set_title('Hourly Income/Loss')
    axs[1].set_ylabel('Change ($)')
    axs[1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    # Second figure for individual stock holding values
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # For each ticker, multiply its ask price series by the number of shares held to get holding value
    for ticker, shares in portfolio.items():
        holding_value = portfolio_history[ticker] * shares
        ax2.plot(portfolio_history.index, holding_value, label=ticker)
    ax2.set_title('Individual Holding Values Over Time')
    ax2.set_ylabel('Holding Value ($)')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def save_portfolio_history(portfolio_history, file_path):
    portfolio_history.to_csv(file_path)

# Backtest the index fund strategy
if __name__ == "__main__":
    # Load CSV into DataFrame (update the file path as needed)
    df = load_csv_to_df('./given_resources/stockPrices_hourly.csv')
    
    # Create an index fund portfolio from the CSV data
    portfolio = create_index_fund(df, capital=1000000) # start capital is $1,000,000
    print("Portfolio holdings:", portfolio)
    
    # Compute portfolio value over time using historical data from the CSV
    portfolio_history = compute_portfolio_value_from_csv(portfolio, df)
    #print("Portfolio value history:")
    #print(portfolio_history.head())
    
    # Plot portfolio value, hourly income/loss (stationary time series) and individual stock holding values
    plot_portfolio(portfolio_history, portfolio)

    # Save the portfolio history to a CSV file
    save_portfolio_history(portfolio_history, 'eliot/index_portfolio.csv')


# The saved CSV file, "index_portfolio.csv", is structured as follows:

# 1. "gmtTime" (datetime index): 
#    - Represents the timestamp for each recorded market price.
#    - This column is used as the index in the DataFrame.

# 2. Columns for individual stock tickers:
#    - Each column represents a different stock.
#    - Values in these columns correspond to the stock's "askMedian" price at the given timestamp.
#    - These values are used to track the price changes over time for each stock in the portfolio.

# 3. "portfolio_value":
#    - Represents the total value of the portfolio at each timestamp.
#    - Computed as the sum of (stock price * shares held) for all stocks in the portfolio.
#    - This column is used for analyzing the performance of the index strategy.

# 4. "hourly_change":
#    - Represents the change in portfolio value from the previous hour.
#    - Computed using the difference between consecutive rows in the "portfolio_value" column.
#    - Used for tracking hourly gains/losses of the portfolio.
