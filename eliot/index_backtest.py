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

def create_index_fund(df, capital=1000000, index_only=False):
    # Ticker column
    ticker_col = df.columns[-1]
    
    # Get the first measurement per ticker based on gmtTime
    initial_prices = df.sort_values('gmtTime').groupby(ticker_col).first().reset_index()
    
    if index_only:
        # Only invest in the index stock "INDEX1"
        row = initial_prices[initial_prices[ticker_col] == "INDEX1"].iloc[0]
        ask_price = row['askMedian']
        shares = int(capital // ask_price)
        portfolio = {"INDEX1": shares}
        total_spent = shares * ask_price
        leftover = capital - total_spent
        print(f"Capital left after buys: ${leftover:,.2f}")
        return portfolio
    else:
        # Default portfolio: 50% index stock and 50% equally divided among the other stocks
        all_tickers = initial_prices[ticker_col].unique()
        
        allocation_index = capital * 0.5
        other_tickers = [ticker for ticker in all_tickers if ticker != "INDEX1"]
        allocation_other = (capital * 0.5) / len(other_tickers)
        
        portfolio = {}
        total_spent = 0
        for idx, row in initial_prices.iterrows():
            ticker = row[ticker_col]
            ask_price = row['askMedian']
            allocation = allocation_index if ticker == "INDEX1" else allocation_other
            shares = int(allocation // ask_price)  # integer division for shares
            portfolio[ticker] = shares
            cost = shares * ask_price
            total_spent += cost
        leftover = capital - total_spent
        print(f"Capital left after buys: ${leftover:,.2f}")
        return portfolio

def compute_portfolio_value_from_csv(portfolio, df):
    ticker_col = df.columns[-1]
    
    # Use pivot_table to aggregate duplicate entries by taking the first measurement
    pivot_df = df.pivot_table(index='gmtTime', columns=ticker_col, values='askMedian', aggfunc='first')
    
    # Ensure the DataFrame is sorted by time and fill missing values if needed
    pivot_df = pivot_df.sort_index().fillna(method='ffill')
    
    # Compute the portfolio value at each timestamp
    pivot_df['portfolio_value'] = sum(pivot_df[ticker] * shares for ticker, shares in portfolio.items())
    
    # Compute hourly change in portfolio value
    pivot_df['hourly_change'] = pivot_df['portfolio_value'].diff()
    
    return pivot_df

def plot_portfolio(portfolio_history, portfolio):
    """
    Plot the portfolio's value over time and the hourly income/loss changes in one window,
    and plot the value of each individual holding (ask price * shares) in a separate window.
    """
    # First figure with two subplots for portfolio value and hourly change
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(portfolio_history.index, portfolio_history['portfolio_value'])
    axs[0].set_title('Portfolio Value Over Time')
    axs[0].set_ylabel('Portfolio Value ($)')
    
    axs[1].bar(portfolio_history.index, portfolio_history['hourly_change'], color='skyblue')
    axs[1].set_title('Hourly Income/Loss')
    axs[1].set_ylabel('Change ($)')
    axs[1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    # Second figure for individual stock holding values
    fig2, ax2 = plt.subplots(figsize=(10, 6))
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
    df = load_csv_to_df('./given_resources/Historical_Data.csv')  # new data file

    # Ask the user for portfolio type
    choice = input("Enter 1 for default portfolio (50% index and 50% others), or 2 for index-only portfolio (100% index): ").strip()
    if choice == "2":
        portfolio = create_index_fund(df, capital=1000000, index_only=True)
    else:
        portfolio = create_index_fund(df, capital=1000000)
    
    print("Portfolio holdings:", portfolio)
    
    portfolio_history = compute_portfolio_value_from_csv(portfolio, df)
    plot_portfolio(portfolio_history, portfolio)
    save_portfolio_history(portfolio_history, 'eliot/index_portfolio.csv')
