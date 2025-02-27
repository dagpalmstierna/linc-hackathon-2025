import torch
import numpy as np
import pandas as pd
from kymatio import Scattering1D
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from index_backtest import load_csv_to_df

def compute_scattering_features(signal, scattering):
    # Convert the 1D numpy array to a torch tensor and add a batch dimension.
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # shape: (1, window_size)
    # Apply WST - Compute scattering coefficients
    features = scattering(x)
    # Squeeze the batch dimension and flatten the scattering coefficients.
    return features.squeeze().flatten()


def preprocess_and_feature_extract(pivot_df, window_size, scattering):
    """
    For each ticker in pivot_df, slide a window of length window_size over its price series,
    compute scattering features for the window, and set the target as the next hour’s return.

    Return: 
    X - feature matrix, 
    y - target vector, 
    time_lists - list of timestamps,
    ticker_list - corresponding tickers
    """
    X_list, y_list, time_list, ticker_list = [], [], [], []
    
    for ticker in pivot_df.columns: # Loop over every stock
        prices = pivot_df[ticker].values  # price series
        
        # Loop over time indices where we have enough history and a following return.
        for t in range(window_size, len(prices) - 1):
            window = prices[t - window_size:t] 
            std_val = np.std(window) 
            if std_val == 0:
                continue
            
            # Normalize the window
            window = (window - np.mean(window)) / std_val
            
            # Extract features using the scattering transform.
            features = compute_scattering_features(window, scattering) # shape: (n_features,) TODO: true?
            
            # If scattering features contain NaN (feature extraction fail), skip this sample.
            if np.isnan(features).any():
                continue
            
            # Define the target as the return from time t to t+1.
            target = (prices[t + 1] - prices[t]) / prices[t]
            
            X_list.append(features) # shape: (n_samples, n_features) TODO: true?
            y_list.append(target) # shape: (n_samples,), TODO: true?
            time_list.append(pivot_df.index[t])
            ticker_list.append(ticker)
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, time_list, ticker_list

def ml_trading_strategy(pivot_df, window_size=24, capital=1000000, train_ratio=0.7, J = 3, Q = 1, plot=True):
    """
    J - scale parameter for the scattering transform
    Q - quality factor for the scattering transform
    train_ratio - proportion of data to use for training
    
    Implements a long-only ML trading strategy:
      - Uses a sliding window of ask prices to compute wavelet scattering features.
      - Trains a linear regression model to predict the next hour's return.
      - Generates a trading signal (1 if predicted return > 0, else 0).
      - Simulates portfolio performance by averaging actual returns for all stocks with a positive signal at each timestamp.
    """
    
    # Initialize the scattering transform (Kymatio) with chosen parameters.
    scattering = Scattering1D(J=J, shape=window_size, Q=Q)
    
    # Build the ML dataset.
    X, y, time_list, tickers = preprocess_and_feature_extract(pivot_df, window_size, scattering)
    
    # Split the dataset chronologically into training and test sets.
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    times_train, times_test = np.array(time_list[:split_idx]), np.array(time_list[split_idx:])
    tickers_train, tickers_test = tickers[:split_idx], tickers[split_idx:]
    
    # print(n_samples , " = ", len(X_train), " + ", len(X_test)) # OK!

    # Train a simple linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train) 
    
    # Predict returns on the test set.
    y_pred = model.predict(X_test)
    
    # Compile test results into a DataFrame.
    test_results = pd.DataFrame({
        'time': times_test,
        'ticker': tickers_test,
        'predicted_return': y_pred,
        'actual_return': y_test
    })
    
    # Generate a long-only trading signal: invest only if predicted return is positive.
    test_results['signal'] = (test_results['predicted_return'] > 0).astype(int)
    
    # For each timestamp in the test set, compute the average actual return of stocks with signal=1.
    grouped = test_results.groupby('time').apply(
        lambda grp: np.mean(grp.loc[grp['signal'] == 1, 'actual_return']) if np.any(grp['signal'] == 1) else 0
    )
    portfolio_returns = grouped.sort_index()
    
    # Simulate the portfolio value over the test period.
    portfolio_value = [capital]
    for r in portfolio_returns:
        portfolio_value.append(portfolio_value[-1] * (1 + r))
    portfolio_value = portfolio_value[1:]
    
    portfolio_df = pd.DataFrame({
        'time': portfolio_returns.index,
        'portfolio_value': portfolio_value,
        'strategy_return': portfolio_returns.values
    }).set_index('time')

    if plot:
        # Plot the portfolio value over time.
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='ML Strategy Portfolio')
        plt.title('ML Trading Strategy Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return model, portfolio_df, test_results


if __name__ == "__main__":
    # Load data and convert to pandas DataFrame
    df = load_csv_to_df('./given_resources/stockPrices_hourly.csv')

    # Pivot the CSV so that each column corresponds to a ticker's ask price.
    # Essentially, each column is a time series of ask prices for a given stock. First column is the timestamp.
    ticker_col = df.columns[-1]
    pivot_df = df.pivot(index='gmtTime', columns=ticker_col, values='askMedian')
    pivot_df = pivot_df.sort_index().fillna(method='ffill')

    # Run the ML trading strategy backtest
    model, portfolio_df, test_results = ml_trading_strategy(pivot_df, window_size=24, capital=1000000, train_ratio=0.7)

    # Save the backtesting results to CSV files.
    portfolio_df.to_csv('eliot/ml_strategy_portfolio.csv')
    test_results.to_csv('eliot/ml_strategy_test_results.csv')
