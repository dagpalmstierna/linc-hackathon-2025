import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

def train_wst_model(pivot_df, window_size=24, train_ratio=0.7, J=3, Q=1):
    """
    J - scale parameter for the scattering transform
    Q - quality factor for the scattering transform
    train_ratio - proportion of data (by time) to use for training

    Implements a long-only ML trading strategy:
      - Uses a sliding window of ask prices to compute wavelet scattering features.
      - Trains a linear regression model to predict the next hour's return.
      - Generates a trading signal (1 if predicted return > 0, else 0).
      - Simulates portfolio performance.
    """

    # 1) Initialize scattering transform
    scattering = Scattering1D(J=J, shape=window_size, Q=Q)

    # 2) Build the ML dataset for *all* tickers
    X, y, time_list, ticker_list = preprocess_and_feature_extract(
        pivot_df, window_size, scattering
    )

    # -----------------------------
    # 3) Create a DataFrame so we can sort by actual time across all tickers
    # -----------------------------
    df_all = pd.DataFrame({
        'time': time_list,
        'ticker': ticker_list,
        'y': y
    })

    # X is a 2D array of features. We need to add them as separate columns:
    feature_columns = [f'feat_{i}' for i in range(X.shape[1])]
    df_features = pd.DataFrame(X, columns=feature_columns)
    df_all = pd.concat([df_all, df_features], axis=1)

    # 4) Sort by 'time' so that we have one global chronological order
    df_all.sort_values(by='time', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # -----------------------------
    # 5) Perform a purely time-based split
    # -----------------------------
    # Find the row at the 70% index (or any ratio you wish) in chronological order:
    split_index = int(len(df_all) * train_ratio)
    split_time = df_all.loc[split_index, 'time']  # The time boundary

    # Define train/test masks based on the actual 'time':
    train_mask = df_all['time'] <= split_time
    test_mask = df_all['time'] > split_time

    # Create train sets
    df_train = df_all[train_mask]
    X_train = df_train[feature_columns].values
    y_train = df_train['y'].values
    times_train = df_train['time'].values
    tickers_train = df_train['ticker'].values

    # Create test sets
    df_test = df_all[test_mask]
    X_test = df_test[feature_columns].values
    y_test = df_test['y'].values
    times_test = df_test['time'].values
    tickers_test = df_test['ticker'].values

    # 6) Fit a simple linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test, times_test, tickers_test

    
def predict(model, X_test, y_test, times_test, tickers_test, capital=1000000):
    
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
    return model, portfolio_df, test_results

def plot_wst_with_stats(portfolio_df, test_results):
    """
    Plots the ML strategy portfolio value over time and annotates the plot with performance statistics.

    Parameters:
        portfolio_df (pd.DataFrame): DataFrame containing the ML strategy portfolio history.
                                     It must include 'portfolio_value' and 'strategy_return' columns,
                                     and have a datetime index.
        test_results (pd.DataFrame): DataFrame containing test predictions and actual returns.
                                     It must include 'signal', 'actual_return', and 'time' columns.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Compute directional accuracy: percentage of times the signal correctly predicted the movement.
    # We define the actual movement as 1 if actual_return > 0, else 0.
    test_results = test_results.copy()
    test_results['actual_movement'] = (test_results['actual_return'] > 0).astype(int)
    accuracy = np.mean(test_results['signal'] == test_results['actual_movement'])
    
    # Compute strategy statistics from the portfolio returns.
    # Assuming portfolio_df has a column 'strategy_return' with the return for each period.
    avg_return = portfolio_df['strategy_return'].mean()
    std_return = portfolio_df['strategy_return'].std()
    # Annualize Sharpe ratio (scaling factor sqrt(252) for hourly data may need adjustment).
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return != 0 else np.nan
    
    # Plot the portfolio value over time.
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='ML Strategy Portfolio', color='orange')
    plt.title('ML Trading Strategy Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    # Create a text box with the statistics.
    stats_text = (
        f"Directional Accuracy: {accuracy:.2%}\n"
        f"Avg Return per period: {avg_return:.4f}\n"
        f"Std Dev of Return: {std_return:.4f}\n"
        f"Approx. Sharpe Ratio: {sharpe_ratio:.4f}"
    )
    
    # Annotate the plot with the performance statistics.
    plt.gcf().text(0.15, 0.75, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()


def plot_wst_and_index(df_index, portfolio_df):
    """
    Plots the portfolio values over time for both the index fund strategy and the ML trading strategy
    on the same figure for easy comparison.
    
    Parameters:
        df_index (pd.DataFrame): DataFrame containing the index fund portfolio history.
                                 Expected to include a 'portfolio_value' column and a datetime column.
        portfolio_df (pd.DataFrame): DataFrame containing the ML strategy portfolio history.
                                     Its index is assumed to be datetime and it includes a 'portfolio_value' column.
    """
    # Ensure the index fund DataFrame uses a datetime index.
    # If 'gmtTime' is a column, set it as the DataFrame index.
    if 'gmtTime' in df_index.columns:
        df_index = df_index.set_index('gmtTime')
    
    plt.figure(figsize=(10, 6))
    
    # Plot the Index Fund Portfolio Value.
    plt.plot(df_index.index, df_index['portfolio_value'], label='Index Fund Portfolio', color='blue')
    
    # Plot the ML Strategy Portfolio Value.
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='ML Strategy Portfolio', color='orange')
    
    plt.title("Portfolio Value Comparison: ML Strategy vs. Index Fund")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Load data and convert to pandas DataFrame
    df = load_csv_to_df('./given_resources/stockPrices_hourly.csv')

    # Pivot the CSV so that each column corresponds to a ticker's ask price.
    ticker_col = df.columns[-1]
    pivot_df = df.pivot(index='gmtTime', columns=ticker_col, values='askMedian')
    pivot_df = pivot_df.sort_index().fillna(method='ffill')

    #print(pivot_df)

    # Train
    model, X_test, y_test, times_test, tickers_test = train_wst_model(pivot_df, window_size=24, train_ratio=0.7, J = 3, Q = 1)
    
    # Predict and results
    model, portfolio_df, test_results = predict(model, X_test, y_test, times_test, tickers_test, capital=1000000)
    
    # Extract index
    df_index = load_csv_to_df('./given_resources/index_portfolio.csv')

    # Plot results and compare against index
    plot_wst_and_index(df_index, portfolio_df)

    plot_wst_with_stats(portfolio_df, test_results)

    # Save the backtesting results to CSV files.
    portfolio_df.to_csv('eliot/ml_strategy_portfolio.csv')
    test_results.to_csv('eliot/ml_strategy_test_results.csv')
