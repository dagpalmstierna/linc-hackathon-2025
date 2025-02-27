import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from kymatio import Scattering1D
import matplotlib.pyplot as plt
from index_backtest import load_csv_to_df


def compute_scattering_features(signal, scattering):
    """
    Given a 1D numpy array (signal), convert it to a torch tensor and compute scattering features.
    Applies:
      - Dropping the zeroth order coefficient,
      - A log transform (with a small epsilon) to reduce dynamic range,
      - Averaging along the time dimension to obtain a time-invariant representation.
    """
    # Convert the 1D numpy array to a torch tensor and add a batch dimension.
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # shape: (1, window_size)
    
    # Apply the scattering transform.
    features = scattering(x)
    
    # In case the scattering output is not a torch.Tensor, convert it.
    if not torch.is_tensor(features):
        features = torch.from_numpy(features)
    
    # Optionally drop the zeroth order coefficient.
    features = features[:, 1:, :]
    
    # Apply a log transform with a small epsilon to avoid taking the log of zero.
    log_eps = 1e-6
    features = torch.log(torch.abs(features) + log_eps)
    
    # Average along the last (time) dimension to get a time-invariant representation.
    features = torch.mean(features, dim=-1)
    
    # Remove the batch dimension and return.
    return features.squeeze(0)


def preprocess_and_feature_extract(pivot_df, window_size, scattering, prediction_horizon = 1):
    """
    For each ticker in pivot_df, slide a window of length window_size over its price series,
    compute scattering features for the window, and set the target as the next hour’s return.

    Returns: 
        X - feature matrix (numpy array),
        y - target vector (numpy array),
        time_list - list of timestamps,
        ticker_list - corresponding tickers.
    """
    X_list, y_list, time_list, ticker_list = [], [], [], []

    for ticker in pivot_df.columns:
        prices = pivot_df[ticker].values
        
        # Loop over time indices with enough history and a following return.
        for t in range(window_size, len(prices) - prediction_horizon):
            window = prices[t - window_size:t]
            std_val = np.std(window)
            if std_val == 0:
                continue
            
            # Normalize the window.
            window = (window - np.mean(window)) / std_val
            
            # Extract scattering features.
            features = compute_scattering_features(window, scattering)
            if np.isnan(features).any():
                continue
            
            # Define the target as the return from time t to t+1.
            target = (prices[t + prediction_horizon] - prices[t]) / prices[t]
            
            X_list.append(features)
            y_list.append(target)
            time_list.append(pivot_df.index[t])
            ticker_list.append(ticker)
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, time_list, ticker_list


class MlpRegressor(nn.Module):
    """
    A single-layer perceptron for regression.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MlpRegressor, self).__init__()
        self.net = nn.Linear(input_dim, output_dim)  # Only one linear transformation

    """
    A simple multi-layer perceptron for regression.
    """
    '''def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MlpRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )'''
    
    def forward(self, x):
        return self.net(x)


def train_wst_model(pivot_df, window_size=24, train_ratio=0.7, J=3, Q=1,
                    hidden_dim=64, num_epochs=10, batch_size=32, lr=1e-3, prediction_horizon = 1):
    """
    Trains a wavelet scattering + MLP regression model to predict next-hour returns.
    
    Suggestions for hyperparameter improvements:
      - Experiment with different window_size values.
      - Adjust scattering parameters (J, Q) to better capture the frequency scales relevant for your assets.
      - Modify hidden_dim, num_epochs, and batch_size to balance training speed and performance.
      - Tweak the learning rate (lr) for improved convergence.
    
    Returns:
        model (nn.Module): Trained PyTorch MLP model.
        X_test (np.array): Test features.
        y_test (np.array): Test targets.
        times_test (np.array): Timestamps corresponding to X_test.
        tickers_test (list): Tickers for each sample in X_test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize scattering transform.
    scattering = Scattering1D(J=J, shape=window_size, Q=Q)
    
    # Build the dataset across all tickers.
    X, y, time_list, ticker_list = preprocess_and_feature_extract(pivot_df, window_size, scattering, prediction_horizon = prediction_horizon)
    
    # Create a DataFrame to sort samples in global chronological order.
    df_all = pd.DataFrame({
        'time': time_list,
        'ticker': ticker_list,
        'y': y
    })
    feature_columns = [f'feat_{i}' for i in range(X.shape[1])]
    df_features = pd.DataFrame(X, columns=feature_columns)
    df_all = pd.concat([df_all, df_features], axis=1)
    
    # Sort by time.
    df_all.sort_values(by='time', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    
    # Time-based split: first train_ratio portion for training, rest for testing.
    split_index = int(len(df_all) * train_ratio)
    split_time = df_all.loc[split_index, 'time']
    train_mask = df_all['time'] <= split_time
    test_mask = df_all['time'] > split_time
    
    df_train = df_all[train_mask]
    df_test = df_all[test_mask]
    
    X_train = df_train[feature_columns].values
    y_train = df_train['y'].values
    times_train = df_train['time'].values
    tickers_train = df_train['ticker'].values
    
    X_test = df_test[feature_columns].values
    y_test = df_test['y'].values
    times_test = df_test['time'].values
    tickers_test = df_test['ticker'].values
    
    # Convert training data to PyTorch tensors.
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    
    # Build the MLP model.
    input_dim = X_train.shape[1]
    model = MlpRegressor(input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    
    # Prepare DataLoader for mini-batch training.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Using MSE loss for regression on returns.
    
    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.6f}")
    
    return model, X_test, y_test, times_test, tickers_test


def predict(model, X_test, y_test, times_test, tickers_test, capital=1000000):
    """
    Uses the trained PyTorch model to predict on test data and simulates trading
    based on the rule: invest if predicted return > 0.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert X_test to a torch tensor.
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    y_pred = y_pred_tensor.cpu().numpy().flatten()
    
    # Compile test results into a DataFrame.
    test_results = pd.DataFrame({
        'time': times_test,
        'ticker': tickers_test,
        'predicted_return': y_pred,
        'actual_return': y_test
    })
    
    # Generate a long-only trading signal: invest only if predicted return > 0.
    test_results['signal'] = (test_results['predicted_return'] > 0).astype(int)
    
    # For each timestamp, compute the average actual return of stocks with signal=1.
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
    """
    test_results = test_results.copy()
    test_results['actual_movement'] = (test_results['actual_return'] > 0).astype(int)
    accuracy = np.mean(test_results['signal'] == test_results['actual_movement'])
    
    avg_return = portfolio_df['strategy_return'].mean()
    std_return = portfolio_df['strategy_return'].std()
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return != 0 else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='ML Strategy Portfolio', color='orange')
    plt.title('ML Trading Strategy Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    stats_text = (
        f"Directional Accuracy: {accuracy:.2%}\n"
        f"Avg Return per period: {avg_return:.4f}\n"
        f"Std Dev of Return: {std_return:.4f}\n"
        f"Approx. Sharpe Ratio: {sharpe_ratio:.4f}"
    )
    print(stats_text) # Added print statement
    
    plt.gcf().text(0.15, 0.75, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()


def plot_wst_and_index(df_index, portfolio_df):
    """
    Plots portfolio values for both the index fund strategy and the ML trading strategy.
    """
    if 'gmtTime' in df_index.columns:
        df_index = df_index.set_index('gmtTime')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_index.index, df_index['portfolio_value'], label='Index Fund Portfolio', color='blue')
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='ML Strategy Portfolio', color='orange')
    plt.title("Portfolio Value Comparison: ML Strategy vs. Index Fund")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load data and convert to DataFrame.
    df = load_csv_to_df('./given_resources/stockPrices_hourly.csv')
    
    # Pivot the CSV so each column corresponds to a ticker's ask price.
    ticker_col = df.columns[-1]
    pivot_df = df.pivot(index='gmtTime', columns=ticker_col, values='askMedian')
    pivot_df = pivot_df.sort_index().ffill()  # Updated from fillna(method='ffill') to avoid deprecation warning.
    
    # Train the model.
    window_size=9  # 1 day is 9 hours, so 5 days is 45 hours
    J=3 # 2^J < window_size
    Q=2 # arbitrary number, keep it not too high
    hidden_dim=4 # removed
    num_epochs=10
    batch_size=32
    lr=0.0025
    prediction_horizon = 9
    print("Training the model with the following hyperparameters:")
    print(f"Window Size: {window_size}, J: {J}, Q: {Q}, Hidden Dim: {hidden_dim}, Num Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}")
    model, X_test, y_test, times_test, tickers_test = train_wst_model(pivot_df, window_size=window_size, train_ratio=0.7, J=J, Q=Q,
                    hidden_dim=hidden_dim, num_epochs=num_epochs, batch_size=batch_size, lr=lr, prediction_horizon = prediction_horizon)

    # Predict and obtain results.
    model, portfolio_df, test_results = predict(model, X_test, y_test, times_test, tickers_test, capital=1000000)
    
    # Extract index portfolio.
    df_index = load_csv_to_df('./given_resources/index_portfolio.csv')
    plot_wst_and_index(df_index, portfolio_df)
    plot_wst_with_stats(portfolio_df, test_results)
    
    # Save the backtesting results to CSV files.
    portfolio_df.to_csv('eliot/ml_strategy_portfolio.csv')
    test_results.to_csv('eliot/ml_strategy_test_results.csv')
