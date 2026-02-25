# LINC Hackathon 2025 Winning Team

Winning submission by team Algo-rythmics for the LINC-STEM 2025 algorithmic trading hackathon.

Team: Dag Palmstierna, Paddy Fernandez de Viana, Eliot Montesino Petrén, Douglas Eklund

## What we built

The core idea was to run multiple strategies in parallel without blocking each other. A background process continuously polls the API and caches the latest market data, individual strategy processes read from that cache and place orders independently.

The main strategy we ran live combined MACD with dynamic RSI thresholds with tuning of overbought/oversold levels to each stock's recent price history rather than using fixed values. We also trained per-stock LSTM models offline and experimented with a wavelet scattering transform approach for feature extraction.

```
main.py              ← runs DataCollection + Strategy as separate processes
strategies/
  momentum_strategy.py   ← MACD + dynamic RSI 
  moving_average.py      ← simple MA crossover
  lstm_model.py          ← per-stock LSTM price predictor
backtests/
  index_backtest.py      ← equal-weight buy-and-hold baseline
  wst_backtest.py        ← wavelet scattering transform model
data/                ← historical price data from LINC
```

## Setup

Python 3.10 required. On Mac/Linux:

```bash
./setup.sh
source venv/bin/activate
```

On Windows, do the above manually with `python -m venv venv`.

> TA-Lib needs a C library installed separately — see [ta-lib/ta-lib-python](https://github.com/ta-lib/ta-lib-python#dependencies).

## Running

```bash
# Live strategy
python main.py

# Backtests
python -m backtests.index_backtest
python -m backtests.wst_backtest
```