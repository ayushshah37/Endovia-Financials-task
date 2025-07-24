# Endovia-Financials-task
strategy implementation
# Options Selling Strategy with ML Confirmation

This project implements and backtests an options selling strategy on NIFTY index data for the year 2023. It uses a composite signal derived from technical indicators to trigger trades and employs an LSTM model for additional analysis.

## Directory Structure

```
strategy-backtest/
├── data/
│   ├── spot_with_signals_2023.csv
│   └── options_data_2023.csv
├── results/
│   ├── equity_curve.png
│   ├── drawdown.png
│   ├── metrics.csv
│   └── trades.csv
├── indicators.py
├── signal_engine.py
├── model.py
├── backtest.py
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd strategy-backtest
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Place your `spot_with_signals_2023.csv` and `options_data_2023.csv` files inside the `data/` directory.
2.  Run the main script from the root directory of the project:
    ```bash
    python main.py
    ```
3.  The script will execute the full pipeline: data loading, signal generation, backtesting, and ML model training.
4.  All results, including charts and CSV files, will be saved in the `results/` directory.

## Strategy & Model Logic

### Indicators & Composite Signal

The strategy is driven by a composite signal generated from a voting system of three indicators:

1.  **RSI (Relative Strength Index):** A score of +1 is given for RSI > 65 (overbought), and -1 for RSI < 35 (oversold).
2.  **MACD (Moving Average Convergence Divergence):** A score of +1 is given when the MACD line crosses above the signal line, and -1 when it crosses below.
3.  **In-House `cross` Signal:** The pre-computed signal in the data is used directly, contributing +1 for a buy and -1 for a sell.

A **final signal** is generated only when there is high conviction (a total vote of +2 or -2), reducing noise and focusing on stronger trends.

### Options Strategy

The strategy involves selling options to profit from premium decay (theta).

-   **Buy Signal (Vote >= 2):** **Sell an At-The-Money (ATM) PUT option.** This is a bullish stance, expecting the market to rise or stay stable, causing the PUT premium to decrease.
-   **Sell Signal (Vote <= -2):** **Sell an At-The-Money (ATM) CALL option.** This is a bearish stance, expecting the market to fall or stay stable, causing the CALL premium to decrease.

### Risk Management

-   **Starting Capital:** ₹200,000
-   **Stop-Loss:** A trade is closed if the option premium increases by **150%** from the entry premium (e.g., premium sold at ₹100 is exited if it reaches ₹250). This is a standard risk-control measure for option selling.
-   **Take-Profit:** A trade is closed if **90%** of the premium has decayed (e.g., premium sold at ₹100 is exited if it reaches ₹10).
-   **Force Exit:** All open positions are squared off at **15:15 local time** to avoid holding overnight risk.

### Machine Learning Model (LSTM)

An **LSTM (Long Short-Term Memory)** model is trained on the spot indicator data to classify the composite signal. Its purpose is to learn the sequential patterns in the indicator data that lead to strong trading signals. While not used directly to trigger trades in this backtest, its accuracy provides confidence in the signal generation logic and can be used for building more advanced predictive models in the future.

## Interpreting Results

-   **`equity_curve.png`:** Visualizes the growth of your capital over the backtest period. An upward-sloping curve indicates a profitable strategy.
-   **`drawdown.png`:** Shows the percentage loss from the peak capital. Lower drawdowns are desirable.
-   **`metrics.csv`:** Provides key performance indicators:
    -   **Total Return %:** The overall profit/loss as a percentage of starting capital.
    -   **Max Drawdown %:** The largest peak-to-trough decline in capital.
    -   **Sharpe Ratio:** Measures risk-adjusted return. A higher Sharpe Ratio (typically > 1) is better.
-   **`trades.csv`:** A detailed log of every trade executed, including entry/exit times, instrument details, and P&L.
