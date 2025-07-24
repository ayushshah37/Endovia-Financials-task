import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_spot_data, load_options_data, get_option_price,
    get_atm_strike, calculate_expiry, calculate_expenses
)
from signal_engine import generate_signals

LOT_SIZE = 75 
INTEREST_CHARGE = 32.87

STARTING_CAPITAL = 200000

def backtest(spot_data, options_data):
    spot_data = generate_signals(spot_data, options_data)
    trades = []
    position = None
    capital = STARTING_CAPITAL

    for i in range(len(spot_data)):
        row = spot_data.iloc[i]
        timestamp = row['datetime']
        signal = row['final_signal']
        print(f"Row {i}, Signal: {signal}")
        spot_price = row['close']
        trade_date = timestamp.date()

        if position is None and signal != 0:
            strike = get_atm_strike(spot_price)
            option_type = 'CE' if signal == 1 else 'PE'
            expiry = calculate_expiry(timestamp)
            entry_price = get_option_price(options_data, timestamp, strike, expiry, option_type)
            if entry_price is None:
                continue
            position = {
                'strike_price': strike,
                'option_type': option_type,
                'entry_time': timestamp,
                'entry_price': entry_price,
                'expiry_date': expiry,
                'signal': signal,
                'trade_date': trade_date,
                'highest_price': entry_price,
            }

        elif position:
            current_price = get_option_price(
                options_data, timestamp,
                position['strike_price'],
                position['expiry_date'],
                position['option_type']
            )
            if current_price is None:
                continue  

            position['highest_price'] = max(position['highest_price'], current_price)

            exit_reason = None
            # Signal Change
            if row['signal'] != position['signal'] and row['signal'] != 0 and row['signal'] != 0:
                exit_reason = 'Signal Change'
            # Stop-loss 1.5%
            elif current_price <= position['entry_price'] * (1 - 0.015):
                exit_reason = 'Stop-loss'
            # Take-profit 3%
            elif current_price >= position['entry_price'] * (1 + 0.03):
                exit_reason = 'Take-profit'
            # Force exit at 15:15
            elif timestamp.time() >= pd.to_datetime("15:15").time():
                exit_reason = 'Force Exit 15:15'
            # End of Day
            elif i == len(spot_data) - 1 or spot_data.iloc[i + 1]['datetime'].date() != position['trade_date']:
                exit_reason = 'EOD'

            if exit_reason:
                m2m = (current_price - position['entry_price']) * LOT_SIZE
                gross_pnl = m2m
                expenses = 0  # Ignore expenses
                interest = 0  # Ignore interest
                net_pnl = gross_pnl

                capital += net_pnl  # Update capital

                trades.append({
                    'strike_price': position['strike_price'],
                    'option_type': position['option_type'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': timestamp,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'expiry_date': position['expiry_date'],
                    'm2m': m2m,
                    'trade_date': position['trade_date'],
                    'gross_pnl': gross_pnl,
                    'expenses': expenses,
                    'interest': interest,
                    'net_pnl': net_pnl,
                    'capital': capital
                })
                print(f"Entry price: {entry_price}, Exit price: {current_price}")
                position = None

    trades_df = pd.DataFrame(trades)
    print(trades[:5])
    trades_df.to_csv("trades.csv", index=False)

    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()
        trades_df['drawdown'] = trades_df['cum_pnl'] - trades_df['cum_pnl'].cummax()
        trades_df['drawdown_pct'] = trades_df['drawdown'] / trades_df['cum_pnl'].cummax() * 100

        trades_df['cum_pnl'].plot(title="Equity Curve")
        plt.xlabel("Trade Number")
        plt.ylabel("Cumulative PnL")
        plt.savefig("equity_curve.png")
        plt.clf()

        trades_df['drawdown'].plot(title="Drawdown")
        plt.xlabel("Trade Number")
        plt.ylabel("Drawdown")
        plt.savefig("drawdown.png")
        plt.clf()

        metrics = {
            'Total Trades': len(trades_df),
            'Winning Trades': (trades_df['net_pnl'] > 0).sum(),
            'Losing Trades': (trades_df['net_pnl'] < 0).sum(),
            'Win Rate (%)': (trades_df['net_pnl'] > 0).mean() * 100,
            'Total Net PnL': trades_df['net_pnl'].sum(),
            'Max Drawdown': trades_df['drawdown'].min(),
            'Max Drawdown %': trades_df['drawdown_pct'].min()
        }

        pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

    return trades_df
if __name__ == "__main__":
    spot_data = load_spot_data("spot_with_signals_2023.csv")
    options_data = load_options_data("options_data_2023.csv")
    trades_df = backtest(spot_data, options_data)

    print("Backtest Completed!")
    print(trades_df[['entry_time', 'exit_time', 'net_pnl', 'exit_reason']].tail())