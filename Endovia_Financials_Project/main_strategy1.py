from utils import load_spot_data, load_options_data
from backtest import backtest

spot_data = load_spot_data("spot_with_signals_2023.csv")
options_data = load_options_data("options_data_2023.csv")

trades_df = backtest(spot_data, options_data)       
print("Backtest Completed!")
print(trades_df[['entry_time', 'exit_time', 'net_pnl', 'exit_reason']].tail()) 