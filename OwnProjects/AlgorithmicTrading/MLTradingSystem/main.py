""" Main script to run the ML trading system. """

# Import used libraries
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_manager import DataManager
from strategy import MLTradingStrategy
from backtest import Backtester

def download_and_prepare_data(symbol: str, 
                             start_date: str, 
                             end_date: str,
                             timeframe: str = '1Min') -> pd.DataFrame:
    """
    Download and prepare data for backtesting.
    
    Args:
        symbol: Stock symbol to download
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Data timeframe (default 1Min)
        
    Returns:
        DataFrame with prepared market data
    """
    print(f"\nDownloading {symbol} data from {start_date} to {end_date}...")
    
    # Initialize data manager
    dm = DataManager()
    
    # Download data
    data = dm.get_data(symbol, start_date, end_date, timeframe)
    
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    print(f"Downloaded {len(data)} bars of data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Data quality checks
    print("\n--- Data Quality Check ---")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {data.index.duplicated().sum()}")
    
    # Remove any duplicates
    if data.index.duplicated().sum() > 0:
        print("Removing duplicate timestamps...")
        data = data[~data.index.duplicated(keep='first')]
    
    # Basic statistics
    print("\n--- Data Statistics ---")
    print(f"Average daily volume: ${data['volume'].mean():,.0f}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Average daily range: {(data['high'] - data['low']).mean() / data['close'].mean() * 100:.2f}%")
    
    return data


def run_backtest(data: pd.DataFrame, 
                symbol: str,
                save_results: bool = True) -> dict:
    """
    Run the complete ML trading strategy backtest.
    
    Args:
        data: Market data for backtesting
        symbol: Stock symbol for reporting
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with backtest results
    """
    print("\n" + "="*60)
    print(f"ML TRADING STRATEGY BACKTEST - {symbol}")
    print("="*60)
    
    # Strategy parameters (optimized for minute-level trading)
    strategy_params = {
        'lookback_period': 60,           # 1 hour lookback for indicators
        'prediction_horizon': 30,         # Predict 30 minutes ahead
        'lgb_weight': 0.6,               # 60% weight to LightGBM
        'ridge_weight': 0.4,             # 40% weight to Ridge
        'entry_threshold': 0.55,         # Conservative entry threshold
        'max_position': 0.4,             # Max 40% of capital in one position
        'stop_loss_pct': 0.02,           # 2% stop loss
        'trailing_stop_atr': 1.5         # 1.5x ATR trailing stop
    }
    
    print("\n--- Strategy Configuration ---")
    for param, value in strategy_params.items():
        print(f"  {param}: {value}")
    
    # Initialize backtest engine
    backtest = Backtester(
        initial_capital=100000,
        spread_cost=0.0001,           # 1 basis point
        market_impact_const=0.00005,  # Small market impact
        commission_per_share=0.0,     # Alpaca has no commission
        min_trade_interval=15,         # Minimum 15 minutes between trades
        market_open="09:45",           # Start trading 15 min after open
        market_close="15:45"           # Stop trading 15 min before close
    )
    
    print("\n--- Backtest Configuration ---")
    print(f"  Initial Capital: $100,000")
    print(f"  Transaction Costs: 1bp spread + market impact")
    print(f"  Trading Hours: 9:45 AM - 3:45 PM")
    print(f"  Min Trade Interval: 15 minutes")
    
    # Run walk-forward backtest
    results = backtest.run_walk_forward_backtest(data, strategy_params)
    
    # Print comprehensive report
    backtest.print_report()
    
    # Generate plots
    print("\n--- Generating Performance Visualizations ---")
    backtest.plot_results()


def main():

    data = download_and_prepare_data(symbol='AAPL', start_date='2024-05-01', end_date='2024-12-31', timeframe='1Min')

    run_backtest(data, symbol='AAPL')


if __name__ == "__main__":
    main()