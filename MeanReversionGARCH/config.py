"""
This file contains the configuration for the mean reversion strategy with GARCH.
"""

# Import used libraries
import os
from dotenv import load_dotenv

load_dotenv()

# Define class for configuration
class Config:

    # API Configuration
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

    # Strategy Parameters
    LOOKBACK_PERIOD = 20 # Number of days to calculate the rolling mean and standard deviation
    ENTRY_THRESHOLD = 2 # Number of standard deviations from the mean to trigger an entry
    EXIT_THRESHOLD = 0.5 # Number of standard deviations from the mean to trigger an exit

    # GARCH Parameters
    GARCH_LOOKBACK = 365 # Number of days to calculate the GARCH parameters
    GARCH_P = 1 # Number of lagged squared returns
    GARCH_Q = 1 # Number of lagged volatility
    GARCH_DIST = 'StudentsT' # Distribution of the returns

    # Backtesting Parameters
    INITIAL_CAPITAL = 100000 # Initial capital
    DATA_PERIOD_DAYS = 180 # Number of days to backtest

    # Data Parameters
    TIMEFRAME = '1Min' # Timeframe of the data
    SYMBOL = 'AAPL' # Symbol of the stock