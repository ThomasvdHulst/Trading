"""  This file contains the data manager for the mean reversion trading strategy with GARCH """

# Import used libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import warnings
warnings.filterwarnings('ignore')

# Define class for data manager
class DataManager:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL
        )
        self.data = None


    def download_data(self, symbol=None, days=None):
        """ Download historical data from Alpaca """
        symbol = symbol or self.config.SYMBOL
        days = days or self.config.DATA_PERIOD_DAYS

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            data = self.api.get_bars(
                symbol,
                timeframe=self.config.TIMEFRAME,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw',
                feed='iex'
            ).df

            print(f"Downloaded {len(data)} bars for {symbol}")
            print(f"Data time range: {data.index[0]} to {data.index[-1]}")

            self.data = data.dropna()
            print(f"Cleaned data: {len(self.data)} rows")

            return self.data
        
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None


    def get_data(self):
        """ Simply return the data (can also just use self.data, but more safe)"""
        return self.data
