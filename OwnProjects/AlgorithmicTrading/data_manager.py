"""
Alpace Data Manager
This file contains all code for downloading and preparing historical data from Alpace
for backtesting and training models.

Usage:

    from data_manager import DataManager
    dm = DataManager()

    # Get data for backtesting
    data = dm.get_data('AAPL', '2024-01-01', '2024-12-31', timeframe='1Min')

    # Get latest data for training
    latest_data = dm.get_latest_data('AAPL', days_back=30)
"""

# Import used libraries
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Union
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv


class DataManager:
    
    def __init__(self):
        """ Initialize the data manager """
        
        # Load environment variables from .env file
        load_dotenv()

        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.api_base_url = 'https://paper-api.alpaca.markets'

        if not self.api_key or not self.api_secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        # Initialize the Alpaca API
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret_key,
            base_url=self.api_base_url
        )

        self.market_tz = pytz.timezone('America/New_York')


    def get_data(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: str = '1Day') -> pd.DataFrame:
        """ Get historical data from Alpaca 
        
        Args:
            symbol: the symbol of the stock to get data for
            start_date: the start date of the data (YYYY-MM-DD or datetime object)
            end_date: the end date of the data (YYYY-MM-DD or datetime object)
            timeframe: the timeframe of the data

        Returns:
            pd.DataFrame: the historical data
        """

        # Convert strings to datetime if necessary
        if isinstance(start_date, str):
            start = pd.Timestamp(start_date, tz=self.market_tz)
        else:
            start = start_date
        if isinstance(end_date, str):
            end = pd.Timestamp(end_date, tz=self.market_tz)
        else:
            end = end_date

        print(f"Downloading data for {symbol} from {start_date} to {end_date} at {timeframe}")

        try:
            
            # Get bars from Alpace
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start = start.isoformat(),
                end = end.isoformat(),
                adjustment = 'raw',
            ).df

            if bars.empty:
                print(f"No data found for {symbol} from {start_date} to {end_date} at {timeframe}")
                return pd.DataFrame()

            # Clean up the data
            bars.index = pd.to_datetime(bars.index)
            bars.index = bars.index.tz_convert(self.market_tz)

            # Add returns (always usefull...)
            bars['returns'] = bars['close'].pct_change()

            print(f"Downloaded {len(bars)} bars for {symbol}")
            return bars


        except Exception as e:
            print(f"Error downloading data: {e}")
            raise e



        return pd.DataFrame()
        

    def get_latest_data(self, symbol: str, days_back: int = 30, timeframe: str = '1Day') -> pd.DataFrame:
        """ Get latest data from Alpaca 
        
        Args:
            symbol: the symbol of the stock to get data for
            days_back: the number of days to go back
            timeframe: the timeframe of the data

        Returns:
            pd.DataFrame: the latest data
        """

        end_date = datetime.now(tz=self.market_tz) - timedelta(minutes=15) # Take 15 minutes off due to free subscription...
        start_date = end_date - timedelta(days=days_back)

        return self.get_data(symbol, start_date, end_date, timeframe)


if __name__ == "__main__":

    dm = DataManager()

    data = dm.get_data('AAPL', '2024-01-01', '2024-01-10', timeframe='1Min')
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print("Last 5 rows:")
    print(data.tail())

    recent = dm.get_latest_data('AAPL', days_back=5, timeframe='1Min')
    print(f"Recent data shape: {recent.shape}")
    print(f"Recent data columns: {recent.columns.tolist()}")
    print("Last 5 rows:")
    print(recent.tail())

