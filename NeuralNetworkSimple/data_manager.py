""" This file contains the data manager for the neural network trading strategy """

# Import used libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL
        )
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None


    def download_data(self, symbol=None, days=None):
        """ Download historical data from Alpaca """
        
        # Set the symbol and number of days
        symbol = symbol or self.config.SYMBOL
        days = days or self.config.DATA_PERIOD_DAYS

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Download the data
        try:
            data = self.api.get_bars(
                symbol=symbol,
                timeframe=self.config.TIMEFRAME,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw',
                feed='iex'
            ).df

            print(f"Downloaded {len(data)} bars for {symbol}")
            print(f"Data time range: {data.index[0]} to {data.index[-1]}")

            self.raw_data = data.dropna()
            print(f"Cleaned data: {len(self.raw_data)} rows")

            return self.raw_data
        
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
        

    def calc_indicators(self, data):
        """ Calculate technical indicators for LSTM features """

        data = data.copy()

        # Simple moving averages
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()

        # Exponential moving averages
        data['EMA_12'] = data['close'].ewm(span=12).mean()
        data['EMA_26'] = data['close'].ewm(span=26).mean()

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

        # Relative strength index (RSI)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        bb_std = data['close'].rolling(window=bb_period).std()
        data['BB_middle'] = data['close'].rolling(window=bb_period).mean()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (data['close'] - data['BB_lower']) / data['BB_width']

        # Volume indicators
        data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['volume'] / data['Volume_SMA']

        # Price change
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        data['returns_1h'] = data['close'].pct_change(periods=12)
        data['volatility'] = data['returns'].rolling(window=20).std()

        # High-low features
        data['HL_ratio'] = (data['high'] - data['low']) / data['close']
        data['CO_ratio'] = (data['close'] - data['open']) / data['open']

        return data
    
    
    def prepare_lstm_features(self, data):
        """ Prepare data for LSTM model """

        data = data.copy()

        # Calculate technical indicators
        data = self.calc_indicators(data)

        # Select features (for now, all features)
        #feature_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'RSI_14', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position', 'Volume_SMA', 'Volume_ratio', 'returns', 'log_returns', 'returns_1h', 'volatility', 'HL_ratio', 'CO_ratio']
        feature_columns = ['close', 'volume', 'RSI_14', 'MACD', 'BB_position', 'volatility', 'returns', 'Volume_ratio']

        # Calculate forward return: (future_price - current_price) / current_price
        data['target'] = (data['close'].shift(-self.config.PREDICTION_HORIZON) - data['close']) / data['close']

        # Remove rows with missing values
        data = data.dropna()

        self.feature_columns = feature_columns
        self.processed_data = data

        return data
    

    def get_processed_data(self):
        """ Get the processed data """
        return self.processed_data
    
    
    def get_feature_columns(self):
        """ Get the feature columns """
        return self.feature_columns
    
    
    def print_data_summary(self):
        """ Print a summary of the data """

        if self.processed_data is None:
            print("No processed data available")
            return

        print("Data Summary:")
        print(f"Total samples: {len(self.processed_data)}")
        print(f"Features: {self.feature_columns} ({len(self.feature_columns)} columns)")
        print(f"Data time range: {self.processed_data.index[0]} to {self.processed_data.index[-1]}")

        print("Target statistics:")
        print(f"Mean: {self.processed_data['target'].mean():.4f}")
        print(f"Std: {self.processed_data['target'].std():.4f}")
        print(f"Min: {self.processed_data['target'].min():.4f}")
        print(f"Max: {self.processed_data['target'].max():.4f}")
        print("\n")

