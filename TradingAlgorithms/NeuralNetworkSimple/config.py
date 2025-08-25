""" This file contains the configuration for the neural network trading strategy """

# Import used libraries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:

    # API Configuration
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

    # LSTM model parameters
    SEQUENCE_LENGTH = 60 # Number of periods to use for prediction
    PREDICTION_HORIZON = 1 # Number of periods to predict
    LSTM_UNITS = 50 # Number of units in the LSTM layer
    DROPOUT_RATE = 0.2 # Dropout rate for regularization
    EPOCHS = 50 # Training epochs
    BATCH_SIZE = 32 # Batch size for training
    VALIDATION_SPLIT = 0.2 # Validation split for training
    LEARNING_RATE = 0.001 # Learning rate for the optimizer

    # Trading strategy parameters
    CONFIDENCE_THRESHOLD = 0.7 # Minimum prediction confidence to enter a trade (0-1 scale)
    POSITION_SIZE = 0.8 # Use XX% of available capital
    STOP_LOSS_PCT = 0.02 # Stop loss percentage
    TAKE_PROFIT_PCT = 0.04 # Take profit percentage

    # Transactions costs and market conditions
    COMMISSION_PER_TRADE = 0.0 # Commission per trade
    BID_ASK_SPREAD_PCT = 0.001 # Bid-ask spread simulation
    SLIPPAGE_PCT = 0.0005 # Slippage percentage

    # Backtesting parameters
    INITIAL_CAPITAL = 100000 # Initial capital
    DATA_PERIOD_DAYS = 365 # Number of days to backtest
    TRAIN_TEST_SPLIT = 0.8 # Split for training and testing

    # Data parameters
    TIMEFRAME = '5Min' # Timeframe of the data
    SYMBOL = 'AAPL' # Symbol of the stock