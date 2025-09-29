""" This file contains the trading strategy including feature engineering,
model training and signal generation. """

# Import used libraries
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_manager import DataManager

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb


class MLTradingStrategy:
    """ This class contains the main trading strategy.
    It uses an ensemble of LightGBM and Ridge regression to generate
    signals and positions.
    """

    def __init__(self,
                lookback_period: int = 60,
                prediction_horizon: int = 30,
                lgb_weight: float = 0.6,
                ridge_weight: float = 0.4,
                entry_threshold: float = 0.55,
                exit_threshold: float = 0.45,
                max_position: float = 0.4,
                stop_loss_pct: float = 0.02,
                trailing_stop_atr: float = 1.5
                ):

        """ Initialize the trading strategy 
        
        Args:
            lookback_period: Number of periods for calculating indicatores
            prediction_horizon: Minutes ahead to predict
            lgb_weight: Weight for LightGBM in the ensemble
            rigde_weight: Weight for Ridge regression in the ensemble
            entry_threshold: Probability threshold for entering a position
            exit_threshold: Probability threshold for exiting a position
            max_position: Maximum position size as a fraction of portfolio
            stop_loss_pct: Stop loss percentage
            trailing_stop_atr: Trailing stop loss based on ATR
        """

        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.lgb_weight = lgb_weight
        self.ridge_weight = ridge_weight
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_atr = trailing_stop_atr

        # Models and scalers
        self.lgb_model = None
        self.ridge_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = None

        # Feature importance tracking
        self.feature_importance = {}

    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Calculate all indicators and features for the model.
        
        Args:
            data: The data to calculate the indicators for

        Returns:
            Dataframe with all indicators and features
        """

        df = data.copy()

        # Price-based features
        df['returns_1m'] = df['close'].pct_change(periods=1)
        df['returns_5m'] = df['close'].pct_change(periods=5)
        df['returns_15m'] = df['close'].pct_change(periods=15)
        df['returns_30m'] = df['close'].pct_change(periods=30)
        df['returns_60m'] = df['close'].pct_change(periods=60)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price position in range
        df['price_position'] = (df['close'] - df['low'].rolling(self.lookback_period).min()) / \
            (df['high'].rolling(self.lookback_period).max() - df['low'].rolling(self.lookback_period).min())

        # RSI - Relative Strength Index
        df['rsi'] = self.calculate_rsi(df['close'], period=14)

        # MACD - Moving Average Convergence Diverence
        exp1 = df['close'].ewm(span=12, adjust=False).mean() # EWM - Exponential moving average
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger bands
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['bb_upper'] = rolling_mean + (2 * rolling_std)
        df['bb_lower'] = rolling_mean - (2 * rolling_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR - Average True Range
        df['atr'] = self.calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr']/df['close']

        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_change'] = df['volume'].pct_change()

        # VWAP - Volume Weighted Average Price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        # OBV - On Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_change'] = df['obv'].pct_change()

        # Microstructure features
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_to_close_vol'] = df['close'].pct_change().rolling(window=20).std()

        # Time features
        df['hour'] = pd.to_datetime(df.index).hour
        df['minute'] = pd.to_datetime(df.index).minute
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        return df


    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """ Calculate the Relative Strength Index (RSI) 
        
        Args:
            prices: Series of prices
            period: RSI period

        Returns:
            Series with RSI values
        """

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ Calculate the Average True Range (ATR)
        
        Args:
            data: DataFrame with prices and high/low prices
            period: ATR period
            
        Returns:
            Series with ATR values
        """

        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr


    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Prepare the features for the model
        
        Args:
            data: DataFrame with prices and high/low prices

        Returns:
            DataFrame with features
        """

        feature_columns = [
            'returns_1m', 'returns_5m', 'returns_15m', 'returns_30m', 'returns_60m',
            'log_returns', 'price_position', 'rsi', 'macd_diff', 'bb_position',
            'atr_ratio', 'volume_ratio', 'volume_change', 'vwap_deviation', 
            'obv_change', 'high_low_spread', 'close_to_close_vol',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
        ]

        # Select only feature columns that exist
        available_features = [col for col in feature_columns if col in data.columns]
        features = data[available_features].copy()

        # Handle infinities and NaNs
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)

        return features


    def create_labels(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """ Create labels for training, predict if the price will go up in the next 'horizon' minutes
        
        Args:
            data: DataFrame with prices and high/low prices
            horizon: Number of minutes ahead to predict

        Returns:
            Series with labels (binary, 1 for up, 0 for down)
        """
        
        future_returns = data['close'].shift(-horizon) / data['close'] - 1

        # Create binary labels: 1 if positive return, 0 if negative return
        # Add small threshold to avoid noise around 0
        labels = (future_returns > 0.0001).astype(int)

        return labels


    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> None:
        """ Train the ensemble model

        Args:
            train_data: DataFrame with training data
            validation_data: DataFrame with validation data, optional and used for hyperparameter tuning
        """

        print("Starting model training!")

        # Calculate features
        train_features_df = self.calculate_indicators(train_data)
        X_train = self.prepare_features(train_features_df)
        y_train = self.create_labels(train_data, self.prediction_horizon)

        # Remove NaN rows
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        if validation_data is not None:

            validation_features_df = self.calculate_indicators(validation_data)
            X_val = self.prepare_features(validation_features_df)
            y_val = self.create_labels(validation_data, self.prediction_horizon)

            # Remove NaN rows
            mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())
            X_val = X_val[mask_val]
            y_val = y_val[mask_val]

            # Scale features
            X_val_scaled = self.feature_scaler.transform(X_val)

            eval_set = [(X_val_scaled, y_val)]
        else:
            eval_set = None


        # Train LightGBM Model
        print("Training LightGBM Model!")
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 37
        }

        lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
        if eval_set:
            lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)
            self.lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        else:
            self.lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=100,
            )

        # Store feature importance
        importance = self.lgb_model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(X_train.columns, importance))

        # Train Ridge Regression Model
        print("Training Ridge Regression Model!")
        self.ridge_model = Ridge(alpha=1.0, random_state=37)
        self.ridge_model.fit(X_train_scaled, y_train)

        self.is_fitted = True
        self.feature_columns = X_train.columns.tolist()

        # Calculate training accuracy
        train_pred = self.predict_proba(train_data)
        train_accuracy = np.mean((train_pred > 0.5) == y_train[mask])
        print(f"Training accuracy: {train_accuracy:.2%}")

        if validation_data is not None:
            val_pred = self.predict_proba(validation_data)
            val_accuracy = np.mean((val_pred > 0.5) == y_val[mask_val])
            print(f"Validation accuracy: {val_accuracy:.2%}")


    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """ Predict the probability of the price going up
        
        Args:
            data: DataFrame with prices and high/low prices
            
        Returns:
            Array of probabilities
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Calculate features
        features_df = self.calculate_indicators(data)
        X = self.prepare_features(features_df)

        # Ensure we have the same columns as the training data
        X = X[self.feature_columns]

        # Handle NaN values
        X = X.fillna(method='ffill').fillna(0)

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Get predictions for both models
        lgb_pred = self.lgb_model.predict(X_scaled, num_iteration=self.lgb_model.best_iteration)
        ridge_pred = self.ridge_model.predict(X_scaled)

        # Clip Ridge predictions to 0-1
        ridge_pred = np.clip(ridge_pred, 0, 1)

        # Ensemble prediction
        ensemble_pred = self.lgb_weight * lgb_pred + self.ridge_weight * ridge_pred

        return ensemble_pred 


    def generate_signals(self, data: pd.DataFrame, current_position: int = 0) -> pd.DataFrame:
        """ Generate signals based on the model predictions
        
        Args:
            data: DataFrame with prices and high/low prices
            current_position: Current position size (0 to 1)

        Returns:
            DataFrame with signals and position sizes
        """

        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']

        # Get model predictions
        predictions = self.predict_proba(data)
        signals['prediction'] = predictions

        # Calculate confidence-based position sizing
        signals['confidence'] = np.abs(predictions - 0.5) * 2

        # Generate entry/exit signals
        signals['signal'] = 0
        signals['position_size'] = 0

        for i in range(len(signals)):
            pred = signals['prediction'].iloc[i]
            confidence = signals['confidence'].iloc[i]

            if current_position == 0: # No position
                if pred > self.entry_threshold:
                    # Enter long position
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1

                    # Scale position by confidence, cap at max_position
                    base_position = 0.2
                    position = min(base_position * (1 + confidence), self.max_position)
                    signals.iloc[i, signals.columns.get_loc('position_size')] = position

            else: # In position
                if pred < self.exit_threshold:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0

        # Add ATR for stop-loss calculations
        signals['atr'] = self.calculate_atr(data, period=14)

        return signals


    def calculate_stops(self, entry_price: float, current_price: float, atr: float) -> Tuple[float, float]:
        """ Calculate the stop-loss and trailing stop-loss

        Args:
            entry_price: Entry price
            current_price: Current market price
            atr: Current ATR value

        Returns:
            Tuple of stop-loss and trailing stop-loss
        """

        # Fixed percentage stop
        stop_loss = entry_price * (1 - self.stop_loss_pct)

        # ATR-based trailing stop
        trailing_stop = current_price - (self.trailing_stop_atr * atr)

        return stop_loss, trailing_stop


    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """ Get the feature importance

        Ag
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names and importance
        """
        
        if not self.feature_importance:
            return {}

        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_features[:top_n])


def main():
    # Initialize the data manager
    dm = DataManager()
    
    # Set data parameters
    symbol = 'AAPL'
    start_date = '2024-12-01'
    end_date = '2024-12-31'
    timeframe = '1Min'
    init_capital = 100000

    # Get the data
    data = dm.get_data(symbol, start_date, end_date, timeframe)

    if data.empty:
        print("No data was found, please check the data manager and the inputs.")
        return
    
    print("Data was found:")
    print(data.head())
    
    # Initialize the strategy
    strategy = MLTradingStrategy()

    # Calculate indicators and features
    #data = strategy.calculate_indicators(data)
    #features = strategy.prepare_features(data)
    #print(features.head())

    train_data = data.iloc[:int(len(data)*0.8)]
    test_data = data.iloc[int(len(data)*0.8):]
    strategy.train(train_data, test_data)

    signals = strategy.generate_signals(test_data)
    print(signals.tail())

    # Print signal summary
    signals_1 = signals[signals['signal'] == 1]
    signals_minus_1 = signals[signals['signal'] == -1]
    signals_0 = signals[signals['signal'] == 0]
    print(f"Long signals: {len(signals_1)}")
    print(f"Short signals: {len(signals_minus_1)}")
    print(f"Neutral signals: {len(signals_0)}")
    print(f"Total signals: {len(signals)}")

    # Print feature importance
    feature_importance = strategy.get_feature_importance()
    print(feature_importance)
    
    

if __name__ == "__main__":
    main()


