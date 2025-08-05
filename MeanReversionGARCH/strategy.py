""" This file contains the strategy for the mean reversion trading strategy with GARCH """

# Import used libraries
import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, config):
        self.config = config
        self.lookback = config.LOOKBACK_PERIOD
        self.entry_threshold = config.ENTRY_THRESHOLD
        self.exit_threshold = config.EXIT_THRESHOLD


    def calculate_indicators(self, data):
        """ Calculate the indicators for the strategy """

        # Copy the data
        data = data.copy()

        # Calculate rolling statistics
        data['rolling_mean'] = data['close'].rolling(window=self.lookback).mean()
        data['rolling_std'] = data['close'].rolling(window=self.lookback).std()

        # Calculate z-scores
        data['z_score'] = (data['close'] - data['rolling_mean']) / data['rolling_std']

        if 'forecasted_volatility' in data.columns:
            # Use GARCH volatility if available
            data['garch_z_score'] = (data['close'] - data['rolling_mean']) / (data['forecasted_volatility'] * np.sqrt(252))

            # Volatility trend indicator (high/low volatility periods)
            median_vol = data['forecasted_volatility'].median()
            data['high_vol_period'] = (data['forecasted_volatility'] > median_vol).astype(int)

        else:
            print("Warning: GARCH volatility not available, using simple z-score")

        data = data.dropna()

        return data
    

    def generate_signals(self, data):
        """ Generate trading signals based on the indicators """

        data = data.copy()

        # Choose which z-score to use (should be GARCH z-score...)
        if 'garch_z_score' in data.columns:
            signal_column = 'garch_z_score'
            print("Using GARCH z-score for signal generation")
        else:
            signal_column = 'z_score'
            print("Using simple z-score for signal generation")


        # Initialize signal columns
        data['position'] = 0 # 0 = no position, 1 = long, -1 = short
        data['signal'] = 0 # 0 = no signal, 1 = long entry, -1 = short entry

        current_position = 0

        for i in range(len(data)):

            z_score = data.iloc[i][signal_column]

            # Adjust thresholds based on volatility
            if 'forecasted_volatility' in data.columns:

                # Set volatility adjustment based on high/low volatility
                if 'high_vol_period' not in data.iloc[i].index:
                    vol_adjustment = 1 # If column doesnt exist, just use 1
                elif data.iloc[i]['high_vol_period'] == 1:
                    vol_adjustment = 1.3 # If high volatility, increase threshold by 30%
                else:
                    vol_adjustment = 0.7 # If low volatility, decrease threshold by 30%
                    
                # Adjust thresholds
                entry_threshold = self.entry_threshold * vol_adjustment
                exit_threshold = self.exit_threshold * vol_adjustment

            else:
                entry_threshold = self.entry_threshold
                exit_threshold = self.exit_threshold


            if current_position == 0: # In case of no position
                if z_score < -entry_threshold: # If z-score is below entry threshold, enter a long position
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    current_position = 1
                elif z_score > entry_threshold: # If z-score is above entry threshold, enter a short position
                    data.iloc[i, data.columns.get_loc('signal')] = -1
                    current_position = -1

            elif current_position == 1: # In case we have a long position
                if abs(z_score) < exit_threshold: # If z-score is below exit threshold, exit the long position
                    data.iloc[i, data.columns.get_loc('signal')] = 0
                    current_position = 0

            elif current_position == -1: # In case we have a short position
                if abs(z_score) < exit_threshold: # If z-score is below exit threshold, exit the short position
                    data.iloc[i, data.columns.get_loc('signal')] = 0
                    current_position = 0

            data.iloc[i, data.columns.get_loc('position')] = current_position

        return data
    

    def print_signal_summary(self, data):
        """ Print summary statistics of the signals """

        print("---------------------")
        print("Signal summary:")
        print("Z-score statistics:")
        print(f"Mean: {data['z_score'].mean():.4f}")
        print(f"Std: {data['z_score'].std():.4f}")
        print(f"Min: {data['z_score'].min():.4f}")
        print(f"Max: {data['z_score'].max():.4f}")

        if 'garch_z_score' in data.columns:
            print("GARCH z-score statistics:")
            print(f"Mean: {data['garch_z_score'].mean():.4f}")
            print(f"Std: {data['garch_z_score'].std():.4f}")
            print(f"Min: {data['garch_z_score'].min():.4f}")
            print(f"Max: {data['garch_z_score'].max():.4f}")

        print("---------------------")

        if 'high_vol_period' in data.columns:
            print("Volatility trend statistics:")
            print(f"High volatility periods: {len(data[data['high_vol_period'] == 1])} ({len(data[data['high_vol_period'] == 1]) / len(data) * 100:.2f}%)")
            print(f"Low volatility periods: {len(data[data['high_vol_period'] == 0])} ({len(data[data['high_vol_period'] == 0]) / len(data) * 100:.2f}%)")
            print("---------------------")

        print("Position statistics:")
        print(f"Long positions: {len(data[data['position'] == 1])}")
        print(f"Short positions: {len(data[data['position'] == -1])}")
        print(f"Neutral positions: {len(data[data['position'] == 0])}")
        print("---------------------")

        print("Signal statistics:")
        print(f"Long signals: {len(data[data['signal'] == 1])}")
        print(f"Short signals: {len(data[data['signal'] == -1])}")
        print(f"Neutral signals: {len(data[data['signal'] == 0])}")
        print("---------------------")

        return data
    