""" This file contains the strategy class for the neural network trading strategy """

# Import used libraries
import pandas as pd
import numpy as np

class LSTMTradingStrategy:

    def __init__(self, config):
        self.config = config
        self.lstm_model = None


    def set_lstm_model(self, lstm_model):
        """ Set the LSTM model """
        self.lstm_model = lstm_model


    def generate_signals(self, data):
        """ Generate signals based on LSTM model predictions """
        
        if self.lstm_model is None:
            raise ValueError("LSTM model is not set, cannot generate signals")
        
        data = data.copy()

        # Get LSTM predictions with uncertainty
        predictions, confidence = self.lstm_model.predict_with_uncertainty(data)

        # Align predictions with data
        pred_start_indx = self.config.SEQUENCE_LENGTH
        data_aligned = data.iloc[pred_start_indx:pred_start_indx + len(predictions)].copy()
        data_aligned['predicted_return'] = predictions

        # Use model uncertainty-based confidence
        data_aligned['prediction_confidence'] = confidence

        # Generate signals
        data_aligned['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        data_aligned['position'] = 0 # 0 = no position, 1 = long, -1 = short

        # Signal logic
        # Buy when the predicted return is positive and the confidence is high
        buy_condition = (
            (data_aligned['predicted_return'] > 0) &
            (data_aligned['prediction_confidence'] > self.config.CONFIDENCE_THRESHOLD)
        )

        # Sell when the predicted return is negative and the confidence is high
        sell_condition = (
            (data_aligned['predicted_return'] < 0) &
            (data_aligned['prediction_confidence'] > self.config.CONFIDENCE_THRESHOLD)
        )

        data_aligned.loc[buy_condition, 'signal'] = 1
        data_aligned.loc[sell_condition, 'signal'] = -1

        # Simple position tracking (hold until opposite signal)
        current_position = 0
        for i in range(len(data_aligned)):
            signal = data_aligned.iloc[i]['signal']

            if signal == 1 and current_position <= 0: # Buy
                current_position = 1
            elif signal == -1 and current_position >= 0: # Sell
                current_position = -1

            # else, hold

            data_aligned.iloc[i, data_aligned.columns.get_loc('position')] = current_position

        return data_aligned
    

    def print_signal_summary(self, data):
        """ Print the signal summary """

        if self.lstm_model is None:
            print("LSTM model is not set, cannot print signal summary")
            return
        
        if 'predicted_return' in data.columns:
            print("Prediction statistics:")
            print(f"Mean predicted return: {data['predicted_return'].mean():.4f}")
            print(f"Median predicted return: {data['predicted_return'].median():.4f}")
            print(f"Std predicted return: {data['predicted_return'].std():.4f}")
            print(f"Max predicted return: {data['predicted_return'].max():.4f}")
            print(f"Min predicted return: {data['predicted_return'].min():.4f}")
            print("\n")

        if 'prediction_confidence' in data.columns:
            print("Confidence statistics:")
            print(f"Mean confidence: {data['prediction_confidence'].mean():.4f}")
            print(f"Median confidence: {data['prediction_confidence'].median():.4f}")
            
            high_confidence_count = (data['prediction_confidence'] > self.config.CONFIDENCE_THRESHOLD).mean() * 100
            print(f"Percentage of high confidence predictions: {high_confidence_count:.2f}%")
            print("\n")
        
        print("Signal statistics:")
        print(f"Total signals: {len(data)}")
        print(f"Buy signals: {len(data[data['signal'] == 1])}")
        print(f"Sell signals: {len(data[data['signal'] == -1])}")
        print(f"Hold signals: {len(data[data['signal'] == 0])}")
        print("\n")