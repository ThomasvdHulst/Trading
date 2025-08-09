""" This file contains the LSTM model for the neural network trading strategy """

# Import used libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


class LSTMModel:
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_columns = None
        self.is_trained = False


    def prepare_sequences(self, data, feature_columns):
        """ Prepare sequences for LSTM model """

        self.feature_columns = feature_columns

        # Prepare features and target
        features = data[feature_columns].values
        target = data['target'].values.reshape(-1, 1)

        # Scale the data
        features_scaled = self.scaler_features.fit_transform(features)
        target_scaled = self.scaler_target.fit_transform(target)

        # Create sequences
        X = []
        y = []
        sequence_length = self.config.SEQUENCE_LENGTH

        # Iterate over the data
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        return X, y
    

    def build_model(self, input_shape):
        """ Build the LSTM model """

        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=self.config.LSTM_UNITS,
            return_sequences=True, # Return sequences for next layer (needed for next LSTM layer)
            input_shape=input_shape
        ))
        model.add(Dropout(self.config.DROPOUT_RATE))

        # Second LSTM layer
        model.add(LSTM(
            units=self.config.LSTM_UNITS,
            return_sequences=False # Do not return sequences for next layer (not needed for the Dense layer)
        ))
        model.add(Dropout(self.config.DROPOUT_RATE))

        # Dense layers
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1, activation='linear'))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        return model
    

    def train_model(self, data, feature_columns):
        """ Train the LSTM model """

        print("Training the model...")

        # Prepare sequences
        X, y = self.prepare_sequences(data, feature_columns)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # Split train/validation
        train_size = int(len(X) * (1 - self.config.VALIDATION_SPLIT))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Build the model
        input_shape = (self.config.SEQUENCE_LENGTH, len(feature_columns))
        self.model = self.build_model(input_shape)
        print(f"Model input shape: {input_shape}")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', # Watch the validation loss
            patience=10, # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True # Restore the best weights after training
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', # Watch the validation loss
            factor=0.5, # Reduce learning rate by half
            patience=5, # Number of epochs with no improvement after which learning rate will be reduced
            min_lr=0.0001 # Minimum learning rate
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        self.is_trained = True

        # Print training results
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]
        print(f"Training completed!")
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Final training MAE: {final_train_mae:.4f}")
        print(f"Final validation MAE: {final_val_mae:.4f}")
        print("\n")

        return history
    

    def predict_sequences(self, data):
        """ Make predictions on new data """

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features = data[self.feature_columns].values
        features_scaled = self.scaler_features.transform(features)

        # Create sequences
        X = []
        sequence_length = self.config.SEQUENCE_LENGTH

        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])

        X = np.array(X)

        # Make predictions
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler_target.inverse_transform(predictions_scaled)

        return predictions.flatten()
    

    def evaluate_model(self, data):
        """ Evaluate the model performance """

        if not self.is_trained:
            print("Model is not trained yet....")
            return
        
        # Get predictions
        predictions = self.predict_sequences(data)

        # Get actual values
        actual = data['target'].iloc[self.config.SEQUENCE_LENGTH:].values

        if len(predictions) != len(actual):
            min_len = min(len(predictions), len(actual))
            predictions = predictions[:min_len]
            actual = actual[:min_len]
            print(f"Warning: Predictions and actual values have different lengths. Using first {min_len} samples.")

        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)

        # Direction accuracy
        actual_direction = np.sign(actual)
        pred_direction = np.sign(predictions)
        direction_accuracy = np.mean(actual_direction == pred_direction)

        # Print results
        print("Model Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.2%}")
        print("\n")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions,
            'actual': actual
        }
        

    def predict_with_uncertainty(self, data, n_samples=50):
        """ Make predictions with uncertainty estimation using Monte Carlo Dropout """

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features = data[self.feature_columns].values
        features_scaled = self.scaler_features.transform(features)

        # Create sequences
        X = []
        sequence_length = self.config.SEQUENCE_LENGTH

        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])

        X = np.array(X)

        # Enable dropout during inference for uncertainty estimation
        predictions_samples = []
        for i in range(n_samples):
            # Make prediction with dropout enabled (ensured by training=True)
            preds = self.model(X, training=True)
            preds_unscaled = self.scaler_target.inverse_transform(preds.numpy())
            predictions_samples.append(preds_unscaled.flatten())

        predictions_samples = np.array(predictions_samples)

        # Calculate the main and the uncertainty
        mean_predictions = np.mean(predictions_samples, axis=0)
        prediction_std = np.std(predictions_samples, axis=0)

        # The confidence is then the inverse of the uncertainty
        confidence = 1 / (1 + prediction_std * 10)  # Scale factor can be tuned

        return mean_predictions, confidence


