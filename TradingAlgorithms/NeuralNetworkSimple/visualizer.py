""" This file contains the visualizer for the LSTM strategy """

# Import used libraries
import matplotlib.pyplot as plt


class LSTMVisualizer:

    def __init__(self, config):
        self.config = config
        
    def plot_lstm_results(self, data, evaluation_results=None):
        """ Create a visualization of the LSTM strategy """

        n_plots = 5

        # Initialize a figure with the number of plots
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 2*n_plots))
        fig.suptitle(f'{self.config.SYMBOL} LSTM Trading Strategy Results', fontsize=14)
        
        # Plot 1: Price and predictions
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label=f'{self.config.SYMBOL} Price', color='blue')
        
        if 'predicted_return' in data.columns:
            # Show predicted direction as colored background
            for i in range(len(data)):
                if data.iloc[i]['predicted_return'] > 0:
                    ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.1, color='green')
                else:
                    ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.1, color='red')
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price with LSTM Predictions (Green=Bullish, Red=Bearish)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Prediction confidence and signals
        if 'prediction_confidence' in data.columns:
            ax2 = axes[1]
            ax2.plot(data.index, data['prediction_confidence'], label='Prediction Confidence', color='purple')
            ax2.axhline(y=self.config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', label='Confidence Threshold')
            
            # Set trading signals
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            if not buy_signals.empty:
                ax2.scatter(buy_signals.index, buy_signals['prediction_confidence'], color='green', marker='^', s=10, label='Buy Signals')
            if not sell_signals.empty:
                ax2.scatter(sell_signals.index, sell_signals['prediction_confidence'], color='red', marker='v', s=10, label='Sell Signals')
            
            ax2.set_ylabel('Confidence')
            ax2.set_title('Prediction Confidence and Trading Signals')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Positions and trades
        ax3 = axes[2] if 'prediction_confidence' in data.columns else axes[1]
        ax3.plot(data.index, data['close'], label=f'{self.config.SYMBOL} Price', color='blue')
        
        # Color background based on position
        long_positions = data['position'] == 1
        if long_positions.any():
            ax3.fill_between(data.index, ax3.get_ylim()[0], ax3.get_ylim()[1], where=long_positions, alpha=0.2, color='green', label='Long Position')
        
        ax3.set_ylabel('Price ($)')
        ax3.set_title('Positions Over Time')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Portfolio performance
        ax4 = axes[3] if 'prediction_confidence' in data.columns else axes[2]
        ax4.plot(data.index, data['portfolio_value'], label='Portfolio Value', color='purple', linewidth=2)
        ax4.axhline(y=self.config.INITIAL_CAPITAL, color='black', linestyle='--', label='Initial Capital')
        
        # Calculate and show buy and hold comparison
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price
        buy_hold_value = self.config.INITIAL_CAPITAL * (1 + buy_hold_return)
        
        ax4.axhline(y=buy_hold_value, color='orange', linestyle=':', label=f'Buy & Hold (${buy_hold_value:,.0f})')
        
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.set_title('Portfolio Performance vs Buy & Hold')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: Model evaluation (if available)
        if evaluation_results:
            ax5 = axes[4]
            predictions = evaluation_results['predictions']
            actual = evaluation_results['actual']
            
            # Plot actual vs predicted
            ax5.scatter(actual, predictions, alpha=0.5, s=10)
            
            # Add perfect prediction line
            min_val = min(min(actual), min(predictions))
            max_val = max(max(actual), max(predictions))
            ax5.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax5.set_xlabel('Actual Returns')
            ax5.set_ylabel('Predicted Returns')
            ax5.set_title(f'Prediction Accuracy (Direction: {evaluation_results["direction_accuracy"]:.1%})')
            ax5.legend()
            ax5.grid(True)
        
        plt.tight_layout()
        plt.show()
    

    def plot_training_history(self, history):
        """ Plot the LSTM training history """

        # Check if there is a history to plot
        if history is None:
            return
            
        # Initialize a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training metrics
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()