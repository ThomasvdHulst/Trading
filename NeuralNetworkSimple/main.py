

from config import Config
from data_manager import DataManager
from lstm_model import LSTMModel
from strategy import LSTMTradingStrategy
from backtester import LSTMBacktester
from visualizer import LSTMVisualizer

def main():
    config = Config()

    print("LSTM Trading Strategy! ")
    print(f"Symbol: {config.SYMBOL}")
    print(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Commission rate: {config.COMMISSION_PER_TRADE:.2%}")
    print(f"Bid-ask spread: {config.BID_ASK_SPREAD_PCT:.2%}")
    print(f"Slippage: {config.SLIPPAGE_PCT:.2%}")
    print("\n")

    # Initialize components
    data_manager = DataManager(config)
    lstm_model = LSTMModel(config)
    strategy = LSTMTradingStrategy(config)
    backtester = LSTMBacktester(config)
    visualizer = LSTMVisualizer(config)

    # Download and prepare data
    data = data_manager.download_data()
    data = data_manager.prepare_lstm_features(data)
    data_manager.print_data_summary()

    # Split data into training and testing sets
    train_size = int(len(data) * config.TRAIN_TEST_SPLIT)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    # Train the model
    history = lstm_model.train_model(train_data, data_manager.get_feature_columns())

    # Evaluate the model on test set
    evaluation = lstm_model.evaluate_model(test_data)

    # Plot training history
    visualizer.plot_training_history(history)

    # Set model to strategy and generate signals
    strategy.set_lstm_model(lstm_model)
    signals = strategy.generate_signals(test_data)
    strategy.print_signal_summary(signals)

    # Run backtest
    final_data = backtester.run_backtest(signals)
    backtester.print_backtest_results()
    backtester.get_trade_analysis()

    # Plot final results
    visualizer.plot_lstm_results(final_data, evaluation)

if __name__ == "__main__":
    main()