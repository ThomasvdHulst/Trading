""" This file is the main execution script for the mean reversion trading strategy with GARCH. This simple script
will download the data, calculate the GARCH model, calculate the indicators, generate the signals, run the backtest
and visualize the results. GARCH is implemented to adjust the entry and exit thresholds based on the volatility and
to adjust the z-score.
"""

# Import used libraries
from config import Config
from garch_model import GARCHModel
from data_manager import DataManager
from strategy import MeanReversionStrategy
from backtester import Backtester
from visualizer import Visualizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    # Initialize configuration
    config = Config()
    
    print("Mean Reversion Trading Strategy with GARCH")
    print(f"Symbol: {config.SYMBOL}")
    print(f"Lookback Period: {config.LOOKBACK_PERIOD}")
    print(f"Entry Threshold: {config.ENTRY_THRESHOLD}")
    print(f"Exit Threshold: {config.EXIT_THRESHOLD}")

    # Initialize data manager
    data_manager = DataManager(config)
    data = data_manager.download_data()

    # Initialize GARCH model and calculate rolling GARCH forecast
    garch_model = GARCHModel(config)
    data = garch_model.rolling_garch_forecast(data)
    garch_model.print_garch_summary(data)

    # Initialize strategy and calculate indicators
    strategy = MeanReversionStrategy(config)
    data = strategy.calculate_indicators(data)

    # Generate signals and print signal summary
    data = strategy.generate_signals(data)
    strategy.print_signal_summary(data)

    # Run backtest
    backtester = Backtester(config)
    data = backtester.run_backtest(data)
    backtester.print_results()

    # Visualize results
    visualizer = Visualizer(config)
    visualizer.plot_strategy_results(data)
    plt.show()


if __name__ == "__main__":
    main()