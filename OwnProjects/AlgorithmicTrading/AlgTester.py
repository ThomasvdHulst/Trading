""" This is the main file where all algorithmic trading strategies are tested """

# Import used libraries
from MovingAverage import MovingAverageStrategy
from MeanReversion import MeanReversionStrategy
from data_manager import DataManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VolatilityBreakout import VolatilityBreakoutStrategy


class StrategyTester:
    """ This class compares the performance of different strategies """

    def __init__(self, init_capital: float = 100000):
        """ Initialize the strategy tester """
        self.init_capital = init_capital
        self.results = {}
        self.strategies = {}

    
    def add_strategy(self, name: str, strategy: object) -> None:
        """ Add a strategy to the tester 
        
        Args:
            name: the name of the strategy
            strategy: the strategy object

        Returns:
            None
        """
        self.strategies[name] = strategy


    def run_comparison(self, data: pd.DataFrame) -> None:
        """ Run the comparison of the strategies 
        
        Args:
            data: the data to run the comparison on
        """

        print("Running comparison of strategies...")
        print(f"Data length: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Number of trading days: {len(data)}")
        print(f"Initial capital: {self.init_capital}")

        # Calculate buy and hold benchmark
        buy_hold_return = self.init_capital * (data['close'] / data['close'].iloc[0])
        bh_returns = buy_hold_return.pct_change().dropna()
        bh_sharpe = np.sqrt(252) * bh_returns.mean() / bh_returns.std() if bh_returns.std() > 0 else 0
        bh_cumulative = (1 + bh_returns).cumprod()
        bh_running_max = bh_cumulative.expanding().max()
        bh_drawdown = (bh_cumulative - bh_running_max) / bh_running_max.abs().clip(lower=1) * 100
        bh_max_drawdown = bh_drawdown.min()

        # Store results
        self.results['Buy & Hold'] = {
            'portfolio_value': buy_hold_return,
            'final_value': buy_hold_return.iloc[-1],
            'total_return': (buy_hold_return.iloc[-1] - self.init_capital) / self.init_capital * 100,
            'returns': bh_returns,
            'metrics': {
                'total_return': (buy_hold_return.iloc[-1] - self.init_capital) / self.init_capital * 100,
                'final_value': buy_hold_return.iloc[-1],
                'sharpe_ratio': bh_sharpe,
                'max_drawdown': bh_max_drawdown,
                'num_trades': 1,
                'win_rate': 0
            }
        }

        # Run each strategy
        for name, strategy in self.strategies.items():
            print(f"Running {name}...")

            # Run the backtest
            strategy_results = strategy.backtest(data)

            # Store results
            self.results[name] = {
                'portfolio_value': strategy_results['portfolio']['total'],
                'final_value': strategy_results['metrics']['final_value'],
                'total_return': strategy_results['metrics']['total_return'],
                'returns': strategy_results['portfolio']['returns'],
                'metrics': strategy_results['metrics']
            }

            print(f"Completed {name}, final value: ${self.results[name]['final_value']:.2f}")

        print("---------------------------")


    def print_results(self):
        pass


    def plot_results(self):
        """ Plot the results of the strategies """

        if not self.results:
            print("No results available")
            return 

        # Create the plots
        plt.figure(figsize=(12, 8))

        # Plot 1: Portfolio value over time
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

        for i, (name, result) in enumerate(self.results.items()):
            color = colors[i]
            plt.plot(result['portfolio_value'].index, result['portfolio_value'], label=name, color=color)

        plt.title('Portfolio value over time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    
    # Initialize the data manager
    dm = DataManager()
    
    # Set data parameters
    symbol = 'AAPL'
    start_date = '2024-10-01'
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
    
    # Initialize the strategy tester
    comparison = StrategyTester(init_capital)

    # Create and add the strategies
    print("Creating and adding the strategies...")

    # Moving Average Strategy
    ma_strategy = MovingAverageStrategy(
        small_window=5,
        long_window=20,
        init_capital=init_capital
    )
    comparison.add_strategy('Moving Average (5, 20)', ma_strategy)

    # Mean Reversion Strategy
    mr_strategy = MeanReversionStrategy(
        lookback_period=20,
        entry_threshold=1,
        exit_threshold=1,
        init_capital=init_capital
    )
    comparison.add_strategy('Mean Reversion (20, 1, 1)', mr_strategy)

    # Volatility Breakout Strategy
    vb_strategy = VolatilityBreakoutStrategy(
        lookback_period=20,
        entry_multiplier=1.0,
        tier_sizes=[0.05, 0.05, 0.05],
        cash_reserve=0.10,
        max_position_size=0.15,
        init_capital=init_capital
    )
    comparison.add_strategy('Volatility Breakout (20, 1.0, [0.05, 0.05, 0.05], 0.10, 0.15)', vb_strategy)


    print(f"Added {len(comparison.strategies)} strategies to the tester")

    # Run the comparison
    print("Running the comparison...")
    comparison.run_comparison(data)

    # Plot the results
    print("Plotting the results...")
    comparison.plot_results()


if __name__ == "__main__":
    main()