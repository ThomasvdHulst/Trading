""" This file contains a simple moving average trading strategy. In this strategy, 
each day we compare the short moving average and the long moving average. If the short
moving average is greater than the long moving average, we go long, signaling a positive momentum.
If the short moving average is less than the long moving average, we go short, signaling a negative momentum.
"""

# Import used libraries
from data_manager import DataManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MovingAverageStrategy:
    """ This class contains the moving average strategy """

    def __init__(self, small_window: int = 5, long_window: int = 20, init_capital: float = 100000):
        """ Initialize the moving average strategy 
        
        Args:
            small_window: the window size for the short moving average
            long_window: the window size for the long moving average
            init_capital: the initial capital
        """
        self.small_window = small_window
        self.long_window = long_window
        self.init_capital = init_capital
        self.results = {}
    

    def calculate_signals(self, data):
        """ Calculate the signals based on the moving average. 
        
        Args:
            data: the data to calculate the signals for

        Returns:
            signals: the signals and positions as calculated by the strategy
        """
        
        # Initialize dataframe to store the signals
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']

        # Calculate the moving averages
        signals['short_ma'] = data['close'].rolling(window=self.small_window).mean()
        signals['long_ma'] = data['close'].rolling(window=self.long_window).mean()

        # Generate the signals based on the moving averages
        # Here 1 denotes that we should buy and 0 denotes that we should sell
        signals['signal'] = 0
        signals.iloc[self.small_window:, signals.columns.get_loc('signal')] = np.where(
            signals['short_ma'].iloc[self.small_window:] > signals['long_ma'].iloc[self.small_window:],
            1,
            0
        )

        # The position is the difference between the signal and the previous signal
        signals['position'] = signals['signal'].diff()

        return signals



    def backtest(self, data):
        """ Backtest the strategy 
        
        Args:
            data: the data to backtest the strategy on

        Returns:
            results: the results of the backtest
        """
        
        # Calculate the signals
        signals = self.calculate_signals(data)

        # Initialize a portfolio dataframe to store the portfolio values
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['cash'] = 0.0
        portfolio['shares'] = 0.0
        portfolio['holdings_value'] = 0.0
        portfolio['total'] = 0.0
        
        # Initialize with lists to track values
        cash_values = []
        shares_values = []
        holdings_values = []
        total_values = []
        
        # Start with the first day
        cash_values.append(self.init_capital)
        shares_values.append(0.0)
        holdings_values.append(0.0)
        total_values.append(self.init_capital)

        # Process each signal and update the portfolio
        for i in range(1, len(signals)):
            # Get the current price and position change
            curr_price = signals['price'].iloc[i]
            position_change = signals['position'].iloc[i]

            # Obtain the previous values
            prev_cash = cash_values[i-1]
            prev_shares = shares_values[i-1]

            # Depending on the position change, update the portfolio
            if position_change == 1: # Buy signal, so invest all cash
                new_shares = prev_cash / curr_price if prev_cash > 0 else 0
                new_cash = 0.0

            elif position_change == -1: # Sell signal, so sell all shares
                new_cash = prev_shares * curr_price
                new_shares = 0.0

            else: # Hold, so just keep the same position
                new_cash = prev_cash
                new_shares = prev_shares

            # Calculate holdings and total value
            holdings_value = new_shares * curr_price
            total_value = new_cash + holdings_value

            # Store all values
            cash_values.append(new_cash)
            shares_values.append(new_shares)
            holdings_values.append(holdings_value)
            total_values.append(total_value)

        # Assign back to portfolio DataFrame
        portfolio['cash'] = cash_values
        portfolio['shares'] = shares_values
        portfolio['holdings_value'] = holdings_values
        portfolio['total'] = total_values
        
        # Calculate the returns
        portfolio['returns'] = portfolio['total'].pct_change()

        # For comparison, calculate the buy and hold return
        buy_hold = self.init_capital * (data['close'] / data['close'].iloc[0])

        # Performance metrics
        total_return = (portfolio['total'].iloc[-1] - self.init_capital) / self.init_capital * 100
        buy_hold_return = (buy_hold.iloc[-1] - self.init_capital) / self.init_capital * 100

        # Count trades
        trades = signals[signals['position'] != 0]['position']
        num_trades = len(trades)
        
        # Calculate Sharpe ratio
        returns = portfolio['returns'].dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.abs().clip(lower=1) * 100
        max_drawdown = drawdown.min()
        
        # Win rate calculation
        winning_trades = 0
        losing_trades = 0
        entry_price = None
        
        for i, row in signals.iterrows():
            if row['position'] == 1: # Buy signal
                entry_price = row['price']
            elif row['position'] == -1 and entry_price: # Sell signal
                if row['price'] > entry_price: # Profit if sell price > buy price
                    winning_trades += 1
                else:
                    losing_trades += 1
                entry_price = None
        
        win_rate = winning_trades / (winning_trades + losing_trades) * 100 if (winning_trades + losing_trades) > 0 else 0
        
        # Store results
        results = {
            'signals': signals,
            'portfolio': portfolio,
            'buy_hold': buy_hold,
            'metrics': {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': num_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'final_value': portfolio['total'].iloc[-1]
            }
        }

        self.results = results
        
        return results


    def print_results(self):
        """ Show the results of the backtest """

        if not self.results:
            print("No results available")
            return

        metrics = self.results['metrics']

        print("---------------------")
        print("Backtest results:")
        print(f"Initial capital: {self.init_capital:.2f}")
        print(f"Final value: {metrics['final_value']:.2f}")
        print(f"Total return: {metrics['total_return']:.2f}%")
        print(f"Number of trades: {metrics['num_trades']}")
        print(f"Winning trades: {metrics['winning_trades']}")
        print(f"Losing trades: {metrics['losing_trades']}")
        print(f"Win rate: {metrics['win_rate']:.2f}%")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        print("---------------------")


    def plot_results(self):
        """ Plot the results of the backtest """
        
        if not self.results:
            print("No results available")
            return

        signals = self.results['signals']
        portfolio = self.results['portfolio']
        buy_hold = self.results['buy_hold']

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Price and moving averages
        axes[0].plot(signals.index, signals['price'], label='Close Price')
        axes[0].plot(signals.index, signals['short_ma'], label='Short MA')
        axes[0].plot(signals.index, signals['long_ma'], label='Long MA')

        # Mark buy signals
        buy_signals = signals[signals['position'] == 1]
        axes[0].scatter(buy_signals.index, buy_signals['price'], marker='^', color='g', label='Buy')

        # Mark sell signals
        sell_signals = signals[signals['position'] == -1]
        axes[0].scatter(sell_signals.index, sell_signals['price'], marker='v', color='r', label='Sell')
        
        axes[0].set_title('Price and Trading Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Portfolio value
        axes[1].plot(portfolio.index, portfolio['total'], label='Strategy Value', linewidth=2)
        axes[1].plot(portfolio.index, buy_hold, label='Buy & Hold')
        axes[1].set_title('Portfolio Value')
        axes[1].set_ylabel('Value ($)')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 3: cash vs holdings
        axes2[0].fill_between(portfolio.index, 0, portfolio['cash'], label='Cash', alpha=0.7, color='lightblue')
        axes2[0].fill_between(portfolio.index, portfolio['cash'], portfolio['total'], label='Holdings Value', alpha=0.7, color='lightgreen')
        axes2[0].set_title('Portfolio Composition (Cash vs Holdings)')
        axes2[0].set_ylabel('Value ($)')
        axes2[0].legend(loc='best')
        axes2[0].grid(True, alpha=0.3)

        # Plot 4: returns
        axes2[1].plot(portfolio.index, portfolio['returns'] * 100, label='Strategy Returns', alpha=0.7)
        axes2[1].set_title('Daily Returns')
        axes2[1].set_ylabel('Returns (%)')
        axes2[1].legend(loc='best')
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    # Initialize the data manager
    dm = DataManager()

    # Set the strategy parameters
    small_window = 5
    long_window = 20
    init_capital = 100000

    # Get the data
    data = dm.get_data('AAPL', '2024-01-01', '2024-12-31', '1Day')
    
    if data.empty:
        print("No data was found, please check the data manager and the inputs.")
        return

    print("Data was found:")
    print(data.head())

    print("\nStarting the moving average strategy!")
    strategy = MovingAverageStrategy(small_window=small_window, long_window=long_window, init_capital=init_capital)
    results = strategy.backtest(data)

    strategy.print_results()
    strategy.plot_results()



if __name__ == "__main__":
    main()