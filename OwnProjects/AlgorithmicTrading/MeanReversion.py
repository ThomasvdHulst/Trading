""" This file contains the mean reversion trading strategy. In this strategy, we use the z-score to determine when to enter and exit positions. """

# Import used libraries
from data_manager import DataManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanReversionStrategy:
    """ This class contains the mean reversion strategy """

    def __init__(self, lookback_period: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5, init_capital: float = 100000):
        """ Initialize the mean reversion strategy """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.init_capital = init_capital
        self.results = {}

    
    def calculate_indicators(self, data):
        """ Calculate the indicators for the strategy 
        
        Args:
            data: the data to calculate the indicators for

        Returns:
            data: the data with the indicators
        """

        data = data.copy()

        # Calculate the indicators
        data['rolling_mean'] = data['close'].rolling(window=self.lookback_period).mean()
        data['rolling_std'] = data['close'].rolling(window=self.lookback_period).std()

        # Calculate the z-score
        data['z_score'] = (data['close'] - data['rolling_mean']) / data['rolling_std']

        return data

    
    def generate_signals(self, data):
        """ Generate the signals for the strategy 
        
        Args:
            data: the data to generate the signals for

        Returns:
            data: the data with the signals
        """

        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['z_score'] = data['z_score']

        # Initialize the signals
        signals['signal'] = 0
        signals['position'] = 0

        # Generate the signals with proper state management
        current_position = 0  # 0 = no position, 1 = long position
        
        for i in range(len(signals)):
            z_score = signals['z_score'].iloc[i]
            
            if current_position == 0:  # Currently no position
                if z_score < -self.entry_threshold:  # Price significantly below mean
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    current_position = 1
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    
            elif current_position == 1:  # Currently holding long position
                if z_score > -self.exit_threshold:  # Price significantly above mean - SELL
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    current_position = 0
                else:  # Hold position
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1

        signals['signal'] = signals['signal'].fillna(0)

        # The position is the difference between the signal and the previous signal
        signals['position'] = signals['signal'].diff().fillna(0)

        return signals

    
    def backtest(self, data):
        """ Backtest the strategy """
        
        data = self.calculate_indicators(data)
        signals = self.generate_signals(data)
        
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
                new_shares = prev_cash / curr_price if prev_cash > 0 else prev_shares
                new_cash = 0.0

            elif position_change == -1: # Sell signal, so sell all shares
                new_cash = prev_cash + prev_shares * curr_price
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
            'data': data,  # Store the data with indicators
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
        data = self.results['data']

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Price, rolling mean, and z-score bands
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        axes[0].plot(data.index, data['rolling_mean'], label='Rolling Mean', color='blue', linewidth=2)
        
        # Calculate and plot z-score bands around the rolling mean
        upper_band = data['rolling_mean'] + (self.entry_threshold * data['rolling_std'])
        lower_band = data['rolling_mean'] - (self.entry_threshold * data['rolling_std'])
        exit_upper_band = data['rolling_mean'] + (self.exit_threshold * data['rolling_std'])
        exit_lower_band = data['rolling_mean'] - (self.exit_threshold * data['rolling_std'])
        
        axes[0].plot(data.index, upper_band, label=f'Upper Band (+{self.entry_threshold}$\sigma$)', color='red', linestyle='--', alpha=0.7)
        axes[0].plot(data.index, lower_band, label=f'Lower Band (-{self.entry_threshold}$\sigma$)', color='red', linestyle='--', alpha=0.7)
        axes[0].plot(data.index, exit_upper_band, label=f'Exit Upper (+{self.exit_threshold}$\sigma$)', color='orange', linestyle=':', alpha=0.7)
        axes[0].plot(data.index, exit_lower_band, label=f'Exit Lower (-{self.exit_threshold}$\sigma$)', color='orange', linestyle=':', alpha=0.7)
        
        # Fill the area between entry bands
        axes[0].fill_between(data.index, lower_band, upper_band, alpha=0.1, color='red', label='Entry Zone')
        
        # Mark buy signals
        buy_signals = signals[signals['position'] == 1]
        axes[0].scatter(buy_signals.index, buy_signals['price'], marker='^', color='g', s=50, label='Buy', zorder=5)

        # Mark sell signals
        sell_signals = signals[signals['position'] == -1]
        axes[0].scatter(sell_signals.index, sell_signals['price'], marker='v', color='r', s=50, label='Sell', zorder=5)
        
        axes[0].set_title('Price, Rolling Mean, and Z-Score Bands')
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

    # Get the data
    data = dm.get_data('AAPL', '2024-01-01', '2024-12-31', '1Day')
    
    if data.empty:
        print("No data was found, please check the data manager and the inputs.")
        return
    
    print("Data was found:")
    print(data.head())
    
    # Set the strategy parameters
    lookback_period = 20
    entry_threshold = 2
    exit_threshold = 0.5
    init_capital = 100000

    print("\nStarting the mean reversion strategy!")

    strategy = MeanReversionStrategy(lookback_period, entry_threshold, exit_threshold, init_capital)
    results = strategy.backtest(data)

    strategy.print_results()
    strategy.plot_results()


if __name__ == "__main__":
    main()