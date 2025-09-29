""" This file contains the volatility breakout trading strategy. In this strategy, we use the volatility to determine when to enter and exit positions. """

# Import used libraries
from data_manager import DataManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VolatilityBreakoutStrategy:
    """ This class contains the volatility breakout strategy """
    
    def __init__(self, lookback_period: int = 20, entry_multiplier: float = 1.0,
    tier_sizes: list = None, cash_reserve: float = 0.30,
    max_position_size: float = 0.15, init_capital: float = 100000):
        """ Initialize the volatility breakout strategy 
        
        Args:
            lookback_period: the lookback period for the volatility calculation
            entry_multiplier: the multiplier for the entry price
            tier_sizes: the sizes of the tiers
            cash_reserve: the cash reserve
            max_position_size: the maximum position size
            init_capital: the initial capital
        """

        self.lookback_period = lookback_period
        self.entry_multiplier = entry_multiplier
        self.tier_sizes = tier_sizes if tier_sizes is not None else [0.05, 0.05, 0.05]
        self.cash_reserve = cash_reserve
        self.max_position_size = max_position_size
        self.init_capital = init_capital
        self.results = {}


    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
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

        # Calculate the entry thresholds
        data['entry_threshold'] = data['rolling_mean'] + (self.entry_multiplier * data['rolling_std'])

        # Calculate the strength score for position sizing
        data['strength_score'] = (data['close'] - data['rolling_mean']) / data['rolling_std']

        return data


    def calculate_position_size(self, strength_score: float, available_cash_ratio: float) -> float:
        """ Calculate the position size based on signal strength and available cash.
        
        Args:
            strength_score: the strength score for the position sizing
            available_cash_ratio: the available cash ratio for the position sizing

        Returns:
            position_size: fraction of portfolio to invest
        """

        if strength_score <= self.entry_multiplier:
            return 0.0

        # Determine tier based on strength score
        tier_thresholds = [self.entry_multiplier, 1.5, 2.0]
        cumulative_size = 0

        for i, threshold in enumerate(tier_thresholds):
            if strength_score > threshold and i < len(self.tier_sizes):
                cumulative_size += self.tier_sizes[i]

        # Cap at maximum position size
        position_size = min(cumulative_size, self.max_position_size)

        # Ensure we do not exceed the available cash
        max_investable = available_cash_ratio - self.cash_reserve
        position_size = min(position_size, max_investable)

        return max(0.0, position_size)
        

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Generate the signals for the strategy 
        
        Args:
            data: the data to generate the signals for

        Returns:
            signals: the data with the signals
        """

        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['entry_threshold'] = data['entry_threshold']
        signals['strength_score'] = data['strength_score']
        signals['rolling_mean'] = data['rolling_mean']

        # Initialize the signals
        signals['signal'] = 0.0 # Represents position size as fraction of portfolio
        signals['position'] = 0.0 # Position change
        signals['in_position'] = False # Boolean to track if in position

        # Track portfolio state for position sizing
        current_cash_ratio = 1.0 # 1.0 = 100% cash, 0.0 = 100% invested
        current_position_size = 0.0 # Fraction of portfolio invested
        in_position = False # Start with no position

        for i in range(len(signals)):
            price = signals['price'].iloc[i]
            strength = signals['strength_score'].iloc[i]
            entry_threshold = signals['entry_threshold'].iloc[i]
            rolling_mean = signals['rolling_mean'].iloc[i]

            if not in_position: # If not in position, check if we should enter
                if price > entry_threshold: # Price broke above entry threshold
                    position_size = self.calculate_position_size(strength, current_cash_ratio)
                    if position_size > 0:
                        signals.iloc[i, signals.columns.get_loc('signal')] = position_size
                        signals.iloc[i, signals.columns.get_loc('in_position')] = True
                        current_position_size = position_size
                        current_cash_ratio -= position_size # Reduce available cash
                        in_position = True
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 0.0
                        signals.iloc[i, signals.columns.get_loc('in_position')] = False

                else: # Price did not break above entry threshold
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0.0
                    signals.iloc[i, signals.columns.get_loc('in_position')] = False

            else: # If in position, check if we should exit
                if price <= rolling_mean: # Price broke below rolling mean
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0.0
                    signals.iloc[i, signals.columns.get_loc('in_position')] = False
                    current_cash_ratio += current_position_size # Add back the position size
                    current_position_size = 0.0
                    in_position = False
                else: # Price did not break below rolling mean
                    signals.iloc[i, signals.columns.get_loc('signal')] = current_position_size
                    signals.iloc[i, signals.columns.get_loc('in_position')] = True

        # Calculate position changes for trading
        signals['prev_signal'] = signals['signal'].shift(1).fillna(0)
        signals['position'] = signals['signal'] - signals['prev_signal']

        return signals


    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Backtest the strategy 
        
        Args:
            data: the data to backtest the strategy on

        Returns:
            results: the results of the backtest
        """

        data = self.calculate_indicators(data)
        signals = self.generate_signals(data)

        # Initialize a portfolio dataframe to store the portfolio values
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['cash'] = 0.0
        portfolio['shares'] = 0.0
        portfolio['holdings_value'] = 0.0
        portfolio['total'] = 0.0
        portfolio['position_size'] = 0.0 # Track current position size

        # Initialize with lists to track values
        cash_values = []
        shares_values = []
        holdings_values = []
        total_values = []
        position_sizes = []

        # Start with the first day
        cash_values.append(self.init_capital)
        shares_values.append(0.0)
        holdings_values.append(0.0)
        total_values.append(self.init_capital)
        position_sizes.append(0.0)

        # Process each signal and update the portfolio
        for i in range(1, len(signals)):
            # Get the current price and position change
            curr_price = signals['price'].iloc[i]
            position_change = signals['position'].iloc[i]
            target_signal = signals['signal'].iloc[i]

            # Obtain the previous values
            prev_cash = cash_values[i-1]
            prev_shares = shares_values[i-1]
            prev_total = total_values[i-1]

            # Calculate current holdings value with updated price
            current_holdings_value = prev_shares * curr_price
            current_total_before_trade = prev_cash + current_holdings_value

            if position_change > 0: # Increase position (buy)
                # Calculate dollar amount to invest
                dollar_amount = position_change * current_total_before_trade
                if dollar_amount > prev_cash:
                    dollar_amount = prev_cash # Cannot invest more than available cash

                if dollar_amount > 0:
                    new_shares_bought = dollar_amount / curr_price
                    new_shares = prev_shares + new_shares_bought
                    new_cash = prev_cash - dollar_amount
                else:
                    new_shares = prev_shares
                    new_cash = prev_cash

            elif position_change < 0: # Decrease position (sell)
                # Calculate how much to sell
                sell_ratio = abs(position_change) / (position_sizes[i-1] if position_sizes[i-1] > 0 else 1)
                shares_to_sell = prev_shares * sell_ratio

                new_shares = prev_shares - shares_to_sell
                new_cash = prev_cash + shares_to_sell * curr_price

            else: # Hold, so keep the same position
                new_shares = prev_shares
                new_cash = prev_cash

            # Calculate holdings and total value
            holdings_value = new_shares * curr_price
            total_value = new_cash + holdings_value

            # Store all values
            cash_values.append(new_cash)
            shares_values.append(new_shares)
            holdings_values.append(holdings_value)
            total_values.append(total_value)
            position_sizes.append(target_signal)

        # Assign back to portfolio DataFrame
        portfolio['cash'] = cash_values
        portfolio['shares'] = shares_values
        portfolio['holdings_value'] = holdings_values
        portfolio['total'] = total_values
        portfolio['position_size'] = position_sizes

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
        entry_total_value = None

        for i, row in signals.iterrows():
            if row['position'] > 0: # Entry signal
                entry_price = row['price']
                entry_idx = signals.index.get_loc(i)
                entry_total_value = portfolio['total'].iloc[entry_idx] if entry_idx < len(portfolio) else None
            elif row['position'] < 0 and entry_price: # Exit signal
                exit_idx = signals.index.get_loc(i)
                if exit_idx < len(portfolio) and entry_total_value:
                    exit_total_value = portfolio['total'].iloc[exit_idx] if exit_idx < len(portfolio) else None
                    if exit_total_value > entry_total_value: # Profit if exit price > entry price
                        winning_trades += 1
                    else:
                        losing_trades += 1
                entry_price = None
                entry_total_value = None

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
        print("Volatility Breakout Strategy Results:")
        print(f"Initial capital: {self.init_capital:.2f}")
        print(f"Final value: {metrics['final_value']:.2f}")
        print(f"Total return: {metrics['total_return']:.2f}%")
        print(f"Buy & Hold return: {metrics['buy_hold_return']:.2f}%")
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

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Plot 1: Price, rolling mean, and entry threshold
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        axes[0].plot(data.index, data['rolling_mean'], label='Rolling Mean', color='blue', linewidth=2)
        axes[0].plot(data.index, data['entry_threshold'], label=f'Entry Threshold (Mean + {self.entry_multiplier}σ)', 
                    color='red', linestyle='--', alpha=0.7)
        
        # Mark buy signals with different colors for different position sizes
        buy_signals = signals[signals['position'] > 0]
        if not buy_signals.empty:
            # Color by position size
            scatter = axes[0].scatter(buy_signals.index, buy_signals['price'], 
                                    c=buy_signals['signal'], cmap='Greens', 
                                    s=100, marker='^', label='Buy (Size by Color)', 
                                    zorder=5, alpha=0.8)
            plt.colorbar(scatter, ax=axes[0], label='Position Size')

        # Mark sell signals
        sell_signals = signals[signals['position'] < 0]
        if not sell_signals.empty:
            axes[0].scatter(sell_signals.index, sell_signals['price'], marker='v', 
                          color='r', s=50, label='Sell', zorder=5)
        
        axes[0].set_title('Price, Rolling Mean, and Entry Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Portfolio value comparison
        axes[1].plot(portfolio.index, portfolio['total'], label='Strategy Value', linewidth=2)
        axes[1].plot(portfolio.index, buy_hold, label='Buy & Hold', alpha=0.7)
        axes[1].set_title('Portfolio Value Comparison')
        axes[1].set_ylabel('Value ($)')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Position size over time
        axes[2].fill_between(portfolio.index, 0, portfolio['position_size'] * 100, 
                           label='Position Size', alpha=0.6, color='lightgreen')
        axes[2].axhline(y=self.cash_reserve * 100, color='red', linestyle='--', 
                       alpha=0.7, label=f'Cash Reserve ({self.cash_reserve*100}%)')
        axes[2].set_title('Position Size Over Time')
        axes[2].set_ylabel('Position Size (% of Portfolio)')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Second figure for cash/holdings breakdown and returns
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

        # Plot: cash vs holdings
        axes2[0].fill_between(portfolio.index, 0, portfolio['cash'], 
                            label='Cash', alpha=0.7, color='lightblue')
        axes2[0].fill_between(portfolio.index, portfolio['cash'], portfolio['total'], 
                            label='Holdings Value', alpha=0.7, color='lightgreen')
        axes2[0].set_title('Portfolio Composition (Cash vs Holdings)')
        axes2[0].set_ylabel('Value ($)')
        axes2[0].legend(loc='best')
        axes2[0].grid(True, alpha=0.3)

        # Plot: returns
        axes2[1].plot(portfolio.index, portfolio['returns'] * 100, 
                     label='Strategy Returns', alpha=0.7)
        axes2[1].set_title('Daily Returns')
        axes2[1].set_ylabel('Returns (%)')
        axes2[1].set_xlabel('Date')
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
    strategy = VolatilityBreakoutStrategy()
    results = strategy.backtest(data)
    strategy.print_results()
    strategy.plot_results()

if __name__ == "__main__":
    main()