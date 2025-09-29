""" This file contains the backtest for the ML trading system """

# Import used libraries
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import MLTradingStrategy


class Backtester:
    """ Backtester for the ML trading system. Includes transaction costs,
    market impact and proper walk-forward analysis. """
    

    def __init__(self, 
                initial_capital: float = 100000,
                spread_cost: float = 0.0001, # 1 basis point half-spread
                market_impact_const: float = 0.00005, # linear impact cost
                commission_per_share: float = 0.0, # Alpaca has zero commission
                min_trade_interval: int = 15, # Minimum time between trades in minutes
                market_open: str = "09:45", # market open time
                market_close: str = "15:45", # market close time
                ):

        """ Initialize the backtester.
        
        Args:
            initial_capital: Starting capital in USD
            spread_cost: Half-spread as a fraction (0.0001 = 1 basis point)
            market_impact_const: Market impact constant for linear model
            commission_per_share: Commission per share traded
            min_trade_interval: Minimum minutes between trades
            market_open: Time to start trading (avoid market open volatility)
            market_close: Time to stop trading (close positions before market close)
        """

        self.initial_capital = initial_capital
        self.spread_cost = spread_cost
        self.market_impact_const = market_impact_const
        self.commission_per_share = commission_per_share
        self.min_trade_interval = min_trade_interval
        self.market_open = market_open
        self.market_close = market_close

        # Track all results
        self.results = {}
        self.trade_history = []


    def calculate_transaction_costs(self,
                                    price: float,
                                    shares: int,
                                    is_buy: bool,
                                    avg_volume: float) -> float:

        """ Calculate the transaction costs for a trade

        Args:
            price: Execution price
            shares: Number of shares traded
            is_buy: Whether the trade is a buy, True for buy, False for sell
            avg_volume: Average volume for market impact calculation

        Returns:
            Total transaction costs in dollars
        """

        trade_value = price * shares
        
        # Spread costs (pay half-spread)
        spread_costs = trade_value * self.spread_cost

        # Market impact (square-root model for realistic impact)
        # Impact increases with size relative to average volume
        volume_fraction = shares / max(avg_volume, 1)
        market_impact = trade_value * self.market_impact_const * np.sqrt(volume_fraction)

        # Commission
        commission = trade_value * self.commission_per_share

        total_costs = spread_costs + market_impact + commission

        return total_costs


    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """ Check if the timestamp is within trading hours
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if the timestamp is within trading hours, False otherwise
        """

        time_str = timestamp.strftime('%H:%M')
        return self.market_open <= time_str <= self.market_close


    def walk_forward_split(self,
                            data: pd.DataFrame,
                            train_days: int = 60,
                            val_days: int = 10,
                            test_days: int = 20) -> List[Dict]:

        """ Create walk-forward splits for backtesting.

        Args:
            data: Full dataset
            train_days: Number of days for training
            val_days: Number of days for validation
            test_days: Number of days for testing

        Returns:
            List of dictionaries with train/val/test splits
        """

        splits = []

        # Get unique dates
        data['date'] = pd.to_datetime(data.index).date
        unique_dates = data['date'].unique()

        total_days_needed = train_days + val_days + test_days

        # Create splits
        start_idx = 0
        while start_idx + total_days_needed <= len(unique_dates):
            train_end = start_idx + train_days
            val_end = train_end + val_days
            test_end = val_end + test_days

            train_dates = unique_dates[start_idx:train_end]
            val_dates = unique_dates[train_end:val_end]
            test_dates = unique_dates[val_end:test_end]

            split = {
                'train': data[data['date'].isin(train_dates)].drop('date', axis=1),
                'val': data[data['date'].isin(val_dates)].drop('date', axis=1),
                'test': data[data['date'].isin(test_dates)].drop('date', axis=1),
                'period': f"Period_{len(splits)+1}",
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
            }

            splits.append(split)

            # Move forward by test_days
            start_idx += test_days

        # Drop the date column
        data = data.drop('date', axis=1)

        print(f"Created {len(splits)} splits")
        return splits


    def simulate_trading(self,
                        strategy: MLTradingStrategy,
                        test_data: pd.DataFrame,
                        warmup_periods: int = 60) -> Dict:

        """ Simulate trading on test data with trained strategy.
        
        Args:
            strategy: Trained trading strategy
            test_data: Test data for simulation
            warmup_periods: Number of periods to use for warmup

        Returns:
            Dictionary with portfolio values and trade log
        """

        # Initialize portfolio tracking
        portfolio = {
            'timestamp': [],
            'cash': [],
            'shares': [],
            'position_value': [],
            'total_value': [],
        }

        trades = []

        # Start with initial capital
        cash = self.initial_capital
        shares = 0
        last_trade_time = None
        entry_price = None
        highest_value = self.initial_capital
        trailing_stop = None
        stop_loss = None

        # Skip warmup periods
        test_data = test_data.iloc[warmup_periods:]

        # Calculate average volume for market impact calculation
        avg_volume = test_data['volume'].rolling(window=20).mean()

        for i in range(len(test_data)):
            current_time = test_data.index[i]
            current_price = test_data.iloc[i]['close']

            # Check if we're within trading hours
            if not self.is_trading_time(current_time):
                # Close any positions at the end of the day
                if shares > 0 and current_time.strftime('%H:%M') > self.market_close:
                    # Forced exit at market close
                    trade_value = shares * current_price
                    costs = self.calculate_transaction_costs(
                        current_price, shares, False, avg_volume.iloc[i]
                    )
                    cash += trade_value - costs

                    trades.append({
                        'timestamp': current_time,
                        'action': 'SELL_EOD',
                        'price': current_price,
                        'shares': shares,
                        'value': trade_value,
                        'costs': costs,
                        'reason': 'End of day'
                    })

                    shares = 0
                    entry_price = None
                    stop_loss = None
                    trailing_stop = None

                # Record portfolio state
                position_value = shares * current_price
                total_value = cash + position_value
                portfolio['timestamp'].append(current_time)
                portfolio['cash'].append(cash)
                portfolio['shares'].append(shares)
                portfolio['position_value'].append(position_value)
                portfolio['total_value'].append(total_value)

                continue

            # Check minimum trade interval
            can_trade = True
            if last_trade_time is not None:
                time_since_trade = (current_time - last_trade_time).seconds / 60
                if time_since_trade < self.min_trade_interval:
                    can_trade = False

            # Get current position
            current_position = 1 if shares > 0 else 0

            # Generate signals
            # Use a window of data for feature calculation
            window_start = max(0, i - 100)
            data_window = test_data.iloc[window_start:i+1]

            if len(data_window) < 60: # Need minimum data for feature calculation
                position_value = shares * current_price
                total_value = cash + position_value
                portfolio['timestamp'].append(current_time)
                portfolio['cash'].append(cash)
                portfolio['shares'].append(shares)
                portfolio['position_value'].append(position_value)
                portfolio['total_value'].append(total_value)
                continue
            
            signals = strategy.generate_signals(data_window, current_position)
            current_signal = signals.iloc[-1]

            # Check stop-loss and trailing stop
            if shares > 0:
                atr = current_signal['atr']
                stop_loss, new_trailing_stop = strategy.calculate_stops(
                    entry_price, current_price, atr
                )

                # Update trailing stop
                if trailing_stop is None or new_trailing_stop > trailing_stop:
                    trailing_stop = new_trailing_stop

                # Check if stop hit
                if current_price <= stop_loss or current_price <= trailing_stop:
                    # Exit position
                    trade_value = shares * current_price
                    costs = self.calculate_transaction_costs(
                        current_price, shares, False, avg_volume.iloc[i]
                    )
                    cash += trade_value - costs

                    stop_type = 'Stop-loss' if current_price <= stop_loss else 'Trailing stop'
                    trades.append({
                        'timestamp': current_time,
                        'action': 'SELL_STOP',
                        'price': current_price,
                        'shares': shares,
                        'value': trade_value,
                        'costs': costs,
                        'reason': stop_type,
                        'pnl': (current_price - entry_price) * shares - costs
                    })

                    shares = 0
                    entry_price = None
                    stop_loss = None
                    trailing_stop = None
                    last_trade_time = current_time

            # Execute trades based on signals
            if can_trade and current_signal['signal'] == 1 and shares == 0:
                # Calculate position size
                position_size = current_signal['position_size']
                trade_value = total_value * position_size
                shares_to_buy = int(trade_value / current_price)

                if shares_to_buy > 0:
                    actual_value = shares_to_buy * current_price
                    costs = self.calculate_transaction_costs(
                        current_price, shares_to_buy, True, avg_volume.iloc[i]
                    )
                    
                    if cash >= actual_value + costs:
                        cash -= actual_value + costs
                        shares = shares_to_buy
                        entry_price = current_price
                        last_trade_time = current_time

                        trades.append({
                            'timestamp': current_time,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': actual_value,
                            'costs': costs,
                            'confidence': current_signal['confidence'],
                            'prediction': current_signal['prediction'],
                        })

            elif can_trade and current_signal['signal'] == -1 and shares > 0:
                # Sell signal
                trade_value = shares * current_price
                costs = self.calculate_transaction_costs(
                    current_price, shares, False, avg_volume.iloc[i]
                )
                cash += trade_value - costs

                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': trade_value,
                    'costs': costs,
                    'pnl': (current_price - entry_price) * shares - costs,
                    'reason': 'Model signal',
                })

                shares = 0
                entry_price = None
                stop_loss = None
                trailing_stop = None
                last_trade_time = current_time

            
            # Record portfolio state
            position_value = shares * current_price
            total_value = cash + position_value

            portfolio['timestamp'].append(current_time)
            portfolio['cash'].append(cash)
            portfolio['shares'].append(shares)
            portfolio['position_value'].append(position_value)
            portfolio['total_value'].append(total_value)

            # Track highest value
            if total_value > highest_value:
                highest_value = total_value

        # Calculate returns
        portfolio_df = pd.DataFrame(portfolio)
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()

        return {
            'portfolio': portfolio_df,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'final_value': portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else self.initial_capital,
        }


    def calculate_metrics(self, portfolio: pd.DataFrame, trades: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            trades: DataFrame with trade records
            
        Returns:
            Dictionary with performance metrics
        """
        if len(portfolio) == 0:
            return {}
        
        # Basic returns metrics
        total_return = (portfolio['total_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate daily returns (resample minute to daily)
        portfolio['date'] = pd.to_datetime(portfolio['timestamp']).dt.date
        daily_values = portfolio.groupby('date')['total_value'].last()
        daily_returns = daily_values.pct_change().dropna()
        
        # Annualized metrics
        trading_days = len(daily_values)
        annualization_factor = 252 / max(trading_days, 1)
        
        if len(daily_returns) > 0:
            annualized_return = ((1 + total_return/100) ** annualization_factor - 1) * 100
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return / 100) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return / 100) / downside_deviation if downside_deviation > 0 else 0
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Drawdown analysis
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Trade analysis
        trade_metrics = {}
        if len(trades) > 0:
            buy_trades = trades[trades['action'] == 'BUY']
            sell_trades = trades[trades['action'].isin(['SELL', 'SELL_STOP', 'SELL_EOD'])]
            
            # Calculate P&L for completed trades
            completed_trades = []
            for i, sell in sell_trades.iterrows():
                # Find corresponding buy
                prior_buys = buy_trades[buy_trades['timestamp'] < sell['timestamp']]
                if len(prior_buys) > 0:
                    last_buy = prior_buys.iloc[-1]
                    pnl = (sell['price'] - last_buy['price']) * sell['shares'] - sell['costs'] - last_buy['costs']
                    completed_trades.append(pnl)
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t > 0]
                losing_trades = [t for t in completed_trades if t <= 0]
                
                trade_metrics = {
                    'total_trades': len(buy_trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0,
                    'avg_win': np.mean(winning_trades) if winning_trades else 0,
                    'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                    'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
                    'total_costs': trades['costs'].sum() if 'costs' in trades.columns else 0
                }
            else:
                trade_metrics = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_costs': 0
                }
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'trading_days': trading_days,
            **trade_metrics
        }
        
        return metrics


    def run_walk_forward_backtest(self,
                                data: pd.DataFrame,
                                strategy_params: Dict = None) -> Dict:

        """ Run a walk-forward backtest 
        
        Args:
            data: Full dataset
            strategy_params: Parameters for the strategy

        Returns:
            Dictionary with performance metrics
        """

        print("Start Walk-Forward Backtest")

        # Create walk-forward splits
        splits = self.walk_forward_split(data)

        all_portfolios = []
        all_trades = []
        all_metrics = []
        period_results = []

        for i, split in enumerate(splits):
            print(f"\n--- Period {i+1}/{len(splits)} ---")
            print(f"Test period: {split['test_start']} to {split['test_end']}")
            print(f"Train samples: {len(split['train'])}")
            print(f"Validation samples: {len(split['val'])}")
            print(f"Test samples: {len(split['test'])}")

            # Initialize strategy
            if strategy_params:
                strategy = MLTradingStrategy(**strategy_params)
            else:
                strategy = MLTradingStrategy()

            # Train model
            strategy.train(split['train'], split['val'])

            # Run backtest on test period
            results = self.simulate_trading(strategy, split['test'])

            # Calculate metrics for this period
            if len(results['portfolio']) > 0:
                metrics = self.calculate_metrics(results['portfolio'], results['trades'])
                metrics['period'] = split['period']

                # Add period identifier to portfolio and trades
                results['portfolio']['period'] = split['period']
                if len(results['trades']) > 0:
                    results['trades']['period'] = split['period']

                all_portfolios.append(results['portfolio'])
                if len(results['trades']) > 0:
                    all_trades.append(results['trades'])
                all_metrics.append(metrics)

                # Store feature importance
                top_features = strategy.get_feature_importance()

                period_results.append({
                    'period': split['period'],
                    'metrics': metrics,
                    'top_features': top_features,
                })

                print(f"Period return: {metrics['total_return']:.2f}%")
                print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Win rate: {metrics['win_rate']:.2f}%")

        # Combine all results
        combined_portfolio = pd.concat(all_portfolios, ignore_index=True) if all_portfolios else pd.DataFrame()
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        # Calculate overall metrics
        if len(combined_portfolio) > 0:
            overall_metrics = self.calculate_metrics(combined_portfolio, combined_trades)
        else:
            overall_metrics = {}

        # Calculate buy and hold metrics
        buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100

        self.results = {
            'portfolio': combined_portfolio,
            'trades': combined_trades,
            'metrics': overall_metrics,
            'period_metrics': pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame(),
            'period_results': period_results,
            'buy_hold_return': buy_hold_return,
            'strategy_params': strategy_params,
        }

        return self.results


    def print_report(self):
        """
        Print comprehensive performance report.
        """
        if not self.results or 'metrics' not in self.results:
            print("No results available. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("ML TRADING STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        
        if self.results['portfolio'] is not None and len(self.results['portfolio']) > 0:
            final_value = self.results['portfolio']['total_value'].iloc[-1]
            print(f"Final Portfolio Value: ${final_value:,.2f}")
        
        print("\n--- RETURNS ANALYSIS ---")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Buy & Hold Return: {self.results.get('buy_hold_return', 0):.2f}%")
        
        strategy_edge = metrics.get('total_return', 0) - self.results.get('buy_hold_return', 0)
        print(f"Strategy Edge: {strategy_edge:+.2f}%")
        
        print("\n--- RISK METRICS ---")
        print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        
        print("\n--- TRADING STATISTICS ---")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        if metrics.get('total_trades', 0) > 0:
            print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
            print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
            print(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        print("\n--- COST ANALYSIS ---")
        print(f"Total Transaction Costs: ${metrics.get('total_costs', 0):.2f}")
        
        if metrics.get('total_return', 0) != 0:
            cost_impact = metrics.get('total_costs', 0) / self.initial_capital * 100
            print(f"Cost Impact on Returns: {cost_impact:.2f}%")
        
        # Period-by-period analysis
        if 'period_metrics' in self.results and len(self.results['period_metrics']) > 0:
            print("\n--- PERIOD-BY-PERIOD PERFORMANCE ---")
            period_df = self.results['period_metrics']
            
            for _, period in period_df.iterrows():
                print(f"\n{period['period']}:")
                print(f"  Return: {period['total_return']:.2f}%")
                print(f"  Sharpe: {period['sharpe_ratio']:.2f}")
                print(f"  Win Rate: {period.get('win_rate', 0):.1f}%")
                print(f"  Trades: {period.get('total_trades', 0)}")
        
        # Top features across periods
        if 'period_results' in self.results and self.results['period_results']:
            print("\n--- TOP MODEL FEATURES ---")
            all_features = {}
            for period in self.results['period_results']:
                for feature, importance in period.get('top_features', {}).items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            # Average importance across periods
            avg_importance = {f: np.mean(v) for f, v in all_features.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i}. {feature}: {importance:.2f}")
        
        print("\n" + "="*60)
    

    def print_report(self):
        """
        Print comprehensive performance report.
        """
        if not self.results or 'metrics' not in self.results:
            print("No results available. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("ML TRADING STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        
        if self.results['portfolio'] is not None and len(self.results['portfolio']) > 0:
            final_value = self.results['portfolio']['total_value'].iloc[-1]
            print(f"Final Portfolio Value: ${final_value:,.2f}")
        
        print("\n--- RETURNS ANALYSIS ---")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Buy & Hold Return: {self.results.get('buy_hold_return', 0):.2f}%")
        
        strategy_edge = metrics.get('total_return', 0) - self.results.get('buy_hold_return', 0)
        print(f"Strategy Edge: {strategy_edge:+.2f}%")
        
        print("\n--- RISK METRICS ---")
        print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        
        print("\n--- TRADING STATISTICS ---")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        if metrics.get('total_trades', 0) > 0:
            print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
            print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
            print(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        print("\n--- COST ANALYSIS ---")
        print(f"Total Transaction Costs: ${metrics.get('total_costs', 0):.2f}")
        
        if metrics.get('total_return', 0) != 0:
            cost_impact = metrics.get('total_costs', 0) / self.initial_capital * 100
            print(f"Cost Impact on Returns: {cost_impact:.2f}%")
        
        # Period-by-period analysis
        if 'period_metrics' in self.results and len(self.results['period_metrics']) > 0:
            print("\n--- PERIOD-BY-PERIOD PERFORMANCE ---")
            period_df = self.results['period_metrics']
            
            for _, period in period_df.iterrows():
                print(f"\n{period['period']}:")
                print(f"  Return: {period['total_return']:.2f}%")
                print(f"  Sharpe: {period['sharpe_ratio']:.2f}")
                print(f"  Win Rate: {period.get('win_rate', 0):.1f}%")
                print(f"  Trades: {period.get('total_trades', 0)}")
        
        # Top features across periods
        if 'period_results' in self.results and self.results['period_results']:
            print("\n--- TOP MODEL FEATURES ---")
            all_features = {}
            for period in self.results['period_results']:
                for feature, importance in period.get('top_features', {}).items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            # Average importance across periods
            avg_importance = {f: np.mean(v) for f, v in all_features.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i}. {feature}: {importance:.2f}")
        
        print("\n" + "="*60)
    

    def plot_results(self):
        """
        Generate comprehensive visualization of backtest results.
        """
        if not self.results or 'portfolio' not in self.results:
            print("No results to plot. Run backtest first.")
            return
        
        portfolio = self.results['portfolio']
        trades = self.results['trades']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Cumulative Returns
        ax1 = plt.subplot(3, 2, 1)
        if len(portfolio) > 0:
            portfolio['cumulative_return'] = (portfolio['total_value'] / self.initial_capital - 1) * 100
            ax1.plot(portfolio['timestamp'], portfolio['cumulative_return'], label='Strategy', linewidth=2)
            
            # Add buy & hold line
            buy_hold_line = np.linspace(0, self.results.get('buy_hold_return', 0), len(portfolio))
            ax1.plot(portfolio['timestamp'], buy_hold_line, label='Buy & Hold', linestyle='--', alpha=0.7)
            
            ax1.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio Value
        ax2 = plt.subplot(3, 2, 2)
        if len(portfolio) > 0:
            ax2.fill_between(portfolio['timestamp'], 0, portfolio['cash'], 
                           label='Cash', alpha=0.5, color='green')
            ax2.fill_between(portfolio['timestamp'], portfolio['cash'], 
                           portfolio['total_value'], label='Position Value', 
                           alpha=0.5, color='blue')
            ax2.set_title('Portfolio Composition', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = plt.subplot(3, 2, 3)
        if len(portfolio) > 0:
            # Calculate drawdown
            cummax = portfolio['total_value'].expanding().max()
            drawdown = (portfolio['total_value'] - cummax) / cummax * 100
            ax3.fill_between(portfolio['timestamp'], 0, drawdown, 
                           color='red', alpha=0.3)
            ax3.plot(portfolio['timestamp'], drawdown, color='red', linewidth=1)
            ax3.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Trade Distribution
        ax4 = plt.subplot(3, 2, 4)
        if len(trades) > 0 and 'pnl' in trades.columns:
            pnls = trades.dropna(subset=['pnl'])['pnl']
            if len(pnls) > 0:
                ax4.hist(pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax4.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
                ax4.set_xlabel('P&L ($)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        ax5 = plt.subplot(3, 2, 5)
        if len(portfolio) > 0 and 'returns' in portfolio.columns:
            rolling_returns = portfolio['returns'].rolling(252).mean() * 252
            rolling_std = portfolio['returns'].rolling(252).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_std
            
            ax5.plot(portfolio['timestamp'], rolling_sharpe, linewidth=1.5)
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax5.axhline(y=1, color='green', linestyle='--', alpha=0.5)
            ax5.set_title('Rolling Sharpe Ratio (252-period)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.grid(True, alpha=0.3)
        
        # 6. Period Performance
        ax6 = plt.subplot(3, 2, 6)
        if 'period_metrics' in self.results and len(self.results['period_metrics']) > 0:
            period_df = self.results['period_metrics']
            x_pos = np.arange(len(period_df))
            
            colors = ['green' if r > 0 else 'red' for r in period_df['total_return']]
            ax6.bar(x_pos, period_df['total_return'], color=colors, alpha=0.7)
            ax6.set_title('Period-by-Period Returns', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Period')
            ax6.set_ylabel('Return (%)')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels([f"P{i+1}" for i in range(len(period_df))], rotation=45)
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('ML Trading Strategy Backtest Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
                    


