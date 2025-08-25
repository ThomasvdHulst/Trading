""" This file contains the backtester class for the neural network trading strategy """

# Import used libraries
import pandas as pd

class LSTMBacktester:

    def __init__(self, config):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.commission_rate = config.COMMISSION_PER_TRADE
        self.slippage = config.SLIPPAGE_PCT
        self.bid_ask_spread = config.BID_ASK_SPREAD_PCT


    def calculate_transaction_costs(self, trade_value):
        """ Calculate the transaction costs for a trade """

        costs = {}

        # Commission
        costs['commission'] = trade_value * self.commission_rate

        # Bid-ask spread
        costs['bid_ask_spread'] = trade_value * self.bid_ask_spread

        # Slippage
        costs['slippage'] = trade_value * self.slippage

        total_costs = sum(costs.values())

        return total_costs, costs
    

    def run_backtest(self, data):
        """ Run the backtest """

        cash = self.initial_capital
        shares = 0
        portfolio_values = [] # To store the portfolio value at each time step
        trade_log = [] # To store the trade log
        total_transaction_costs = 0

        # Risk management tracking
        entry_price = None # To store the entry price of the current position
        stop_loss_price = None # Stop loss price, indicating the price at which to sell the position if price drops below this
        take_profit_price = None # Take profit price, indicating the price at which to sell the position if price rises above this

        # Iterate over the data
        for i in range(len(data)):
            current_price = data.iloc[i]['close'] # Get the current price
            signal = data.iloc[i]['signal'] # Get the corresponding calculated signal
            timestamp = data.index[i] # Set the timestaps for logging

            # Check stop loss and take profit
            if shares > 0 and entry_price is not None: # Long position
                # In the case that the current price is below the entry price and below the stop loss price, sell all shares
                if current_price <= entry_price <= stop_loss_price: # Stop loss triggered
                    # Sell all shares
                    trade_value = shares * current_price
                    transaction_costs, costs = self.calculate_transaction_costs(trade_value)

                    cash += trade_value - transaction_costs
                    total_transaction_costs += transaction_costs

                    trade_log.append({
                        'timestamp': timestamp,
                        'action': 'stop_loss',
                        'shares': shares,
                        'price': current_price,
                        'trade_value': trade_value,
                        'transaction_cost': transaction_costs,
                        'total_costs': total_transaction_costs,
                        'cash': cash,
                        'return_pct': (current_price - entry_price) / entry_price,
                    })

                    # Reset the position
                    shares = 0
                    entry_price = None
                    stop_loss_price = None
                    take_profit_price = None

                # In the case that the current price is above the entry price and above the take profit price, sell all shares
                elif current_price >= take_profit_price: # Take profit triggered
                    # Sell all shares
                    trade_value = shares * current_price
                    transaction_costs, costs = self.calculate_transaction_costs(trade_value)

                    cash += trade_value - transaction_costs
                    total_transaction_costs += transaction_costs

                    trade_log.append({
                        'timestamp': timestamp,
                        'action': 'take_profit',
                        'shares': shares,
                        'price': current_price,
                        'trade_value': trade_value,
                        'transaction_cost': transaction_costs,
                        'total_costs': total_transaction_costs,
                        'cash': cash,
                        'return_pct': (current_price - entry_price) / entry_price,
                    })

                    # Reset the position
                    shares = 0
                    entry_price = None
                    stop_loss_price = None
                    take_profit_price = None
                
            # Check for new entry, which happens when the signal is 1 (buy) and the position is 0 (no shares held)
            if signal == 1 and shares == 0: # Buy signal
                # Calculate position size
                position_value = cash * self.config.POSITION_SIZE
                shares_to_buy = int(position_value / current_price)

                if shares_to_buy > 0:
                    trade_value = shares_to_buy * current_price
                    transaction_costs, costs = self.calculate_transaction_costs(trade_value)

                    # We can only buy if we have enough cash to cover the trade value and transaction costs
                    if cash >= trade_value + transaction_costs:
                        # Update the cash and transaction costs
                        cash -= trade_value + transaction_costs
                        total_transaction_costs += transaction_costs

                        # Update the position, including the entry price, stop loss price and take profit price
                        shares = shares_to_buy
                        entry_price = current_price
                        stop_loss_price = current_price * (1 - self.config.STOP_LOSS_PCT)
                        take_profit_price = current_price * (1 + self.config.TAKE_PROFIT_PCT)

                        trade_log.append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'trade_value': trade_value,
                            'transaction_cost': transaction_costs,
                            'cash': cash,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                        })

                # In the case that the signal is -1 (sell) and the position is greater than 0 (shares held), sell all shares
                elif signal == -1 and shares > 0: # Sell signal
                    # Calculate position value
                    position_value = shares * current_price
                    transaction_costs, costs = self.calculate_transaction_costs(position_value)

                    cash += position_value - transaction_costs
                    total_transaction_costs += transaction_costs

                    trade_log.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'shares': shares,
                        'price': current_price,
                        'trade_value': position_value,
                        'transaction_cost': transaction_costs,
                        'cash': cash,
                        'return_pct': (current_price - entry_price) / entry_price if entry_price else 0,
                    })

                    # Reset the position
                    shares = 0
                    entry_price = None
                    stop_loss_price = None
                    take_profit_price = None
                
            # Update portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)

        # Store result
        self.results = {
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'final_value': portfolio_values[-1],
            'total_return_pct': (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100,
            'total_transaction_costs': total_transaction_costs,
            'num_trades': len(trade_log),
            'final_cash': cash,
            'final_shares': shares,
        }

        data_with_results = data.copy()
        data_with_results['portfolio_value'] = portfolio_values

        return data_with_results
    

    def print_backtest_results(self):
        """ Print the backtest results """

        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Final value: ${self.results['final_value']:,.2f}")
        print(f"Total return: {self.results['total_return_pct']:.2f}%")
        print(f"Total transaction costs: ${self.results['total_transaction_costs']:,.2f}")
        print(f"Number of trades: {self.results['num_trades']}")
        print(f"Final cash: ${self.results['final_cash']:,.2f}")
        print("\n")


    def get_trade_analysis(self):
        """ Analyze trade performance """

        if not self.results['trade_log']:
            print("No trades were made, no analysis possible")
            return None
        
        # Create a dataframe from the trade log to easily analyze the trades
        trade_data = pd.DataFrame(self.results['trade_log'])

        # Get different type of trades
        buy_trades = trade_data[trade_data['action'] == 'buy']
        # Get all sell trades, including stop loss and take profit trades
        sell_trades = trade_data[trade_data['action'].isin(['sell', 'stop_loss', 'take_profit'])]

        # Calculate trade metrics
        print("Trade Analysis:")
        print(f"Total trades: {len(trade_data)}")
        print(f"Total buy trades: {len(buy_trades)}")
        print(f"Total sell trades: {len(sell_trades)}")
        print(f"Of which are regular sells: {len(sell_trades[sell_trades['action'] == 'sell'])}")
        print(f"Of which are stop loss sells: {len(sell_trades[sell_trades['action'] == 'stop_loss'])}")
        print(f"Of which are take profit sells: {len(sell_trades[sell_trades['action'] == 'take_profit'])}")

        # Calculate win rate, which is the ratio of winning trades to total trades
        # A winning trade here is a trade that was sold for a profit
        completed_trades = sell_trades[sell_trades['return_pct'].notna()]
        if not completed_trades.empty:
            winning_trades = completed_trades[completed_trades['return_pct'] > 0]
            win_rate = len(winning_trades) / len(completed_trades)

            print(f"Win rate: {win_rate:.2%}")

            # Calculate average return
            avg_return = completed_trades['return_pct'].mean()
            print(f"Average return: {avg_return:.2%}")

        print("\n")

        return trade_data