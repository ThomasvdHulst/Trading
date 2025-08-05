""" This file contains the backtester for the mean reversion trading strategy with GARCH """

# Import used libraries
import pandas as pd

class Backtester:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.results = {}

    
    def run_backtest(self, data):
        """ Run the backtest on the data with signals """

        cash = self.initial_capital
        shares = 0
        portfolio_value = []
        trade_log = []

        for i in range(len(data)):
            price = data.iloc[i]['close']
            signal = data.iloc[i]['signal']
            timestamp = data.index[i]

            # Execute trades based on the observed signal
            if signal == 1 and shares == 0:

                # Enter a long position
                shares = cash // price
                cash = cash - shares * price
                trade_log.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'shares': shares,
                    'cash': cash
                })

            elif signal == -1 and shares == 0:
                # Enter a short position
                cash = cash + shares * price
                shares = 0
                trade_log.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'shares': shares,
                    'cash': cash
                })

            elif signal == 0 and shares > 0:
                # Exit the position
                cash = cash + shares * price
                shares = 0
                trade_log.append({
                    'timestamp': timestamp,
                    'action': 'exit',
                    'price': price,
                    'shares': shares,
                    'cash': cash
                })

            # After actions, calculate the current value of the portfolio
            current_value = cash + shares * price
            portfolio_value.append(current_value)


        # Store results
        self.results = {
            'portfolio_value': portfolio_value,
            'trade_log': trade_log,
            'final_value': portfolio_value[-1],
            'total_return': (portfolio_value[-1] - self.initial_capital) / self.initial_capital * 100,
            'num_trades': len(trade_log)
        }

        data_results = data.copy()
        data_results['portfolio_value'] = portfolio_value

        return data_results
    
    def print_results(self):
        """ Print the backtest results """

        if not self.results:
            print("No backtest results available")
            return

        print("---------------------")
        print("Backtest results:")
        print(f"Initial capital: {self.initial_capital:.2f}")
        print(f"Final value: {self.results['final_value']:.2f}")
        print(f"Total return: {self.results['total_return']:.2f}%")
        print(f"Number of trades: {self.results['num_trades']}")

        if self.results['num_trades'] > 0:
            avg_return = self.results['total_return'] / self.results['num_trades']
            print(f"Average return per trade: {avg_return:.2f}%")
        print("---------------------")


    def get_trade_summary(self):
        """ Get the trade summary """

        if not self.results['trade_log']:
            print("No trade log available")
            return
        
        trade_data = pd.DataFrame(self.results['trade_log'])

        print("---------------------")
        print("Trade summary:")
        print(f"Total trades: {len(trade_data)}")
        print(f"Buy trades: {len(trade_data[trade_data['action'] == 'buy'])}")
        print(f"Sell trades: {len(trade_data[trade_data['action'] == 'sell'])}")
        print("---------------------")

        return trade_data
        