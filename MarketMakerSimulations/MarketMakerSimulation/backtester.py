""" This file contains the backtester for the market maker simulation """

# Import used libraries
import pandas as pd
import numpy as np
import scipy.stats


class MarketMakerBacktester:
    """ Backtests the market maker simulation """

    def __init__(self, config):
        self.config = config
        self.results = None

    
    def run_backtest(self, market_simulator):
        """ Run the backtest """

        print("Starting backtest...")
        print(f"Simulation period: {self.config.SIMULATION_TICKS} ticks")
        print(f"Tick frequency: {self.config.TICK_FREQUENCY_MS} ms")
        print("\n")

        # Run simulation
        results = market_simulator.run_simulation()

        # Store results
        self.results = results

        return results
    

    def analyze_performance(self, results):
        """ Analyze the performance of the market maker """

        analysis = {
            'summary_stats': self._calculate_summary_stats(results),
            'trade_analysis': self._analyze_trades(results),
            'risk_analysis': self._analyze_risk(results),
            'market_quality': self._analyze_market_quality(results),
        }

        return analysis
    

    def _calculate_summary_stats(self, results):
        """ Calculate summary statistics """

        mm_metrics = results['mm_final_metrics']

        stats = {
            'total_pnl': mm_metrics['total_pnl'],
            'total_volume': mm_metrics['total_volume'],
            'num_trades': mm_metrics['num_trades'],
            'final_inventory': mm_metrics['inventory'],
            'total_fees': mm_metrics['total_fees'],
            'simulation_ticks': len(results['price_history'])
        }

        # Calculate returns
        if len(results['mm_pnl_history']) > 0:
            initial_capital = self.config.INITIAL_CAPITAL
            final_value = initial_capital + mm_metrics['total_pnl']
            total_return = (final_value - initial_capital) / initial_capital * 100
            stats['total_return_pct'] = total_return

            # Annualized return (assuming tick frequency)
            ticks_per_year = 252 * 6.5 * 60 * 60 * 1000 / self.config.TICK_FREQUENCY_MS
            years = len(results['price_history']) / ticks_per_year
            if years > 0:
                annualized_return = (final_value / initial_capital) ** (1 / years) - 1
                stats['annualized_return_pct'] = annualized_return * 100
            else:
                stats['annualized_return_pct'] = 0
        else:
            stats['annualized_return_pct'] = 0
            stats['total_return_pct'] = 0

        return stats
    

    def _analyze_trades(self, results):
        """ Analyze trading patterns """
        
        if not results['trade_history']:
            return {}
        
        trades_df = pd.DataFrame(results['trade_history'])

        # Filter for market maker trades
        mm_trades = trades_df[(trades_df['buy_trader'] == 'MM_01') | (trades_df['sell_trader'] == 'MM_01')]

        if mm_trades.empty:
            return {}
        
        analysis = {
            'total_mm_trades': len(mm_trades),
            'avg_trade_size': mm_trades['quantity'].mean(),
            'std_trade_size': mm_trades['quantity'].std(),
            'avg_trade_price': mm_trades['price'].mean(),
            'std_trade_price': mm_trades['price'].std(),
            'price_range': mm_trades['price'].max() - mm_trades['price'].min()
        }

        # Separate buy and sell trades
        buy_trades = mm_trades[mm_trades['buy_trader'] == 'MM_01']
        sell_trades = mm_trades[mm_trades['sell_trader'] == 'MM_01']

        analysis['num_buy_trades'] = len(buy_trades)
        analysis['num_sell_trades'] = len(sell_trades)

        if not buy_trades.empty and not sell_trades.empty:
            analysis['avg_buy_price'] = buy_trades['price'].mean()
            analysis['avg_sell_price'] = sell_trades['price'].mean()
            analysis['realized_spread'] = analysis['avg_sell_price'] - analysis['avg_buy_price']

        # Trade frequency
        if len(mm_trades) > 1:
            trade_intervals = mm_trades['timestamp'].diff().dropna()
            analysis['avg_trade_interval'] = trade_intervals.mean()
            analysis['trades_per_100_ticks'] = 100 * len(mm_trades) / self.config.SIMULATION_TICKS

        return analysis
    

    def _analyze_risk(self, results):
        """ Analyze risk metrics """

        pnl_history = results['mm_pnl_history']
        inventory_history = results['mm_inventory_history']

        risk_analysis = {}

        # PnL volatility
        if len(pnl_history) > 1:
            pnl_returns = np.diff(pnl_history)
            risk_analysis['pnl_volatility'] = np.std(pnl_returns)
            risk_analysis['pnl_skewness'] = scipy.stats.skew(pnl_returns)
            risk_analysis['pnl_kurtosis'] = scipy.stats.kurtosis(pnl_returns)

            # Downside risk
            negative_returns = pnl_returns[pnl_returns < 0]
            if len(negative_returns) > 0:
                risk_analysis['downside_deviation'] = np.std(negative_returns)
                risk_analysis['avg_loss'] = np.mean(negative_returns)
                risk_analysis['max_loss'] = np.min(negative_returns)

            # Upside capture
            positive_returns = pnl_returns[pnl_returns > 0]
            if len(positive_returns) > 0:
                risk_analysis['avg_gain'] = np.mean(positive_returns)
                risk_analysis['max_gain'] = np.max(positive_returns)

            # Win rate
            if len(pnl_returns) > 0:
                risk_analysis['win_rate'] = len(positive_returns) / len(pnl_returns) * 100

        # Inventory risk
        if len(inventory_history) > 0:
            risk_analysis['avg_abs_inventory'] = np.mean(abs(np.array(inventory_history)))
            risk_analysis['inventory_volatility'] = np.std(inventory_history)
            risk_analysis['max_long_exposure'] = np.max(inventory_history)
            risk_analysis['max_short_exposure'] = np.min(inventory_history)

        return risk_analysis
    

    def _analyze_market_quality(self, results):
        """ Analyze market quality """

        spread_history = np.array(results['spread_history'])
        volume_history = np.array(results['volume_history'])
        price_history = np.array(results['price_history'])

        quality_metrics = {}

        # Spread metrics
        if len(spread_history) > 0:
            quality_metrics['avg_spread'] = np.mean(spread_history)
            quality_metrics['spread_volatility'] = np.std(spread_history)
            quality_metrics['max_spread'] = np.max(spread_history)
            quality_metrics['min_spread'] = np.min(spread_history)

        # Volume metrics
        if len(volume_history) > 0:
            quality_metrics['total_volume'] = np.sum(volume_history)
            quality_metrics['avg_volume_per_tick'] = np.mean(volume_history)
            quality_metrics['volume_volatility'] = np.std(volume_history)

            # Liquidity provision
            active_ticks = np.sum(volume_history > 0)
            quality_metrics['liquidity_provision_pct'] = active_ticks / len(volume_history) * 100

        # Price efficiency
        if len(price_history) > 0:
            price_returns = np.diff(price_history) / price_history[:-1]
            quality_metrics['price_volatility'] = np.std(price_returns)

            # Autocorrelation (price efficiency indicator)
            if len(price_returns) > 10:
                autocorr = np.corrcoef(price_returns[:-1], price_returns[1:])[0, 1]
                quality_metrics['price_autocorrelation'] = autocorr

        return quality_metrics
    

    def generate_report(self, analysis):
        """ Generate a report of the backtest results """

        print("Backtest report:")

        # Summary statistics
        print("\nSummary Statistics:")
        stats = analysis['summary_stats']
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        print(f"Total Return: {stats['total_return_pct']:.2f}%")
        print(f"Annualized Return: {stats['annualized_return_pct']:.2f}%")
        print(f"Total Volume: {stats['total_volume']:,}")
        print(f"Number of Trades: {stats['num_trades']}")
        print(f"Final Inventory: {stats['final_inventory']}")
        print(f"Total Fees: ${stats['total_fees']:,.2f}")
        print(f"Simulation Duration: {stats['simulation_ticks']} ticks")

        # Trade Analysis
        if 'trade_analysis' in analysis and analysis['trade_analysis']:
            print("\nTrade Analysis:")
            trade_stats = analysis['trade_analysis']
            print(f"MM Trades: {trade_stats.get('total_mm_trades', 0)}")
            print(f"Avg Trade Size: {trade_stats.get('avg_trade_size', 0):,.2f}")
            print(f"Std Trade Size: {trade_stats.get('std_trade_size', 0):,.2f}")
            print(f"Avg Trade Price: {trade_stats.get('avg_trade_price', 0):,.2f}")
            print(f"Std Trade Price: {trade_stats.get('std_trade_price', 0):,.2f}")
            print(f"Price Range: ${trade_stats.get('price_range', 0):,.2f}")
            print(f"Buy/Sell Trades: {trade_stats.get('num_buy_trades', 0)}/{trade_stats.get('num_sell_trades', 0)}")
            print(f"Avg Buy Price: {trade_stats.get('avg_buy_price', 0):,.2f}")
            print(f"Avg Sell Price: {trade_stats.get('avg_sell_price', 0):,.2f}")
            print(f"Realized Spread: ${trade_stats.get('realized_spread', 0):,.2f}")
            print(f"Trades per 100 ticks: {trade_stats.get('trades_per_100_ticks', 0):.2f}")

        # Risk Analysis
        if 'risk_analysis' in analysis and analysis['risk_analysis']:
            print("\nRisk Analysis:")
            risk_stats = analysis['risk_analysis']
            print(f"PnL Volatility: {risk_stats.get('pnl_volatility', 0):,.2f}")
            print(f"PnL Skewness: {risk_stats.get('pnl_skewness', 0):,.2f}")
            print(f"PnL Kurtosis: {risk_stats.get('pnl_kurtosis', 0):,.2f}")
            print(f"Downside Deviation: {risk_stats.get('downside_deviation', 0):,.2f}")
            print(f"Avg Loss: {risk_stats.get('avg_loss', 0):,.2f}")
            print(f"Max Loss: {risk_stats.get('max_loss', 0):,.2f}")
            print(f"Avg Gain: {risk_stats.get('avg_gain', 0):,.2f}")
            print(f"Max Gain: {risk_stats.get('max_gain', 0):,.2f}")
            print(f"Win Rate: {risk_stats.get('win_rate', 0):.2f}%")

        # Market Quality
        if 'market_quality' in analysis and analysis['market_quality']:
            print("\nMarket Quality:")
            quality_stats = analysis['market_quality']
            print(f"Avg Spread: ${quality_stats.get('avg_spread', 0):,.2f}")
            print(f"Spread Volatility: {quality_stats.get('spread_volatility', 0):,.2f}")
            print(f"Max Spread: ${quality_stats.get('max_spread', 0):,.2f}")
            print(f"Min Spread: ${quality_stats.get('min_spread', 0):,.2f}")
            print(f"Total Volume: {quality_stats.get('total_volume', 0):,.2f}")
            print(f"Avg Volume per Tick: {quality_stats.get('avg_volume_per_tick', 0):,.2f}")
            print(f"Volume Volatility: {quality_stats.get('volume_volatility', 0):,.2f}")
            print(f"Liquidity Provision: {quality_stats.get('liquidity_provision_pct', 0):.2f}%")
            print(f"Price Volatility: {quality_stats.get('price_volatility', 0):,.2f}")
            print(f"Price Autocorrelation: {quality_stats.get('price_autocorrelation', 0):,.2f}")

        print("\n")

    
    
    
    
    
    