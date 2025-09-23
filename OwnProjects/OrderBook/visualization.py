""" This file contains the visualization for the order book. """

# Import used libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from metrics import Metrics
from order_book import OrderBook


class Visualizer:
    """ This class contains the visualization for the order book. """


    def __init__(self):
        """ Initialize the visualizer with default plot settings. """

        # Set up matplotlib style
        plt.style.use('default')
    

    def plot_price_evolution(self, metrics: Metrics, save_path: Optional[str] = None) -> None:
        """ Plot the evolution of prices over time.
        
        Args:
            metrics: The metrics object containing historical data
            save_path: Optional path to save the plot
        """

        if not metrics.metrics_history:
            print("No metrics data available for plotting")
            return
            
        # Extract data
        timestamps = [m['timestamp'] for m in metrics.metrics_history]
        mid_prices = [m['mid_price'] for m in metrics.metrics_history if m['mid_price'] is not None]
        spreads = [m['spread'] for m in metrics.metrics_history if m['spread'] is not None]
    
        # Create time index (relative to start)
        start_time = timestamps[0]
        time_elapsed = [(t - start_time) for t in timestamps]  # Seconds
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mid prices
        ax1.plot(time_elapsed[:len(mid_prices)], mid_prices, color='blue', label='Mid Price')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('Order Book Price Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot spreads
        ax2.plot(time_elapsed[:len(spreads)], spreads, color='orange', label='Bid-Ask Spread')
        ax2.set_ylabel('Spread ($)', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_order_book_depth(self, order_book: OrderBook, levels: int = 10, save_path: Optional[str] = None) -> None:
        """ Plot the current order book depth.
        
        Args:
            order_book: The order book object
            levels: Number of price levels to display
            save_path: Optional path to save the plot
        """

        depth_data = order_book.get_book_depth(levels)
        
        if not depth_data['bids'] and not depth_data['asks']:
            print("No order book data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bids (buy orders)
        if depth_data['bids']:
            bid_prices = [price for price, _ in depth_data['bids']]
            bid_quantities = [qty for _, qty in depth_data['bids']]
            ax.barh(bid_prices, bid_quantities, color='green', alpha=0.7, label='Bids (Buy Orders)')
        
        # Plot asks (sell orders)
        if depth_data['asks']:
            ask_prices = [price for price, _ in depth_data['asks']]
            ask_quantities = [-qty for _, qty in depth_data['asks']]  # Negative for left side
            ax.barh(ask_prices, ask_quantities, color='red', alpha=0.7, label='Asks (Sell Orders)')
        
        # Add mid price line
        mid_price = order_book.get_mid_price()
        if mid_price:
            ax.axhline(y=mid_price, linestyle='--', linewidth=2, label=f'Mid Price: ${mid_price:.2f}')
        
        ax.set_xlabel('Quantity', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Order Book Depth', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_realized_volatility(self, metrics: Metrics, save_path: Optional[str] = None) -> None:
        """ Plot realized volatility over time.
        
        Args:
            metrics: The metrics object containing volatility data
            save_path: Optional path to save the plot
        """

        if not metrics.realized_volatility:
            print("No volatility data available for plotting")
            return
        
        # Create time index for volatility (starts after initial period)
        time_points = list(range(len(metrics.realized_volatility)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time_points, metrics.realized_volatility, linewidth=2, color='blue', label='Realized Volatility')
        
        # Add rolling average
        if len(metrics.realized_volatility) > 5:
            rolling_avg = pd.Series(metrics.realized_volatility).rolling(window=5, center=True).mean()
            ax.plot(time_points, rolling_avg, linewidth=2, color='orange', linestyle='--', label='5-Period Moving Average')
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Annualized Volatility', fontsize=12)
        ax.set_title('Realized Volatility Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    def print_summary_stats(self, metrics: Metrics, order_book: OrderBook) -> None:
        """ Print summary statistics of the simulation.
        
        Args:
            metrics: The metrics object containing historical data
            order_book: The order book object
        """

        print("\n" + "="*50)
        print("Order book simulation summary")
        print("="*50)
        
        if metrics.metrics_history:
            # Price statistics
            mid_prices = [m['mid_price'] for m in metrics.metrics_history if m['mid_price'] is not None]
            spreads = [m['spread'] for m in metrics.metrics_history if m['spread'] is not None]
            
            if mid_prices:
                print(f"Price Statistics:")
                print(f"Average Mid Price: ${np.mean(mid_prices):.4f}")
                print(f"Price Range: ${np.min(mid_prices):.4f} - ${np.max(mid_prices):.4f}")
                print(f"Price Std Dev: ${np.std(mid_prices):.4f}")
            
            if spreads:
                print(f"\nSpread Statistics:")
                print(f"Average Spread: ${np.mean(spreads):.4f}")
                print(f"Min Spread: ${np.min(spreads):.4f}")
                print(f"Max Spread: ${np.max(spreads):.4f}")
        
        # Volatility statistics
        if metrics.realized_volatility:
            print(f"\nVolatility Statistics:")
            print(f"Average Realized Vol: {np.mean(metrics.realized_volatility):.4f}")
            print(f"Vol Range: {np.min(metrics.realized_volatility):.4f} - {np.max(metrics.realized_volatility):.4f}")
        
        # Trade statistics
        if order_book.trade_history:
            trade_volumes = [t['quantity'] for t in order_book.trade_history]
            buy_trades = sum(1 for t in order_book.trade_history if t['is_buy'])
            
            print(f"\nTrading Activity:")
            print(f"Total Trades: {len(order_book.trade_history)}")
            print(f"Buy/Sell Ratio: {buy_trades}/{len(order_book.trade_history) - buy_trades}")
            print(f"Average Trade Size: {np.mean(trade_volumes):.2f}")
            print(f"Total Volume: {np.sum(trade_volumes):.2f}")
        
        # Current order book state
        current_bid = order_book.get_best_bid()
        current_ask = order_book.get_best_ask()
        current_spread = order_book.get_spread()
        
        print(f"\nCurrent Order Book State:")
        if current_bid:
            print(f"Best Bid: ${current_bid[0]:.4f} (Size: {current_bid[1]:.1f})")
        if current_ask:
            print(f"Best Ask: ${current_ask[0]:.4f} (Size: {current_ask[1]:.1f})")
        if current_spread:
            print(f"Current Spread: ${current_spread:.4f}")
        
        print("="*50)