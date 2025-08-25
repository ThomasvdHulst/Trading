""" This file contains visualization tools for oder book analysis. """

# Import used libraries
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
from ..core.order_book import OrderBook


class OrderBookVisualizer:
    """
    Creates various visualizations for order book analysis.
    Includes static plots and animated order book evolution.
    """
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = figsize
    

    def plot_book_snapshot(self, order_book: OrderBook, levels: int = 10, 
                          title: str = "Order Book Snapshot") -> Figure:
        """Create a snapshot visualization of the order book.
        
        Args:
            order_book: OrderBook object to visualize.
            levels: Number of levels to display.
            title: Title of the plot.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Get book depth
        depth = order_book.get_book_depth(levels)
        
        # Prepare data
        bid_prices = [p for p, _ in depth['bids']]
        bid_quantities = [q for _, q in depth['bids']]
        ask_prices = [p for p, _ in depth['asks']]
        ask_quantities = [q for _, q in depth['asks']]
        
        # Plot 1: Ladder view
        ax1.barh(range(len(bid_prices)), bid_quantities, color='green', alpha=0.7, label='Bids')
        ax1.barh(range(len(ask_prices)), [-q for q in ask_quantities], 
                color='red', alpha=0.7, label='Asks')
        
        # Add price labels
        all_prices = bid_prices + ask_prices
        ax1.set_yticks(range(len(all_prices[:levels])))
        ax1.set_yticklabels([f"${p:.2f}" for p in all_prices[:levels]])
        
        ax1.set_xlabel('Quantity')
        ax1.set_ylabel('Price')
        ax1.set_title('Order Book Ladder')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add mid price line
        mid = order_book.get_mid_price()
        if mid:
            ax1.axhline(y=len(bid_prices)-0.5, color='black', linestyle='--', 
                       label=f'Mid: ${mid:.2f}')
        
        # Plot 2: Depth chart
        if bid_prices and ask_prices:
            # Cumulative quantities
            bid_cumsum = np.cumsum(bid_quantities[::-1])[::-1]
            ask_cumsum = np.cumsum(ask_quantities)
            
            ax2.step(bid_prices, bid_cumsum, where='post', color='green', 
                    linewidth=2, label='Bid Depth')
            ax2.step(ask_prices, ask_cumsum, where='post', color='red', 
                    linewidth=2, label='Ask Depth')
            
            ax2.fill_between(bid_prices, bid_cumsum, step='post', alpha=0.3, color='green')
            ax2.fill_between(ask_prices, ask_cumsum, step='post', alpha=0.3, color='red')
            
            ax2.set_xlabel('Price')
            ax2.set_ylabel('Cumulative Quantity')
            ax2.set_title('Market Depth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add spread visualization
            if mid:
                spread = order_book.get_spread()
                ax2.axvline(x=mid, color='black', linestyle='--', alpha=0.5)
                if spread:
                    ax2.text(mid, ax2.get_ylim()[1]*0.9, 
                           f'Spread: ${spread:.3f}', ha='center')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    

    def plot_metrics_history(self, metrics_df: pd.DataFrame) -> Figure:
        """Plot historical microstructure metrics.
        
        Args:
            metrics_df: DataFrame containing the metrics.
        """

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        # Plot 1: Mid price evolution
        if 'mid_price' in metrics_df.columns:
            axes[0].plot(metrics_df.index, metrics_df['mid_price'], color='blue', linewidth=1)
            axes[0].set_title('Mid Price Evolution')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Price ($)')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Spread
        if 'spread' in metrics_df.columns:
            axes[1].plot(metrics_df.index, metrics_df['spread'], color='orange', linewidth=1)
            axes[1].set_title('Bid-Ask Spread')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Spread ($)')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Order book imbalance
        if 'order_book_imbalance' in metrics_df.columns:
            axes[2].plot(metrics_df.index, metrics_df['order_book_imbalance'], 
                        color='purple', linewidth=1)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_title('Order Book Imbalance')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Imbalance')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Depth
        if 'bid_depth_5' in metrics_df.columns and 'ask_depth_5' in metrics_df.columns:
            axes[3].plot(metrics_df.index, metrics_df['bid_depth_5'], 
                        color='green', label='Bid Depth', linewidth=1)
            axes[3].plot(metrics_df.index, metrics_df['ask_depth_5'], 
                        color='red', label='Ask Depth', linewidth=1)
            axes[3].set_title('Market Depth (5 levels)')
            axes[3].set_xlabel('Time')
            axes[3].set_ylabel('Total Quantity')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Book pressure
        if 'book_pressure' in metrics_df.columns:
            axes[4].plot(metrics_df.index, metrics_df['book_pressure'], 
                        color='brown', linewidth=1)
            axes[4].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[4].set_title('Book Pressure')
            axes[4].set_xlabel('Time')
            axes[4].set_ylabel('Pressure')
            axes[4].grid(True, alpha=0.3)
        
        # Plot 6: Weighted vs Simple Mid
        if 'weighted_mid_price' in metrics_df.columns and 'mid_price' in metrics_df.columns:
            axes[5].plot(metrics_df.index, 
                        metrics_df['weighted_mid_price'] - metrics_df['mid_price'],
                        color='teal', linewidth=1)
            axes[5].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[5].set_title('Weighted Mid - Simple Mid')
            axes[5].set_xlabel('Time')
            axes[5].set_ylabel('Difference ($)')
            axes[5].grid(True, alpha=0.3)
        
        plt.suptitle('Microstructure Metrics History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    

    def plot_toxicity_analysis(self, toxicity_metrics: Dict, 
                             volume_buckets: List[Dict]) -> Figure:
        """Plot toxicity analysis results.
        
        Args:
            toxicity_metrics: Dictionary containing the toxicity metrics.
            volume_buckets: List of dictionaries containing the volume buckets.
        """

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot 1: VPIN over time
        if volume_buckets:
            vpin_values = []
            for i in range(1, len(volume_buckets)):
                buckets_subset = volume_buckets[i-1:i]
                total_volume = sum(b['total_volume'] for b in buckets_subset)
                total_imbalance = sum(b['imbalance'] for b in buckets_subset)
                vpin = total_imbalance / total_volume if total_volume > 0 else 0
                vpin_values.append(vpin)
            
            if vpin_values:
                axes[0, 0].plot(vpin_values, color='red', linewidth=2)
                axes[0, 0].set_title('VPIN Evolution')
                axes[0, 0].set_xlabel('Volume Bucket')
                axes[0, 0].set_ylabel('VPIN')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add threshold line
                axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', 
                                 alpha=0.5, label='High Toxicity')
                axes[0, 0].legend()
        
        # Plot 2: Volume imbalance per bucket
        if volume_buckets:
            imbalances = [b['imbalance'] / b['total_volume'] 
                         for b in volume_buckets if b['total_volume'] > 0]
            axes[0, 1].bar(range(len(imbalances)), imbalances, color='blue', alpha=0.6)
            axes[0, 1].set_title('Volume Imbalance by Bucket')
            axes[0, 1].set_xlabel('Bucket Number')
            axes[0, 1].set_ylabel('Imbalance Ratio')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Metrics summary
        if toxicity_metrics:
            metrics_to_plot = {
                'VPIN': toxicity_metrics.get('vpin', 0),
                'Kyle Lambda': toxicity_metrics.get('kyle_lambda', 0) * 1000 if toxicity_metrics.get('kyle_lambda') else 0,
                'Adverse Selection': toxicity_metrics.get('mean_adverse_selection', 0) * 100 if toxicity_metrics.get('mean_adverse_selection') else 0,
            }
            
            bars = axes[1, 0].bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                                 color=['red', 'green', 'blue'], alpha=0.7)
            axes[1, 0].set_title('Toxicity Metrics Summary')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 4: Buy vs Sell volume distribution
        if volume_buckets:
            buy_volumes = [b['buy_volume'] for b in volume_buckets]
            sell_volumes = [b['sell_volume'] for b in volume_buckets]
            
            x = range(len(volume_buckets))
            width = 0.35
            
            axes[1, 1].bar([i - width/2 for i in x], buy_volumes, width, 
                          label='Buy Volume', color='green', alpha=0.6)
            axes[1, 1].bar([i + width/2 for i in x], sell_volumes, width,
                          label='Sell Volume', color='red', alpha=0.6)
            
            axes[1, 1].set_title('Buy vs Sell Volume Distribution')
            axes[1, 1].set_xlabel('Volume Bucket')
            axes[1, 1].set_ylabel('Volume')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Order Flow Toxicity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig