"""Visualization tools for strategy analysis"""

import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    def __init__(self, config):
        self.config = config

        
    def plot_strategy_results(self, data):
        """Create comprehensive plot of strategy results"""

        # Determine number of subplots based on available data
        n_plots = 4 if 'forecasted_volatility' in data.columns else 3
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
        fig.suptitle(f'{self.config.SYMBOL} Mean Reversion Strategy with GARCH Backtest', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Z-score
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label=f'{self.config.SYMBOL} Price', 
                color='blue', alpha=0.7)
        ax1.set_ylabel('Price ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')
        
        # Add Z-score on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data.index, data['z_score'], label='Z-score', 
                     color='red', alpha=0.7)
        
        # Add GARCH z-score if available
        if 'garch_z_score' in data.columns:
            ax1_twin.plot(data.index, data['garch_z_score'], label='GARCH Z-score', 
                         color='purple', alpha=0.7)
        
        ax1_twin.axhline(y=self.config.ENTRY_THRESHOLD, color='red', 
                        linestyle='--', alpha=0.5, label='Entry Threshold')
        ax1_twin.axhline(y=-self.config.ENTRY_THRESHOLD, color='red', 
                        linestyle='--', alpha=0.5)
        ax1_twin.axhline(y=self.config.EXIT_THRESHOLD, color='green', 
                        linestyle='--', alpha=0.5, label='Exit Threshold')
        ax1_twin.axhline(y=-self.config.EXIT_THRESHOLD, color='green', 
                        linestyle='--', alpha=0.5)
        ax1_twin.set_ylabel('Z-score', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1_twin.legend(loc='upper right')
        ax1.set_title('Price and Z-scores')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GARCH Volatility (if available)
        if 'forecasted_volatility' in data.columns:
            ax2 = axes[1]
            ax2.plot(data.index, data['forecasted_volatility'], 
                    label='GARCH Forecasted Volatility', color='orange', alpha=0.8)
            ax2.plot(data.index, data['conditional_volatility'], 
                    label='GARCH Conditional Volatility', color='brown', alpha=0.6)
            
            # Add volatility regime shading
            if 'high_vol_regime' in data.columns:
                high_vol_periods = data['high_vol_regime'] == 1
                ax2.fill_between(data.index, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                               where=high_vol_periods, alpha=0.2, color='red', 
                               label='High Volatility Regime')
            
            ax2.set_ylabel('Volatility')
            ax2.set_title('GARCH Volatility Estimates')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Trading signals
            self._plot_trading_signals(axes[2], data)
            
            # Plot 4: Portfolio performance
            self._plot_portfolio_performance(axes[3], data)
        else:
            # Plot 2: Trading signals
            self._plot_trading_signals(axes[1], data)
            
            # Plot 3: Portfolio performance
            self._plot_portfolio_performance(axes[2], data)
        
        plt.tight_layout()
        plt.show()
    

    def _plot_trading_signals(self, ax, data):
        """Plot trading signals on price chart"""
        
        ax.plot(data.index, data['close'], label=f'{self.config.SYMBOL} Price', 
               color='blue', alpha=0.7)
        
        # Mark entry and exit points
        long_entries = data[data['signal'] == 1]
        short_entries = data[data['signal'] == -1] 
        exits = data[data['signal'] == 0]
        
        if not long_entries.empty:
            ax.scatter(long_entries.index, long_entries['close'], 
                      color='green', marker='^', s=50, label='Long Entry')
        if not short_entries.empty:
            ax.scatter(short_entries.index, short_entries['close'], 
                      color='red', marker='v', s=50, label='Short Entry')
        #if not exits.empty:
        #    ax.scatter(exits.index, exits['close'], 
        #              color='orange', marker='x', s=50, label='Exit')
        
        ax.set_ylabel('Price ($)')
        ax.set_title('Trading Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    

    def _plot_portfolio_performance(self, ax, data):
        """Plot portfolio performance over time"""
        
        ax.plot(data.index, data['portfolio_value'], 
               label='Portfolio Value', color='purple', linewidth=2)
        ax.axhline(y=self.config.INITIAL_CAPITAL, color='black', 
                  linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.set_title('Portfolio Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
