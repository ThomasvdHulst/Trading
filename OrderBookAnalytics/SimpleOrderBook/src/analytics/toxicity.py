""" This file contains the order flow toxicity analytics including VPIN and
adverse selection metrics. These help identify informed trading and potential adverse selection. """

# Import used libraries
import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import deque


class ToxicityAnalyzer:
    """
    Analyzes order flow toxicity using various metrics.
    """

    def __init__(self, volume_bucket_size: float = 1000):
        self.volume_bucket_size = volume_bucket_size
        self.current_bucket_volume = 0
        self.current_bucket_buys = 0
        self.current_bucket_sells = 0

        self.volume_buckets = deque(maxlen=50) # Keep last 50 buckets
        self.trade_history = []


    def process_trade(self, price: float, quantity: float, is_buy_aggressor: bool, mid_price: float) -> None:
        """ Process a trade and update volume buckets. 
        
        Args:
            price: The price of the trade.
            quantity: The quantity of the trade.
            is_buy_aggressor: Whether the aggressor is a buyer.
            mid_price: The mid price of the order book.
        """

        # Store trade
        self.trade_history.append({
            'price': price,
            'quantity': quantity,
            'is_buy_aggressor': is_buy_aggressor,
            'mid_price': mid_price,
            'timestamp': pd.Timestamp.now(),
        })

        # Update current bucket volume
        self.current_bucket_volume += quantity

        if is_buy_aggressor:
            self.current_bucket_buys += quantity
        else:
            self.current_bucket_sells += quantity

        # Update volume buckets
        if self.current_bucket_volume >= self.volume_bucket_size:
            self._complete_bucket()

        
    def _complete_bucket(self) -> None:
        """ Complete the current bucket and start a new one. """

        if self.current_bucket_volume > 0:
            bucket = {
                'total_volume': self.current_bucket_volume,
                'buy_volume': self.current_bucket_buys,
                'sell_volume': self.current_bucket_sells,
                'imbalance': abs(self.current_bucket_buys - self.current_bucket_sells),
                'timestamp': pd.Timestamp.now(),
            }
            self.volume_buckets.append(bucket)

        # Reset for next bucket
        self.current_bucket_volume = 0
        self.current_bucket_buys = 0
        self.current_bucket_sells = 0


    def calculate_vpin(self) -> Optional[float]:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
        Higher values indicate more toxic order flow.

        VPIN estimates the probability that a randomly selected trade comes from
        an informed trader rather than a noise trader.

        Returns:
            VPIN value
        """
        
        if len(self.volume_buckets) < 1: # Need minimum buckets
            return None
        
        total_volume = sum(b['total_volume'] for b in self.volume_buckets)
        total_imbalance = sum(b['imbalance'] for b in self.volume_buckets)

        if total_volume > 0:
            vpin = total_imbalance / total_volume
            return min(1.0, max(0.0, vpin)) # Bound between 0 and 1
        
        return None
    

    def calculate_adverse_selection(self) -> float:
        """ Calculate adverse selection metrics. Measures price movement after
        trades to identify informed trading. 

        Returns:
            Adverse selection scores
        """
        
        if len(self.trade_history) < 10:
            return {}
        
        adverse_selection_scores = []

        for i in range(len(self.trade_history) - 1):
            trade = self.trade_history[i]
            future_trade = self.trade_history[i + 1]

            # Calculate price movement
            price_change = future_trade['mid_price'] - trade['mid_price']

            # Adverse selection: did the price move against the liquidity provider?
            if trade['is_buy_aggressor']:
                # If buy aggressor, price should go up (bad for seller/LP)
                adverse_score = price_change
            else:
                # If sell aggressor, price should go down (bad for buyer/LP)
                adverse_score = -price_change

            adverse_selection_scores.append(adverse_score)

        if adverse_selection_scores:
            return {
                'mean_adverse_selection': np.mean(adverse_selection_scores),
                'adverse_selection_std': np.std(adverse_selection_scores),
                'positive_selection_rate': np.mean([s > 0 for s in adverse_selection_scores])
            }
        
        return {}
    

    def calculate_kyle_lambda(self) -> Optional[float]:
        """ Calculate Kyle's Lambda - price impact coefficient
        Measures permanent price impact per unit of net order flow.
        
        Simplified version: estimate lambda from recent trades.

        Returns:
            Kyle's Lambda value
        """

        if len(self.trade_history) < 20:
            return None
        
        # Get recent trades
        recent_trades = self.trade_history[-20:]

        # Calculate net order flow and price changes
        net_flows = []
        price_changes = []

        for i in range(1, len(recent_trades)):
            # Net order flow (signed volume)
            if recent_trades[i]['is_buy_aggressor']:
                net_flow = recent_trades[i]['quantity']
            else:
                net_flow = -recent_trades[i]['quantity']

            # Price change
            price_change = recent_trades[i]['mid_price'] - recent_trades[i - 1]['mid_price']

            net_flows.append(net_flow)
            price_changes.append(price_change)

        # Estimate lambda using linear regression
        # Price_change = lambda * net_flow + noise
        if len(net_flows) > 0:
            net_flows = np.array(net_flows)
            price_changes = np.array(price_changes)

            if np.var(net_flows) > 0:
                kyle_lambda = np.cov(price_changes, net_flows)[0, 1] / np.var(net_flows)
                return abs(kyle_lambda) # Return absolute value
            

        return None
    

    def get_toxicity_metrics(self) -> Dict:
        """ Get all toxicity metrics in a single dictionary. """

        metrics = {
            'vpin': self.calculate_vpin(),
            'kyle_lambda': self.calculate_kyle_lambda(),
            'bucket_count': len(self.volume_buckets),
            'total_trades': len(self.trade_history),
            'mean_adverse_selection': self.calculate_adverse_selection().get('mean_adverse_selection', 0),
        }

        # Add adverse selection metrics if available
        metrics.update(self.calculate_adverse_selection())

        return metrics
        
