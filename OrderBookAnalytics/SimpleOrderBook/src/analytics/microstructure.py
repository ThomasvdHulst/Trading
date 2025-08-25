""" This file contains the microstructure analytics for order book analysis. """

# Import used libraries
import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..core.order_book import OrderBook


class MicrostructureAnalyzer:
    """
    Calculates various microstructure metrics from the order book data.
    """

    def __init__(self):
        self.metrics_history = []

    
    def calculate_metrics(self, order_book: OrderBook) -> Dict:
        """ Calculate microstructure metrics. 
        
        Args:
            order_book: The order book to analyze.

        Returns:
            A dictionary containing the calculated metrics.
        """

        metrics = {}

        # Basic metrics
        metrics['mid_price'] = order_book.get_mid_price()
        metrics['spread'] = order_book.get_spread()
        metrics['relative_spread'] = self._calculate_relative_spread(order_book)

        # Depth metrics
        metrics['bid_depth_5'] = order_book.get_total_volume('bid', 5)
        metrics['ask_depth_5'] = order_book.get_total_volume('ask', 5)
        metrics['depth_imbalance'] = self._calculate_depth_imbalance(order_book)

        # Order book imbalance
        metrics['order_book_imbalance'] = self._calculate_order_book_imbalance(order_book)
        metrics['weighted_mid_price'] = self._calculate_weighted_mid_price(order_book)

        # Liquidity metrics
        metrics['liquidity_ratio'] = self._calculate_liquidity_ratio(order_book)
        metrics['book_pressure'] = self._calculate_book_pressure(order_book)

        # Store metrics with timestamp
        metrics['timestamp'] = pd.Timestamp.now()
        self.metrics_history.append(metrics)

        return metrics
    

    def _calculate_relative_spread(self, order_book: OrderBook) -> Optional[float]:
        """ Calculate the relative spread of the order book.
        
        Args:
            order_book: The order book to analyze.

        Returns:
            Relative spread
        """

        spread = order_book.get_spread()
        mid = order_book.get_mid_price()

        if spread and mid and mid > 0:
            return (spread / mid) * 10000 # Convert to basis points
        
        return None
    

    def _calculate_depth_imbalance(self, order_book: OrderBook) -> float:
        """ Calculate depth imbalance ratio. Positive values indicate more buy volume than sell volume. 
        
        Args:
            order_book: The order book to analyze.

        Returns:
            Depth imbalance ratio
        """

        bid_depth = order_book.get_total_volume('bid', 5)
        ask_depth = order_book.get_total_volume('ask', 5)

        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            return (bid_depth - ask_depth) / total_depth
        
        return 0
    

    def _calculate_order_book_imbalance(self, order_book: OrderBook) -> float:
        """ Calculate order book imbalance. 
        
        Args:
            order_book: The order book to analyze.

        Returns:
            Order book imbalance
        """

        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()

        if best_bid and best_ask:
            bid_qty = best_bid[1]
            ask_qty = best_ask[1]
            total = bid_qty + ask_qty

            if total > 0:
                return (bid_qty - ask_qty) / total
            
        return 0
    

    def _calculate_weighted_mid_price(self, order_book: OrderBook) -> Optional[float]:
        """ Calculate weighted mid price. 
        
        Args:
            order_book: The order book to analyze.

        Returns:
            Weighted mid price
        """

        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()

        if best_bid and best_ask:
            bid_price, bid_qty = best_bid
            ask_price, ask_qty = best_ask

            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                return (bid_price * bid_qty + ask_price * ask_qty) / total_qty
            
        return None
    

    def _calculate_liquidity_ratio(self, order_book: OrderBook) -> float:
        """ Calculate liquidity ratio. Measures how concentrated liquidity is
        near best prices. """

        depth_1 = order_book.get_total_volume('bid', 1) + order_book.get_total_volume('ask', 1)
        depth_5 = order_book.get_total_volume('bid', 5) + order_book.get_total_volume('ask', 5)

        if depth_5 > 0:
            return depth_1 / depth_5
        
        return 0
    

    def _calculate_book_pressure(self, order_book: OrderBook, levels: int = 3) -> float:
        """ Calculate aggregated book pressure across multiple levels.
        Weighted by inverse distance from mid price. 
        
        Args:
            order_book: The order book to analyze.
            levels: Number of levels to consider.

        Returns:
            Book pressure
        """

        mid_price = order_book.get_mid_price()
        if not mid_price:
            return 0
        
        book_depth = order_book.get_book_depth(levels)

        bid_pressure = 0
        for price, qty in book_depth['bids']:
            weight = 1 / (1 + abs(price - mid_price))
            bid_pressure += weight * qty

        ask_pressure = 0
        for price, qty in book_depth['asks']:
            weight = 1 / (1 + abs(price - mid_price))
            ask_pressure += weight * qty

        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            return (bid_pressure - ask_pressure) / total_pressure
        
        return 0
    

    def get_metrics_df(self) -> pd.DataFrame:
        """ Get historical metrics as a pandas DataFrame. """
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        df.set_index('timestamp', inplace=True)
        return df
    

    def calculate_realized_volatility(self, window: int = 20) -> Optional[float]:
        """ Calculate realized volatility from mid-price changes.
        Important for option pricing and risk management. """
        
        if len(self.metrics_history) < window:
            return None
        
        mid_prices = [m['mid_price'] for m in self.metrics_history[-window:] if m['mid_price'] is not None]

        if len(mid_prices) < 2:
            return None
        
        returns = np.diff(mid_prices)
        return np.std(returns) * np.sqrt(252 * 6.5 * 60) # Annualized