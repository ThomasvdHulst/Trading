""" This file computes the metrics for a given order book. """

# Import used libraries
from order_book import OrderBook
import time
import numpy as np
from typing import Dict


class Metrics:
    """ This class computes the metrics for a given order book. """

    def __init__(self):
        self.metrics_history = []
        self.mids = []
        self.realized_volatility = []


    def calc_metrics(self, order_book: OrderBook) -> Dict:
        """ This method computes the metrics for a given order book. 
        
        Args:
            order_book: The order book to compute the metrics for.

        Returns:
            A dictionary containing the computed metrics.
        """

        metrics = {}

        # Basic metrics
        metrics['mid_price'] = order_book.get_mid_price()
        metrics['spread'] = order_book.get_spread()
        metrics['best_bid'] = order_book.get_best_bid()
        metrics['best_ask'] = order_book.get_best_ask()

        # Depth metrics
        metrics['bid_depth'] = order_book.get_total_volume('bid')
        metrics['ask_depth'] = order_book.get_total_volume('ask')
        metrics['depth_imbalance'] = (metrics['bid_depth'] - metrics['ask_depth']) / (metrics['bid_depth'] + metrics['ask_depth'])

        metrics['order_book_imbalance'] = self._calculate_order_book_imbalance(order_book)
        metrics['weighted_mid_price'] = self._calculate_weighted_mid_price(order_book)

        metrics['timestamp'] = time.time()

        self.metrics_history.append(metrics)
        return metrics


    def _calculate_order_book_imbalance(self, order_book: OrderBook) -> float:
        """ This method calculates the order book imbalance. 
        
        Args:
            order_book: The order book to compute the metrics for.

        Returns:
            The order book imbalance.
        """

        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[1] - best_ask[1]) / (best_bid[1] + best_ask[1])
        
        return 0


    def _calculate_weighted_mid_price(self, order_book: OrderBook) -> float:
        """ This method calculates the weighted mid price. 
        
        Args:
            order_book: The order book to compute the metrics for.

        Returns:
            The weighted mid price.
        """

        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[0] * best_bid[1] + best_ask[0] * best_ask[1]) / (best_bid[1] + best_ask[1])
        
        return 0
    

    def calculate_realized_volatility(self, order_book: OrderBook) -> None:
        """ This method calculates the realized volatility. 
        
        Args:
            order_book: The order book to compute the metrics for.

        """

        self.mids.append(order_book.get_mid_price())
        
        if len(self.mids) < 100:
            return

        mid_prices = [m for m in self.mids[-100:] if m is not None]

        returns = np.diff(mid_prices)
        self.realized_volatility.append(np.std(returns) * np.sqrt(252 * 6.5 * 60)) # Annualized
