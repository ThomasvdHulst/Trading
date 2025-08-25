""" This file contains the order book implementation. """

# Import used libraries
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import heapq
from data_strucs import Order, PriceLevel


class ITCHOrderBook:
    """ ITCH Order Book with individual order tracking. """

    def __init__(self, symbol: str = "TEST"):

        self.symbol = symbol

        # Order storage - flat dictionary for O(1) lookup
        self.orders: Dict[int, Order] = {}

        # Price levels - sorted dictionaries
        # Using integer prices (cents) for performance
        self.bid_levels: Dict[int, PriceLevel] = {}
        self.ask_levels: Dict[int, PriceLevel] = {}

        # Sorted price tracking for best bid/ask
        # Using heaps for O(log n) best price operations
        self.bid_prices: List[int] = [] # Max heap (negative values)
        self.ask_prices: List[int] = [] # Min heap

        # Cache best prices for O(1) access
        self._best_bid: Optional[int] = None
        self._best_ask: Optional[int] = None

        # Performance tracking
        self.operation_times = defaultdict(list)
        self.message_count = 0

        # Track order book events
        self.last_update_time = time.time_ns()
        self.total_volume_traded = 0


    def add_order(self, order_id: int, timestamp: int, side: str, price: int, shares: int) -> bool:
        """
        Add a new order to the order book.
        """

        start_time = time.perf_counter_ns()

        # Check for duplicates
        if order_id in self.orders:
            return False
        
        # Create order
        order = Order(order_id, timestamp, side, price, shares)
        self.orders[order_id] = order

        # Add to appropriate side
        if side == 'BID':
            if price not in self.bid_levels:
                self.bid_levels[price] = PriceLevel(price, 'BID')
                heapq.heappush(self.bid_prices, -price) # Negative for max heap

            self.bid_levels[price].add_order(order_id, shares)

            # Update chached best bid
            if self._best_bid is None or price > self._best_bid:
                self._best_bid = price

        else: # ASK
            if price not in self.ask_levels:
                self.ask_levels[price] = PriceLevel(price, 'ASK')
                heapq.heappush(self.ask_prices, price) # Min heap

            self.ask_levels[price].add_order(order_id, shares)

            # Update cached best ask
            if self._best_ask is None or price < self._best_ask:
                self._best_ask = price

        # Update
        self.message_count += 1
        self.last_update_time = timestamp

        # Track performance
        elapsed = time.perf_counter_ns() - start_time
        self.operation_times['add_order'].append(elapsed)

        return True
    

    def execute_order(self, order_id: int, executed_shares: int) -> bool:
        """
        Execute an partial or full order.
        """
        start_time = time.perf_counter_ns()

        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.shares -= executed_shares
        self.total_volume_traded += executed_shares

        # Update price level volume
        if order.side == 'BID':
            if order.price in self.bid_levels:
                self.bid_levels[order.price].total_volume -= executed_shares
        else: # ASK
            if order.price in self.ask_levels:
                self.ask_levels[order.price].total_volume -= executed_shares

        # Remove if fully executed
        if order.shares <= 0:
            self.delete_order(order_id)

        # Update
        self.message_count += 1

        # Track performance
        elapsed = time.perf_counter_ns() - start_time
        self.operation_times['execute_order'].append(elapsed)

        return True
    

    def cancel_order(self, order_id: int, cancelled_shares: int) -> bool:
        """
        Partial cancel of an order.
        """

        start_time = time.perf_counter_ns()

        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.shares -= cancelled_shares

        # Update level volume
        if order.side == 'BID':
            if order.price in self.bid_levels:
                self.bid_levels[order.price].total_volume -= cancelled_shares
        else: # ASK
            if order.price in self.ask_levels:
                self.ask_levels[order.price].total_volume -= cancelled_shares

        # Remove if fully cancelled
        if order.shares <= 0:
            self.delete_order(order_id)

        # Update
        self.message_count += 1

        # Track performance
        elapsed = time.perf_counter_ns() - start_time
        self.operation_times['cancel_order'].append(elapsed)

        return True
    

    def delete_order(self, order_id: int) -> bool:
        """
        Delete an order from the order book.
        """

        start_time = time.perf_counter_ns()

        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]

        # Remove from price level
        if order.side == 'BID':
            if order.price in self.bid_levels:
                level = self.bid_levels[order.price]
                if level.remove_order(order_id, order.shares):
                    # Remove level if empty
                    if level.order_count == 0:
                        del self.bid_levels[order.price]
                        # Update best bid if necessary
                        if order.price == self._best_bid:
                            self._update_best_bid()
        else: # ASK
            if order.price in self.ask_levels:
                level = self.ask_levels[order.price]
                if level.remove_order(order_id, order.shares):
                    # Remove level if empty
                    if level.order_count == 0:
                        del self.ask_levels[order.price]
                        # Update best ask if necessary
                        if order.price == self._best_ask:
                            self._update_best_ask()

        # Remove order
        del self.orders[order_id]
        self.message_count += 1

        # Track performance
        elapsed = time.perf_counter_ns() - start_time
        self.operation_times['delete_order'].append(elapsed)

        return True
    

    def get_best_bid(self) -> Optional[Tuple[int, int]]:
        """ Get best bid price and total volume at that level. """
        if self._best_bid and self._best_bid in self.bid_levels:
            level = self.bid_levels[self._best_bid]
            return (self._best_bid / 100.0, level.total_volume)
        
        return None
    

    def get_best_ask(self) -> Optional[Tuple[int, int]]:
        """ Get best ask price and total volume at that level. """
        if self._best_ask and self._best_ask in self.ask_levels:
            level = self.ask_levels[self._best_ask]
            return (self._best_ask / 100.0, level.total_volume)
        
        return None
    
    
    def get_spread(self) -> Optional[float]:
        """ Get bid-ask spread in dollars. """
        if self._best_bid and self._best_ask:
            return (self._best_ask - self._best_bid) / 100.0
        
        return None
    

    def get_mid_price(self) -> Optional[float]:
        """ Get mid price (average of best bid and ask) in dollars. """
        if self._best_bid and self._best_ask:
            return (self._best_bid + self._best_ask) / 200.0
        
        return None
    

    def get_queue_position(self, order_id: int) -> Optional[int]:
        """
        Get queue position for specific order.
        """

        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]

        if order.side == 'BID':
            if order.price in self.bid_levels:
                return self.bid_levels[order.price].get_queue_position(order_id)
        else:
            if order.price in self.ask_levels:
                return self.ask_levels[order.price].get_queue_position(order_id)
        
        return None
    

    def get_book_depth(self, levels: int = 5) -> Dict:
        """
        Get order book depth up to N levels.
        """

        # Get top N bid prices
        top_bids = sorted(self.bid_levels.keys(), reverse=True)[:levels]
        bid_depth = [(price/100.0, self.bid_levels[price].total_volume, self.bid_levels[price].order_count) for price in top_bids]

        # Get top N ask prices
        top_asks = sorted(self.ask_levels.keys())[:levels]
        ask_depth = [(price/100.0, self.ask_levels[price].total_volume, self.ask_levels[price].order_count) for price in top_asks]

        return {
            'bids': bid_depth,
            'asks': ask_depth,
            'bid_levels': len(self.bid_levels),
            'ask_levels': len(self.ask_levels),
            'total_orders': len(self.orders),
        }
    

    def _update_best_bid(self) -> None:
        """ Update cached best bid. """
        if self.bid_levels:
            self._best_bid = max(self.bid_levels.keys())
        else:
            self._best_bid = None
    

    def _update_best_ask(self) -> None:
        """ Update cached best ask. """
        if self.ask_levels:
            self._best_ask = min(self.ask_levels.keys())
        else:
            self._best_ask = None


    def get_performance_stats(self) -> Dict:
        """ Get performance statistics. """

        stats = {
            'total_messages': self.message_count,
            'active_orders': len(self.orders),
            'bid_levels': len(self.bid_levels),
            'ask_levels': len(self.ask_levels),
            'total_volume_traded': self.total_volume_traded
        }

        # Add timing statistics
        for operation, times in self.operation_times.items():
            if times:
                times_ns = np.array(times)
                stats[f'{operation}_avg_ns'] = np.mean(times_ns)
                stats[f'{operation}_p50_ns'] = np.percentile(times_ns, 50)
                stats[f'{operation}_p99_ns'] = np.percentile(times_ns, 99)

        return stats