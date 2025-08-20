""" This file contains the order execution implementation """

# Import used libraries
from typing import Dict, List, Optional, Set
from collections import defaultdict

from ..core.base import IOrderManager, IOrderBook, Order, Quote, QuotePair, Side, OrderType, Trade

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import MarketMakerConfig # Import the MarketMakerConfig classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import MarketMakerConfig


class OrderManager(IOrderManager):
    """ Manage order placement, cancellation and tracking. 
    Centralize management to:
    1. Avoid duplicate order placement
    2. Track orders consistently
    3. Implement order management rules (max orders, size, etc.)
    """

    def __init__(self, config: MarketMakerConfig):
        """ Initialize order management 
        
        Args:
            config: Market maker configuration
        """
        self.config = config

        # Active orders by ID
        self.active_orders: Dict[int, Order] = {}

        # Orders by side for quick lookup
        self.buy_orders: Set[int] = set()
        self.sell_orders: Set[int] = set()

        # Track orders by price level
        self.orders_by_price: Dict[float, Set[int]] = defaultdict(set)

        # Statistics
        self.total_orders_placed = 0
        self.total_orders_cancelled = 0
        self.total_orders_filled = 0


    def update_quotes(self, quotes: List[QuotePair], order_book: IOrderBook) -> None:
        """ Update quotes in the market.
        
        Args:
            quotes: List of quote pairs
            order_book: Order book interface
        """

        # First cancel all existing orders
        self.cancel_all_orders(order_book)

        # Place new quotes
        for quote_pair in quotes:
            # Place bid
            if quote_pair.bid:
                self._place_order(quote_pair.bid, order_book)

            # Place ask
            if quote_pair.ask:
                self._place_order(quote_pair.ask, order_book)

    
    def _place_order(self, quote: Quote, order_book: IOrderBook) -> Optional[int]:
        """ Place a single order.
        
        Args:
            quote: Quote to place
            order_book: Order book interface

        Returns:
            Order ID if successful, otherwise None
        """

        # Check if we are within limits
        if quote.side == Side.BUY and len(self.buy_orders) >= self.config.max_orders_per_side:
            print(f"Max buy orders reached: {len(self.buy_orders)}")
            return None
        elif quote.side == Side.SELL and len(self.sell_orders) >= self.config.max_orders_per_side:
            print(f"Max sell orders reached: {len(self.sell_orders)}")
            return None
        
        # Create order
        order = Order(
            order_id = 0, # Will be set by the order book
            trader_id = self.config.trader_id,
            side = quote.side,
            price = quote.price,
            quantity = quote.size,
            timestamp = 0, # Will be set by the order book
            order_type = OrderType.LIMIT
        )

        # Place order
        order_id = order_book.add_order(order)

        if order_id > 0:
            # Track order
            order.order_id = order_id
            self.active_orders[order_id] = order

            if quote.side == Side.BUY:
                self.buy_orders.add(order_id)
            else:
                self.sell_orders.add(order_id)

            self.orders_by_price[quote.price].add(order_id)
            self.total_orders_placed += 1

            return order_id
    
        return None
    

    def cancel_all_orders(self, order_book: IOrderBook) -> None:
        """ Cancel all active orders.
        
        Args:
            order_book: Order book interface
        """
        orders_to_cancel = list(self.active_orders.keys()) 

        for order_id in orders_to_cancel:
            if order_book.cancel_order(order_id):
                self._remove_order(order_id)
                self.total_orders_cancelled += 1

    
    def _remove_order(self, order_id: int) -> None:
        """ Remove order from active orders.
        
        Args:
            order_id: ID of the order to remove
        """

        if order_id not in self.active_orders:
            return 
        
        order = self.active_orders[order_id]

        # Remove from all tracking structures
        del self.active_orders[order_id]

        if order.side == Side.BUY:
            self.buy_orders.discard(order_id)
        else:
            self.sell_orders.discard(order_id)

        self.orders_by_price[order.price].discard(order_id)
        if not self.orders_by_price[order.price]:
            del self.orders_by_price[order.price]


    def on_trade(self, trade: Trade) -> None:
        """ Handle trade execution.
        
        Args:
            trade: Executed trade
        """

        # Check if one of our orders was involved
        if trade.buy_order_id in self.active_orders:
            # Our buy order was filled
            self._remove_order(trade.buy_order_id)
            self.total_orders_filled += 1
        elif trade.sell_order_id in self.active_orders:
            # Our sell order was filled
            self._remove_order(trade.sell_order_id)
            self.total_orders_filled += 1

    
    def get_active_orders(self) -> Dict[int, Order]:
        """ Get dictionary of active orders.
         
        Returns:
            Dictionary of active orders
        """
        return self.active_orders.copy()
    

    def get_order_count(self) -> Dict[str, int]:
        """ Get count of orders by side.
        
        Returns:
            Dictionary with buy/sell counts
        """

        return {
            'buy': len(self.buy_orders),
            'sell': len(self.sell_orders),
            'total': len(self.active_orders)
        }
    

    def get_metrics(self) -> Dict:
        """ Get order management metrics.
        
        Returns:
            Dictionary with order management metrics
        """

        return {
            'active_orders': len(self.active_orders),
            'buy_orders': len(self.buy_orders),
            'sell_orders': len(self.sell_orders),
            'total_placed': self.total_orders_placed,
            'total_cancelled': self.total_orders_cancelled,
            'total_filled': self.total_orders_filled,
            'unique_price_levels': len(self.orders_by_price)
        }
    
