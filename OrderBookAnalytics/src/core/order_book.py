""" This file contains the core order book implementation with price-level aggregation.
This represents the central limit order book that exchanges use to track the state of the order book."""

# Import used libraries
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class PriceLevel:
    """ Represents a single price level in the order book. """

    price: float
    quantity: float
    order_count: int
    last_update: float = field(default_factory=time.time) # Automatically set to current time

    def add_quantity(self, quantity: float) -> None:
        """ Add quantity to the price level. 
        
        Args:
            quantity: The quantity to add to the price level.
        """

        self.quantity += quantity
        self.order_count += 1
        self.last_update = time.time()


    def remove_quantity(self, quantity: float) -> None:
        """ Remove quantity from the price level. 
        
        Args:
            quantity: The quantity to remove from the price level.
        """

        self.quantity = max(0, self.quantity - quantity)
        self.order_count = max(0, self.order_count - 1)
        self.last_update = time.time()


class OrderBook:
    """ Efficient order book implementation using sorted dictionaries.
    Maintains bid/ask sides separately for O(1) best price access. """

    def __init__(self, symbol: str = "SYMBOL"):
        self.symbol = symbol
        self.bids: Dict[float, PriceLevel] = {} # Price -> PriceLevel
        self.asks: Dict[float, PriceLevel] = {} # Price -> PriceLevel
        self.last_trade_price: Optional[float] = None
        self.last_trade_size: Optional[float] = None
        self.timestamp: float = time.time()

        # Track order book events for analytics
        self.event_history = []
        self.trade_history = []


    def add_bid(self, price: float, quantity: float) -> None:
        """ Add a bid order to the order book. 
        
        Args:
            price: The price of the order.
            quantity: The quantity of the order.
        """

        if price in self.bids:
            self.bids[price].add_quantity(quantity)
        else:
            self.bids[price] = PriceLevel(price, quantity, 1)

        self._record_event("ADD_BID", price, quantity)

    
    def add_ask(self, price: float, quantity: float) -> None:
        """ Add an ask order to the order book. 
        
        Args:
            price: The price of the order.
            quantity: The quantity of the order.
        """

        if price in self.asks:
            self.asks[price].add_quantity(quantity)
        else:
            self.asks[price] = PriceLevel(price, quantity, 1)

        self._record_event("ADD_ASK", price, quantity)


    def remove_bid(self, price: float, quantity: float) -> None:
        """ Remove a bid order from the order book. 
        
        Args:
            price: The price of the order.
            quantity: The quantity of the order.
        """

        if price in self.bids:
            self.bids[price].remove_quantity(quantity)
            if self.bids[price].quantity <= 0:
                del self.bids[price]
        
        self._record_event("REMOVE_BID", price, quantity)

    
    def remove_ask(self, price: float, quantity: float) -> None:
        """ Remove an ask order from the order book. 
        
        Args:
            price: The price of the order.
            quantity: The quantity of the order.
        """

        if price in self.asks:
            self.asks[price].remove_quantity(quantity)
            if self.asks[price].quantity <= 0:
                del self.asks[price]
        
        self._record_event("REMOVE_ASK", price, quantity)


    def execute_trade(self, price: float, quantity: float, is_buy_aggressor: bool) -> None:
        """ Record a trade execution. 
        
        Args:
            price: The price of the trade.
            quantity: The quantity of the trade.
            is_buy_aggressor: Whether the trade is a buy or sell.
        """

        self.last_trade_price = price
        self.last_trade_size = quantity

        # Remove liquidity from the order book
        if is_buy_aggressor and price in self.asks:
            self.remove_ask(price, quantity)
        elif not is_buy_aggressor and price in self.bids:
            self.remove_bid(price, quantity)

        # Record trade
        trade_data = {
            'timestamp': time.time(),
            'price': price,
            'quantity': quantity,
            'is_buy_aggressor': is_buy_aggressor,
        }
        self.trade_history.append(trade_data)
        self._record_event("TRADE", price, quantity)


    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """ Get the best bid price and quantity. 
        
        Returns:
            The best bid price and quantity.
        """

        if not self.bids:
            return None
        
        best_price = max(self.bids.keys())
        return best_price, self.bids[best_price].quantity
    

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """ Get the best ask price and quantity. 
        
        Returns:
            The best ask price and quantity.
        """

        if not self.asks:
            return None
        
        best_price = min(self.asks.keys())
        return best_price, self.asks[best_price].quantity
    

    def get_mid_price(self) -> Optional[float]:
        """ Calculate the mid price from best bid and ask. 
        
        Returns:
            The mid price.
        """

        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2 # best_bid[0] is the best bid price, best_ask[0] is the best ask price
        
        return None
    

    def get_spread(self) -> Optional[float]:
        """ Calculate bid ask spread. 
        
        Returns:
            The bid ask spread.
        """
        
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        
        return None
    

    def get_book_depth(self, levels: int = 5) -> Dict:
        """ Get the book depth up to a given level. 
        
        Args:
            levels: The number of levels to get.
        
        Returns:
            The book depth, a dictionary with 'bids' and 'asks' keys, each containing a list of tuples (price, quantity)
        """

        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels] # Get the top N bids
        sorted_asks = sorted(self.asks.keys())[:levels] # Get the top N asks

        return {
            'bids': [(price, self.bids[price].quantity) for price in sorted_bids],
            'asks': [(price, self.asks[price].quantity) for price in sorted_asks],
        }
    

    def get_total_volume(self, side: str, levels: int = 5) -> float:
        """ Calculate total volume on one side of the book. 
        
        Args:
            side: The side of the book to get the volume for.
            levels: The number of levels to get the volume for.
        
        Returns:
            The total volume.
        """

        if side == 'bid':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return sum(self.bids[price].quantity for price in prices)
        else:
            prices = sorted(self.asks.keys())[:levels]
            return sum(self.asks[price].quantity for price in prices)
        
    
    def get_vwap(self, side: str, target_quantity: float) -> Optional[float]:
        """ Calculate volume-weighted average price for a target quantity. 
        
        Args:
            side: The side of the book to get the VWAP for.
            target_quantity: The target quantity to calculate the VWAP for.
        
        Returns:
            The VWAP.
        """

        # Get the book and prices for the given side
        if side == 'buy':
            prices = sorted(self.asks.keys())
            book = self.asks
        else:
            prices = sorted(self.bids.keys(), reverse=True)
            book = self.bids

        total_cost = 0
        remaining_qty = target_quantity

        # For each price level, add the quantity to the total cost and reduce the remaining quantity
        # We therefore iterate through the book and price levels until we hit the target quantity
        for price in prices:
            level_qty = min(remaining_qty, book[price].quantity)
            total_cost += price * level_qty
            remaining_qty -= level_qty

            if remaining_qty <= 0:
                return total_cost / target_quantity
            
        return None # Not enough liquidity
    

    def _record_event(self, event_type: str, price: float, quantity: float) -> None:
        """ Record order book events for analysis. 
        
        Args:
            event_type: The type of event to record.
            price: The price of the event.
            quantity: The quantity of the event.
        """
        
        self.event_history.append({
            'timestamp': time.time(),
            'type': event_type,
            'price': price,
            'quantity': quantity,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
        })


    def clear(self) -> None:
        """ Clear the order book. """

        self.bids.clear()
        self.asks.clear()
        self.event_history.clear()
        self.trade_history.clear()
