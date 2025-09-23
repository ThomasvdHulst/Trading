""" This file contains the main code for the order book. """

# Import used libraries
from dataclasses import dataclass, field
import time
from typing import Dict, Optional, List, Tuple


@dataclass
class PriceLevel:
    """ This class represents a single price level in the order book. """

    price: float
    quantity: float
    order_count: int
    last_update: float = field(default_factory=time.time)

    def add_quantity(self, quantity: float) -> None:
        """ This function adds quantity to the price level. 
        
        Args:
            quantity: quantity to add to the price level
        """

        self.quantity += quantity
        self.order_count += 1
        self.last_update = time.time()

    
    def remove_quantity(self, quantity: float) -> None:
        """ This function removes quantity from the price level. 
        
        Args:
            quantity: quantity to remove from the price level
        """

        self.quantity = max(0, self.quantity - quantity)
        self.order_count = max(0, self.order_count - 1)
        self.last_update = time.time()


class OrderBook:
    """ This class contains the order book implementation. """

    
    def __init__(self) -> None:
        
        self.bids: Dict[float, PriceLevel] = {} # Price -> PriceLevel
        self.asks: Dict[float, PriceLevel] = {}

        self.last_trade_price: Optional[float] = None
        self.last_trade_size: Optional[float] = None
        self.timestamp: float = time.time()

        # Track order book events
        self.event_history: List = []
        self.trade_history: List = []


    def add_bid(self, price: float, quantity: float) -> None:
        """ This function adds a bid to the order book.
        
        Args:
            price: The price of the bid
            quantity: The size of the bid
        """

        if price in self.bids:
            self.bids[price].add_quantity(quantity)
        else:
            self.bids[price] = PriceLevel(price, quantity, 1)

        self._record_event("ADD_BID", price, quantity)


    def add_ask(self, price: float, quantity: float) -> None:
        """ This function adds an ask to the order book. 
        
        Args:
            price: The price of the ask
            quantity: The size of the ask
        """

        if price in self.asks:
            self.asks[price].add_quantity(quantity)
        else:
            self.asks[price] = PriceLevel(price, quantity, 1)

        self._record_event("ADD_ASK", price, quantity)


    def remove_bid(self, price: float, quantity: float) -> None:
        """ This function removes a bid from the order book. 
        
        Args:
            price: The price of the bid to remove
            quantity: The size of the bid to remove
        """

        # We can only remove if any bids for that price exist
        if price in self.bids:
            self.bids[price].remove_quantity(quantity)

            # In case the quantity is 0 (or lower, although it shouldn't be), remove the PriceLevel
            if self.bids[price].quantity <= 0:
                del self.bids[price]

        self._record_event("REMOVE_BID", price, quantity)

    
    def remove_ask(self, price: float, quantity: float) -> None:
        """ This function removes an ask from the orer book.
         
        Args:
            price: The price of the ask to remove
            quantity: The size fo the ask to remove
        """

        # We can only remove if any asks for that price exist
        if price in self.asks:
            self.asks[price].remove_quantity(quantity)

            # In case the quantity is 0 (or lower, although it shouldn't be), remove the PriceLevel
            if self.asks[price].quantity <= 0:
                del self.asks[price]

        self._record_event("REMOVE_ASK", price, quantity)


    def execute_trade(self, price: float, quantity: float, buy: bool) -> None:
        """ Record a trade execution.
        
        Args:
            price: The price of the trade
            quantity: The size of the trade
            buy: Whether the trade is a buy or a sell
        """

        # Remove from the order book
        # If we buy (and there is something to sell), remove the amount from the asks
        # Else if we sell (and there is something to buy), remove the amount from the bids
        if buy and price in self.asks:
            self.remove_ask(price, quantity)
        elif not buy and price in self.bids:
            self.remove_bid(price, quantity)

        # Update trackers
        self.last_trade_price = price
        self.last_trade_size = quantity

        # Record trade
        trade = {
            'timestamp': time.time(),
            'price': price,
            'quantity': quantity,
            'is_buy': buy
        }
        self.trade_history.append(trade)
        self._record_event("TRADE", price, quantity)


    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """ If available, get the best bid price and quantity.
        
        Returns:
            Tuple containing (price, quantity) of best bid
        """
        
        if not self.bids:
            return None
        
        best_price = max(self.bids.keys())
        return best_price, self.bids[best_price].quantity


    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """ If available, get the best ask price and quantity.
        
        Returns:
            Tuple containing (price, quantity) of the best ask
        """

        if not self.asks:
            return None
        
        best_price = min(self.asks.keys())
        return best_price, self.asks[best_price].quantity


    def get_mid_price(self) -> Optional[float]:
        """ If available, get the mid price of the order book.
        
        Returns:
            The mid price of the order book
        """
        
        # Get the best bid and ask data (can be None if not available)
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            # Obtain the mid price by getting the average (best) price
            return (best_bid[0] + best_ask[0]) / 2
        
        # If no bids and/or no asks, return None
        return None


    def get_spread(self) -> Optional[float]:
        """ If available, calculate the spread of the order book. 
        
        Returns:
            The spread of the order book
        """
        
        # Get the best bid and ask data (can be None if not available)
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            # Obtain the spread by taking the difference between the best ask price and best bid price
            return best_ask[0] - best_bid[0]
        
        # If no bids and/or asks, return None
        return None


    def get_book_depth(self, levels: int = 5) -> Dict:
        """ Get the book depth up to a given level.
        
        Args:
            levels: The number of levels to get
        Returns:
            The book depth, a dictionary with 'bids' and 'asks' keys,
            each containing a list of tuples (price, quantity)
        """

        # Sort the bids (reverse=True as highest first), and select the first 'levels' levels
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels] 
        # Sort the bids (reverse=False as lowest first), and select the first 'levels' levels
        sorted_asks = sorted(self.asks.keys())[:levels]

        return {
            'bids': [(price, self.bids[price].quantity) for price in sorted_bids],
            'asks': [(price, self.asks[price].quantity) for price in sorted_asks]
        }


    def get_total_volume(self, side: str) -> float:
        """ Calculate the total volume of a side of the order book.
        
        Args:
            side: Denotes the side of the order book 'bid' or 'ask'
        Return:
            The total volume of that side of the order book
        """

        if side == 'bid':
            prices = sorted(self.bids.keys(), reverse=True)
            return sum(self.bids[price].quantity for price in prices)
        else:
            prices = sorted(self.asks.keys())
            return sum(self.asks[price].quantity for price in prices)


    def get_vwap(self, side: str, target_qty: float) -> Optional[float]:
        """ Calculate the volume-weighted average price for a given target quantity.
        
        Args:
            side: Denotes the side of the order book 'buy' or 'sell' to calculate the VWAP for
            target_qty: The target quantity to calculate the VWAP for
        Returns:
            The VWAP
        """
        
        # Get the order book and prices for a given side
        if side == 'buy':
            prices = sorted(self.asks.keys())
            book = self.asks
        else:
            prices = sorted(self.bids.keys(), reverse=True)
            book = self.bids

        total_cost = 0
        remaining_qty = target_qty

        # For each price level, add the quantity to the total cost and reduce the remaining quantity
        # We therefore iterate through the book and price levels until we hit the target quantity
        for price in prices:
            # Call the quantity stored at that level (or just the remaining quantity if that is all we need)
            level_qty = min(remaining_qty, book[price].quantity)
            total_cost += price * level_qty
            remaining_qty -= level_qty

            # If we do not have any quantity remaining anymore, return the VWAP
            if remaining_qty <= 0:
                return total_cost / target_qty
    
        # Return None if we do not have enough liquidity in the order book for that side
        return None
    

    def _record_event(self, msg: str, price: float, quantity: float) -> None:
        """ Record order book events for analysis. 
        
        Args:
            msg: The message type (e.g. 'ADD_ASK', 'TRADE', etc.)
            price: The price around that event
            quantity: The quantity around that event
        """

        self.event_history.append({
            'timestamp': time.time(),
            'type': msg,
            'price': price,
            'quantity': quantity,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread()
        })

