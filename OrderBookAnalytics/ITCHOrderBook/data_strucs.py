""" This file contains the data structures for the ITCH parser: Order and PriceLevel. """

# Import used libraries
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque


@dataclass
class Order:
    """ Individual order in the book """
    order_id: int
    timestamp: int # nanoseconds
    side: str # 'BID' or 'ASK'
    price: int # Price in cents
    shares: int # Number of shares
    symbol: str = "TEST"

    def __hash__(self):
        return hash(self.order_id)
    
    def __eq__(self, other):
        return self.order_id == other.order_id
    

@dataclass
class PriceLevel:
    """ Price level containing orders in FIFO queue.
    Uses deque for O(1) add/remove at both ends.
    """

    price: int
    side: str
    orders: Deque[int] = field(default_factory=deque) # Order IDs in FIFO order
    total_volume: int = 0
    order_count: int = 0

    def add_order(self, order_id: int, shares: int) -> None:
        """ Add order to back of queue 
        
        Args:
            order_id: ID of the order to add
            shares: Number of shares to add
        """
        self.orders.append(order_id)
        self.total_volume += shares
        self.order_count += 1


    def remove_order(self, order_id: int, shares: int) -> bool:
        """ Remove specific order from queue 
        
        Args:
            order_id: ID of the order to remove
            shares: Number of shares to remove

        Returns:
            True if order was removed, False if order not found
        """

        try:
            self.orders.remove(order_id)
            self.total_volume -= shares
            self.order_count -= 1
            return True
        except ValueError:
            return False
        

    def get_queue_position(self, order_id: int) -> Optional[int]:
        """ Get position in queue (1-indexed).
        
        Args:
            order_id: ID of the order to get position of

        Returns:
            Position in queue (1-indexed), or None if order not found
        """
        try:
            return list(self.orders).index(order_id) + 1
        except ValueError:
            return None