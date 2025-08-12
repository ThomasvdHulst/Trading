""" This file contains the order book for the market maker simulation """

# Import used libraries
import heapq
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class Side(Enum):
    """ This class represents the side of an order. We use Enum to represent
    the two possible sides of an order: buy or sell. The use of Enum makes the code
    more readable and easier to understand. This is because we now can just use the Side.BUY and
    Side.SELL instead of having to use 1 and -1."""
    BUY = 1
    SELL = -1


# @dataclass is a decorator that automatically adds special methods to the class, such
# as __init__, __repr__, __eq__, etc. This is useful because we don't have to write
# these methods ourselves.
@dataclass
class Order:
    """ This class represents a single order in the order book """

    # We use :type to specify the type of the attribute. This is useful because it
    # makes the code more readable and easier to understand. The reader can see
    # what type of data is expected for each attribute.
    order_id: int
    side: Side
    price: float
    quantity: int
    timestamp: int
    trader_id: str

    # We use __lt__ to define the comparison for the heap ordering. This is useful
    # because it allows us to use the heapq module to create a heap of orders.
    def __lt__(self, other):
        """ Comparison for heap orderning """
        if self.side == Side.BUY:
            # For buy orders, the higher price has priority (max heap)
            return self.price > other.price
        else:
            # For sell orders, the lower price has priority (min heap)
            return self.price < other.price
        

class OrderBook:
    """ This class represents the order book """

    def __init__(self, config):
        self.config = config
        self.buy_orders = [] # Max heap for buy orders
        self.sell_orders = [] # Min heap for sell orders
        self.orders_dict = {} # Dictionary to store orders by order_id
        self.next_order_id = 1 # Next order ID to assign
        self.current_timestamp = 0 # Current timestamp

        # Track best bid and ask
        self.best_bid = None
        self.best_ask = None

        # Track market data
        self.last_trade_price = config.INITIAL_PRICE # Start at initial price
        self.mid_price = config.INITIAL_PRICE # Also start at initial price
        self.spread = 0.0 # Obviously, the spread is 0 at the start

        # Trade history
        self.trade_history = []


    def add_order(self, side: Side, price: float, quantity: int, trader_id: str) -> int:
        """ Add an order to the order book """

        # Round the price to the nearest tick size
        # For example, 100.006 becomes 100.01
        price = round(price / self.config.TICK_SIZE) * self.config.TICK_SIZE

        # Round quantity to the nearest lot size
        # For example, 101 becomes 100  
        quantity = int(quantity / self.config.LOT_SIZE) * self.config.LOT_SIZE

        # Do not allow orders with quantity 0 or less
        if quantity <= 0:
            return -1
        
        # Create the order
        order = Order(
            order_id = self.next_order_id, # Assign a new order ID
            side = side, # Side of the order, determined by the Side class
            price = price, # Price of the order
            quantity = quantity, # Quantity of the order
            timestamp = self.current_timestamp, # Timestamp of the order
            trader_id = trader_id # Trader ID of the order
        )

        self.next_order_id += 1 # Increment the order ID
        self.orders_dict[order.order_id] = order # Add the order to the dictionary

        # Try to match the order immediately
        matched_quantity = self._try_match_order(order)

        # Add remaining quantity to the order book (if any)
        if order.quantity > 0:
            if side == Side.BUY:
                # We use tuples so that the heapq module can sort the order by price, timestamp (FIFO) and order_id
                # Since heapq is always a min heap, we use the negative price to make it a max heap
                heapq.heappush(self.buy_orders, (-price, order.timestamp, order.order_id))
            else:
                heapq.heappush(self.sell_orders, (price, order.timestamp, order.order_id))

        # Update best bid/ask
        self._update_best_quotes()

        return order.order_id
    

    def cancel_order(self, order_id: int) -> bool:
        """ Cancel an order from the order book """

        if order_id not in self.orders_dict:
            return False
        
        order = self.orders_dict[order_id]
        order.quantity = 0 # Mark as cancelled
        del self.orders_dict[order_id] # Remove from dictionary

        # Clean up the heaps periodically
        self._clean_heaps()

        # Update best bid/ask
        self._update_best_quotes()

        return True
    

    def modify_order(self, order_id: int, new_quantity: int) -> bool:
        """ Modify order quantity (for price modification, we require to cancel and add a new order) """

        if order_id not in self.orders_dict:
            return False
        
        # Get the order and modify the quantity
        order = self.orders_dict[order_id]
        order.quantity = new_quantity

        # If the quantity is 0, just cancel the order
        if new_quantity == 0:
            return self.cancel_order(order_id)
        
        return True
    

    def _try_match_order(self, order: Order) -> int:
        """ Try to match the order with existing orders in the order book """

        matched_quantity = 0

        if order.side == Side.BUY: # BUY order
            
            # Try to match with the sell orders
            while self.sell_orders and order.quantity > 0:

                # Automatically pop the best sell order, since we use a min heap
                best_sell_price, best_sell_timestamp, best_sell_id = self.sell_orders[0]

                if best_sell_id not in self.orders_dict:
                    # If the order is cancelled, we pop it from the heap
                    heapq.heappop(self.sell_orders)
                    continue

                best_sell_order = self.orders_dict[best_sell_id]

                # Check if prices cross
                if order.price >= best_sell_price:
                    # We found a match, as the buy price is the same or higher than the sell price

                    # Define the trade quantity as the minimum of the buy and sell order quantities
                    # (We can only buy/sell as much as the other side wants to sell/buy)
                    trade_quantity = min(order.quantity, best_sell_order.quantity)

                    trade_price = best_sell_price # Trade price is the sell price, so buy orders are executed at the best sell price

                    # Execute trade
                    self._execute_trade(buy_order = order, sell_order = best_sell_order, quantity = trade_quantity, price = trade_price)

                    # Update quantities
                    order.quantity -= trade_quantity
                    best_sell_order.quantity -= trade_quantity
                    matched_quantity += trade_quantity
                    
                    # Remove filled order
                    if best_sell_order.quantity == 0:
                        heapq.heappop(self.sell_orders)
                        del self.orders_dict[best_sell_id]
                else:
                    # If the buy price is lower than the best sell price, we break the loop
                    break

        else: # SELL order

            # Try to match with the buy orders
            while self.buy_orders and order.quantity > 0:
                neg_best_buy_price, best_buy_timestamp, best_buy_id = self.buy_orders[0]
                best_buy_price = -neg_best_buy_price

                if best_buy_id not in self.orders_dict:
                    # If the order is cancelled, we pop it from the heap
                    heapq.heappop(self.buy_orders)
                    continue

                best_buy_order = self.orders_dict[best_buy_id]

                # Check if prices cross
                if order.price <= best_buy_price:
                    # We found a match, as the sell price is the same or lower than the buy price

                    # Define the trade quantity as the minimum of the sell and buy order quantities
                    # (We can only buy/sell as much as the other side wants to sell/buy)
                    trade_quantity = min(order.quantity, best_buy_order.quantity)

                    trade_price = best_buy_price # Trade price is the buy price, so sell orders are executed at the best buy price

                    # Execute trade
                    self._execute_trade(sell_order = order, buy_order = best_buy_order, quantity = trade_quantity, price = trade_price)

                    # Update quantities
                    order.quantity -= trade_quantity
                    best_buy_order.quantity -= trade_quantity
                    matched_quantity += trade_quantity

                    # Remove filled order
                    if best_buy_order.quantity == 0:
                        heapq.heappop(self.buy_orders)
                        del self.orders_dict[best_buy_id]

                else:
                    # If the sell price is higher than the best buy price, we break the loop
                    break

        return matched_quantity
                
    
    def _execute_trade(self, buy_order: Order, sell_order: Order, quantity: int, price: float):
        """ Execute a trade between two orders """

        # Create and store the trade
        trade = {
            'timestamp': self.current_timestamp, # Timestamp of the trade
            'price': price, # Price of the trade
            'quantity': quantity, # Quantity of the trade
            'buy_trader': buy_order.trader_id, # Trader ID of the buy order
            'sell_trader': sell_order.trader_id, # Trader ID of the sell order
            'buy_order_id': buy_order.order_id, # Order ID of the buy order
            'sell_order_id': sell_order.order_id # Order ID of the sell order
        }

        self.trade_history.append(trade) # Add the trade to the trade history
        self.last_trade_price = price # Update the last trade price


    def _update_best_quotes(self):
        """ Update the best bid and ask """

        # Find best bid
        self.best_bid = None
        while self.buy_orders: # While there are buy orders
            # Automatically pop the best buy order, since we use a max heap (- min heap)
            neg_price, timestamp, order_id = self.buy_orders[0]

            # Check if the order is still in the order book
            if order_id in self.orders_dict:
                self.best_bid = -neg_price
                break
            else:
                heapq.heappop(self.buy_orders)


        # Find best ask
        self.best_ask = None
        while self.sell_orders: # While there are sell orders
            # Automatically pop the best sell order, since we use a min heap
            price, timestamp, order_id = self.sell_orders[0]

            # Check if the order is still in the order book
            if order_id in self.orders_dict:
                self.best_ask = price
                break
            else:
                heapq.heappop(self.sell_orders)

        
        # Update mid price and spread
        if self.best_bid and self.best_ask:
            # If there are both bid and ask, use the average of the two
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
        elif self.best_bid:
            # If there is only a bid, use the bid as the mid price
            self.mid_price = self.best_bid
            self.spread = 0.0
        elif self.best_ask:
            # If there is only an ask, use the ask as the mid price
            self.mid_price = self.best_ask
            self.spread = 0.0
        else:
            # No quotes, use last trade price
            self.mid_price = self.last_trade_price
            self.spread = 0.0

        
    def _clean_heaps(self):
        """ Remove cancelled orders from the heaps """

        # Clean buy orders. This works by popping the best buy order each iteration and
        # checking if the order is still in the order book. If it is, we add it to a temporary list.
        # We then push the temporary list back into the heap. We therefore completely drain the heap
        # first and the rebuild it with the orders that are still in the order book.
        temp = []
        while self.buy_orders:
            price, timestamp, order_id = heapq.heappop(self.buy_orders)
            if order_id in self.orders_dict:
                temp.append((price, timestamp, order_id))

        for item in temp:
            heapq.heappush(self.buy_orders, item)

        # Clean sell orders
        temp = []
        while self.sell_orders:
            price, timestamp, order_id = heapq.heappop(self.sell_orders)
            if order_id in self.orders_dict:
                temp.append((price, timestamp, order_id))

        for item in temp:
            heapq.heappush(self.sell_orders, item)


    def get_book_depth(self, levels: int = 5) -> Dict:
        """ Get the book depth for the order book """

        # Aggregate the order book by price level
        buy_depth = {}
        sell_depth = {}

        # Process buy orders
        for neg_price, timestamp, order_id in self.buy_orders:
            if order_id in self.orders_dict:
                price = -neg_price
                order = self.orders_dict[order_id]
                if price not in buy_depth:
                    buy_depth[price] = 0
                buy_depth[price] += order.quantity

        # Process sell orders
        for price, timestamp, order_id in self.sell_orders:
            if order_id in self.orders_dict:
                order = self.orders_dict[order_id]
                if price not in sell_depth:
                    sell_depth[price] = 0
                sell_depth[price] += order.quantity

        # Sort by price level and limit to requested levels
        buy_levels = sorted(buy_depth.items(), reverse=True)[:levels]
        sell_levels = sorted(sell_depth.items())[:levels]

        return {
            'bids': buy_levels,
            'asks': sell_levels,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread
        }
    

    def get_trader_orders(self, trader_id: str) -> List[Dict]:
        """ Get the orders for a specific trader """

        return [order for order in self.orders_dict.values() if order.trader_id == trader_id]
    

    def tick(self):
        """ Advance the order book by one tick """
        self.current_timestamp += 1
        
    
