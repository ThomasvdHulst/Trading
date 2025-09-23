""" This file contains the data generator for the order book. """

# Import used libraries
from message_parser import Message
from order_book import OrderBook
import time
from typing import List, Dict, Tuple
import numpy as np
import random


class DataGenerator:
    """ This class generates the data for the order book. """

    def __init__(self):
        self.base_price = 100.0
        self.current_price = self.base_price
        self.volatility = 0.05  
        self.mean_reversion = 0.001  # Keep lower to allow more drift
        self.order_id_counter = 0

    def generate_opening_book(self) -> List[Message]:
        """ This method generates the opening book for the order book. 
        
        Returns:
            List of messages representing the opening book.
        """
        
        messages = []

        for i in range(10):
            price = round(self.current_price - (i + 1) * 0.01, 2)
            quantity = round(np.random.exponential(100), 0)
            messages.append(Message.create_add_order(self._get_order_id(), 'BID', price, quantity))

        for i in range(10):
            price = round(self.current_price + (i + 1) * 0.01, 2)
            quantity = round(np.random.exponential(100), 0)
            messages.append(Message.create_add_order(self._get_order_id(), 'ASK', price, quantity))

        return messages


    def generate_random_walk(self, n_messages: int = 1000) -> List[Message]:
        """ This method generates the random walk for the order book. 
        
        Args:
            n_messages: Number of messages to generate.

        Returns:
            List of messages representing the random walk.
        """
        
        messages = []
        messages.extend(self.generate_opening_book())

        # Track active orders for cancellation
        active_orders = {}

        for i in range(n_messages):
            
            # Update price
            price_change = np.random.normal(0, self.volatility)
            self.current_price += price_change
            # Include mean reversion
            self.current_price += self.mean_reversion * (self.base_price - self.current_price)

            # Decide message type
            rand = random.random()
            
            if rand < 0.5: # 50% of the time we add a limit order
                message = self._generate_limit_order()
                messages.append(message)
                active_orders[message.order_id] = message

            elif rand < 0.7: # 20% of the time we add a market order
                message = self._generate_market_order()
                messages.append(message)

            elif rand < 0.9 and active_orders: # 20% of the time we cancel an order, if possible
                message, active_orders = self._generate_cancel_order(active_orders) # Return updated active orders
                messages.append(message)

            else: # If we do not have active orders, we add a limit order
                message = self._generate_limit_order()
                messages.append(message)
                active_orders[message.order_id] = message

        return messages


    def _generate_limit_order(self) -> Message:
        """ This method generates a limit order. 
        
        Returns:
            Message representing the limit order.
        """

        # Randomly choose side
        side = 'BID' if random.random() < 0.5 else 'ASK'

        # Price relative to current price
        if side == 'BID':
            offset = np.random.exponential(0.1) 
            price = self.current_price - offset
        else:
            offset = np.random.exponential(0.1)  
            price = self.current_price + offset
        
        quantity = np.random.exponential(100)

        return Message.create_add_order(
            order_id = self._get_order_id(),
            side = side,
            price = round(price, 2),
            quantity = round(quantity, 0)
        )


    def _generate_market_order(self) -> Message:
        """ This method generates a market order. 
        
        Returns:
            Message representing the market order.
        """
        
        is_buy = random.random() < 0.5
        price = self.current_price + np.random.normal(0, 0.01)
        quantity = np.random.exponential(50)

        return Message.create_trade(
            price = round(price, 2),
            quantity = round(quantity, 0),
            is_aggressor = is_buy
        )


    def _generate_cancel_order(self, active_orders: Dict[int, Message]) -> Tuple[Message, Dict[int, Message]]:
        """ This method generates a cancel order. 
        
        Args:
            active_orders: Dictionary of active orders.

        Returns:
            Tuple containing the cancel order message and the updated active orders dictionary.
        """

        order_to_cancel = random.choice(list(active_orders.values()))
        message = Message.create_cancel_order(
            order_id = order_to_cancel.order_id,
            side = order_to_cancel.side,
            price = order_to_cancel.price,
            quantity = order_to_cancel.quantity
        )
        del active_orders[order_to_cancel.order_id]
        return message, active_orders


    def _get_order_id(self) -> int:
        """ This method gets the next order ID. """
        self.order_id_counter += 1
        return self.order_id_counter
    