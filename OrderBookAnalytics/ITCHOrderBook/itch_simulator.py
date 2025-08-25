""" This file contains the ITCH simulator. """

# Import used libraries
from typing import List
import random
from itch_parser import ITCHParser


class ITCHSimulator:
    """ Simulates ITCH message flow for testing. """

    def __init__(self, base_price: float = 100.0):
        self.base_price = base_price
        self.current_price = base_price
        self.order_id_counter = 1000000
        self.match_number_counter = 1
        self.parser = ITCHParser()

        # Track active orders for realistic cancels/executes
        self.active_orders = {} # order_id -> (side, price, shares)


    def generate_messages(self, n_messages: int) -> List[bytes]:
        """ Generate ITCH messages for testing. """
        messages = []

        messages = [None] * n_messages
        msg_index = 0

        # Initial book population
        for i in range(10):
            # Bids
            price = self.base_price - 0.01 * (i + 1)
            shares = random.randint(100, 5000)
            order_id = self._get_order_id()

            msg = self.parser.create_add_order_message(order_id, 'BID', price, shares)
            messages[msg_index] = msg
            msg_index += 1
            self.active_orders[order_id] = ('BID', price, shares)

            # Asks
            price = self.base_price + 0.01 * (i + 1)
            shares = random.randint(100, 5000)
            order_id = self._get_order_id()

            msg = self.parser.create_add_order_message(order_id, 'ASK', price, shares)
            messages[msg_index] = msg
            msg_index += 1
            self.active_orders[order_id] = ('ASK', price, shares)

        # Generate message flow
        for _ in range(n_messages - 20): # Extract 20 as we have 20 initial orders
            msg_type = self._choose_message_type()

            if msg_type == 'ADD':
                # New order
                side = random.choice(['BID', 'ASK'])
                if side == 'BID':
                    price = self.current_price - random.uniform(0.01, 0.10)
                else:
                    price = self.current_price + random.uniform(0.01, 0.10)

                shares = random.randint(100, 5000)
                order_id = self._get_order_id()

                msg = self.parser.create_add_order_message(order_id, side, price, shares)
                messages[msg_index] = msg
                msg_index += 1
                self.active_orders[order_id] = (side, price, shares)

            elif msg_type == 'EXECUTE' and self.active_orders:
                # Execute random order
                order_id = random.choice(list(self.active_orders.keys()))
                side, price, shares = self.active_orders[order_id]

                executed_shares = min(shares, random.randint(100, 1000))
                msg = self.parser.create_execute_message(order_id, executed_shares, self._get_match_number())
                messages[msg_index] = msg
                msg_index += 1

                # Update tracking
                remaining = shares - executed_shares
                if remaining > 0:
                    self.active_orders[order_id] = (side, price, remaining)
                else:
                    del self.active_orders[order_id]

                # Update price based on trade
                if side == 'BID':
                    self.current_price -= 0.001
                else:
                    self.current_price += 0.001

            elif msg_type == 'CANCEL' and self.active_orders:
                # Cancel random order
                order_id = random.choice(list(self.active_orders.keys()))
                side, price, shares = self.active_orders[order_id]

                cancelled_shares = shares if shares < 100 else random.randint(100, shares)
                msg = self.parser.create_cancel_message(order_id, cancelled_shares)
                messages[msg_index] = msg
                msg_index += 1

                # Update tracking
                remaining = shares - cancelled_shares
                if remaining > 0:
                    self.active_orders[order_id] = (side, price, remaining)
                else:
                    del self.active_orders[order_id]

            elif msg_type == 'DELETE' and self.active_orders:
                # Delete random order
                order_id = random.choice(list(self.active_orders.keys()))
                msg = self.parser.create_delete_message(order_id)
                messages[msg_index] = msg
                msg_index += 1
                del self.active_orders[order_id]

        return messages
    

    def _choose_message_type(self) -> str:
        """ Choose message based on probability. """
        rand = random.random()

        if rand < 0.40: # 40% adds
            return 'ADD'
        elif rand < 0.60: # 20% executes
            return 'EXECUTE'
        elif rand < 0.85: # 25% cancels
            return 'CANCEL'
        else: # 15% deletes
            return 'DELETE'
        

    def _get_order_id(self) -> int:
        """ Get new order ID. """
        order_id = self.order_id_counter
        self.order_id_counter += 1
        return order_id
    

    def _get_match_number(self) -> int:
        """ Get new match number. """
        match_number = self.match_number_counter
        self.match_number_counter += 1
        return match_number
    
    