""" This file generates realistic market data for testing. It uses
statistical models to generate a realistic order flow. """


# Import used libraries
import numpy as np
import pandas as pd
from typing import List
import random
from ..core.message_parser import Message


class MarketDataGenerator:
    """ Generates realistic market data with various market conditions.
    Simulates different participant types and market regimes.
    """

    def __init__(self, base_price: float = 100.0, tick_size: float = 0.01):
        """
        Initialize the market data generator.

        Args:
            base_price: The base price of the asset.
            tick_size: The size of a tick.
        """
        self.base_price = base_price
        self.tick_size = tick_size
        self.current_price = base_price
        self.order_id_counter = 1
        self.time = 0

        # Market dynamics parameters
        self.volatility = 0.001 # Price volatility
        self.mean_reversion = 0.1 # Mean reversion strength
        self.spread_mean = 0.02 # Average spread
        self.depth_lambda = 100 # Average order size

        # Order arrival rates (Poisson process)
        self.limit_order_rate = 10 # Orders per time unit
        self.market_order_rate = 2 # Trades per time unit
        self.cancel_rate = 5 # Cancels per time unit


    def generate_opening_book(self, n_levels: int = 10) -> List[Message]:
        """ Generate initial order book state.
        
        Args:
            n_levels: Number of price levels to generate.

        Returns:
            List of messages representing the initial order book state.
        """
        
        messages = []

        # Generate bid side
        for i in range(n_levels):
            price = self.current_price - (i + 1) * self.tick_size
            quantity = np.random.exponential(self.depth_lambda)

            messages.append(Message.create_add_order(
                order_id = self._get_order_id(),
                side = 'BID',
                price = round(price, 2),
                quantity = round(quantity, 0)
            ))


        # Generate ask side
        for i in range(n_levels):
            price = self.current_price + (i + 1) * self.tick_size
            quantity = np.random.exponential(self.depth_lambda)

            messages.append(Message.create_add_order(
                order_id = self._get_order_id(),
                side = 'ASK',
                price = round(price, 2),
                quantity = round(quantity, 0)
            ))

        return messages
    

    def generate_random_walk(self, n_messages: int = 1000) -> List[Message]:
        """ Generate market data following a random walk with realistic dynamics.
        Includes limit orders, market orders and cancellations.

        Args:
            n_messages: Number of messages to generate.

        Returns:
            List of messages representing the market data.
        """

        messages = []

        # Start with the opening book
        messages.extend(self.generate_opening_book())

        # Track active orders for cancellation
        active_orders = {}

        for _ in range(n_messages):
            # Generate the time step based on the arrival rates,
            # if the arrival rate increases, the time step will be smaller
            # and the order book will be more volatile.
            self.time += np.random.exponential(1.0 / (
                self.limit_order_rate + self.market_order_rate + self.cancel_rate
            ))

            # Update price (mean-reverting random walk)
            price_change = np.random.normal(0, self.volatility)
            self.current_price += price_change
            self.current_price += self.mean_reversion * (self.base_price - self.current_price)

            # Decide message type
            rand = random.random()

            if rand < 0.5: # Limit order
                message = self._generate_limit_order()
                messages.append(message)
                active_orders[message.order_id] = message

            elif rand < 0.7: # Market order (trade)
                message = self._generate_market_order()
                messages.append(message)

            elif rand < 0.9 and active_orders: # Cancel order
                order_to_cancel = random.choice(list(active_orders.values()))
                message = Message.create_cancel_order(
                    order_id = order_to_cancel.order_id,
                    side = order_to_cancel.side,
                    price = order_to_cancel.price,
                    quantity = order_to_cancel.quantity
                )
                messages.append(message)
                del active_orders[order_to_cancel.order_id]

            else: # Another limit order, when we do not have active_orders (or 10% of the time)
                message = self._generate_limit_order()
                messages.append(message)
                active_orders[message.order_id] = message

        return messages
    

    def generate_momentum_scenario(self, direction: str = 'up', n_messages: int = 500) -> List[Message]:
        """ Generate a momentum scenario with a specified direction.
        
        Args:
            direction: 'up' or 'down'
            n_messages: Number of messages to generate.

        Returns:
            List of messages representing the market data.
        """

        messages = []
        messages.extend(self.generate_opening_book())

        # Momentum parameters
        drift = 0.001 if direction == 'up' else -0.001

        for i in range(n_messages):
            self.time += np.random.exponential(0.01)

            # Trending price with momentum
            self.current_price += drift + np.random.normal(0, self.volatility)

            # More aggressive orders in trend direction
            if random.random() < 0.7:
                if direction == 'up':
                    # More buy orders
                    message = self._generate_buy_order()
                else:
                    # More sell orders
                    message = self._generate_sell_order()

                messages.append(message)
            else:
                # Regular order flow
                message = self._generate_limit_order()
                messages.append(message)

        return messages
    

    def generate_high_volatility_scenario(self, n_messages: int = 500) -> List[Message]:
        """ Generate high volatility market conditions.
        
        Args:
            n_messages: Number of messages to generate.

        Returns:
            List of messages representing the market data.
        """

        messages = []
        messages.extend(self.generate_opening_book())

        # Increase volatility
        old_volatility = self.volatility
        self.volatility *= 5

        for _ in range(n_messages):
            self.time += np.random.exponential(0.005) # Faster messages

            # Volatile price movement
            self.current_price += np.random.normal(0, self.volatility)

            # Mix of orders with wider spreads
            if random.random() < 0.4:
                # Market orders causing trades
                message = self._generate_market_order()
            else:
                # Wider limit orders
                message = self._generate_limit_order(spread_multiplier = 3)

            messages.append(message)
        
        self.volatility = old_volatility
        return messages
    

    def _generate_limit_order(self, spread_multiplier: float = 1.0) -> Message:
        """ Generate a limit order with realistic spread.
        
        Args:
            spread_multiplier: Multiplier for the spread.

        Returns:
            Message representing the limit order.
        """

        # Randomly choose side
        side = 'BID' if random.random() < 0.5 else 'ASK'

        # Price relative to current price
        if side == 'BID':
            offset = np.random.exponential(self.spread_mean * spread_multiplier)
            price = self.current_price - offset
        else:
            offset = np.random.exponential(self.spread_mean * spread_multiplier)
            price = self.current_price + offset

        # Order size (exponential distribution)
        quantity = np.random.exponential(self.depth_lambda)

        return Message.create_add_order(
            order_id = self._get_order_id(),
            side = side,
            price = round(price, 2),
            quantity = round(quantity, 0)
        )
    

    def _generate_market_order(self) -> Message:
        """ Generate a market order.
        
        Returns:
            Message representing the market order.
        """

        is_buy = random.random() < 0.5

        # Trade at or near current price
        price = self.current_price + np.random.normal(0, self.tick_size)
        quantity = np.random.exponential(self.depth_lambda/2) # Smaller than limit orders

        return Message.create_trade(
            price = round(price, 2),
            quantity = round(quantity, 0),
            is_buy_aggressor = is_buy
        )
    

    def _generate_buy_order(self) -> Message:
        """ Generate a buy order.
        
        Returns:
            Message representing the buy order.
        """

        price = self.current_price - np.random.exponential(self.spread_mean)
        quantity = np.random.exponential(self.depth_lambda)

        return Message.create_add_order(
            order_id = self._get_order_id(),
            side = 'BID',
            price = round(price, 2),
            quantity = round(quantity, 0)
        )
    

    def _generate_sell_order(self) -> Message:
        """ Generate a sell order.

        Returns:
            Message representing the sell order.
        """

        price = self.current_price + np.random.exponential(self.spread_mean)
        quantity = np.random.exponential(self.depth_lambda)
        
        return Message.create_add_order(
            order_id = self._get_order_id(),
            side = 'ASK',
            price = round(price, 2),
            quantity = round(quantity, 0)
        )
    

    def _get_order_id(self) -> int:
        """ Get a new order ID. 
        
        Returns:
            New order ID.
        """

        order_id = self.order_id_counter
        self.order_id_counter += 1
        return order_id
    

    def generate_dataset(self, scenarios: List[str] = None) -> pd.DataFrame:
        """ Generate a dataset of market data with multiple scenarios.

        Args:
            scenarios: List of scenario names to generate.

        Returns:
            DataFrame with market data.
        """

        if scenarios is None:
            scenarios = ['normal', 'momentum_up', 'momentum_down', 'volatile']

        all_messages = []

        for scenario in scenarios:
            if scenario == 'normal':
                messages = self.generate_random_walk(1000)
            elif scenario == 'momentum_up':
                messages = self.generate_momentum_scenario(direction = 'up', n_messages = 500)
            elif scenario == 'momentum_down':
                messages = self.generate_momentum_scenario(direction = 'down', n_messages = 500)
            elif scenario == 'volatile':
                messages = self.generate_high_volatility_scenario(500)
            else:
                continue

            # Convert to dict format
            for msg in messages:
                all_messages.append({
                    'timestamp': msg.timestamp,
                    'type': msg.msg_type.value,
                    'order_id': msg.order_id,
                    'side': msg.side,
                    'price': msg.price,
                    'quantity': msg.quantity,
                    'is_aggressor': msg.is_aggressor,
                    'scenario': scenario,
                })

        return pd.DataFrame(all_messages)
    
