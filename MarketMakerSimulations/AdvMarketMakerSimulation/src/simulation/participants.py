""" This file contains the code for all market participants. """

# Import used libraries
import random
import numpy as np
from typing import List
from abc import ABC, abstractmethod

from market_maker.core.base import IOrderBook, MarketState, Order, Side, OrderType
from config.config import Config


class MarketParticipant(ABC):
    """ Base class for all market participants. """

    def __init__(self, trader_id: str):
        """ Initialize participant. 
        
        Args:
            trader_id: Unique trader identifier
        """

        self.trader_id = trader_id
        self.trades_executed = 0
        self.total_volume = 0


    @abstractmethod
    def generate_orders(self, order_book: IOrderBook, market_state: MarketState, fair_value: float) -> None:
        """ Generate orders based on market state and fair value.
        
        Args:
            order_book: Order book
            market_state: Current market state
            fair_value: Current fair value
        """

        pass


class InformedTrader(MarketParticipant):
    """ Trader with information about future price movements. 
    These traders simulated informed flow that market makers lose
    money to (adverse selection). """

    def __init__(self, trader_id: str, probability: float, accuracy: float):
        """ Initialize informed trader.
        
        Args:
            trader_id: Unique trader identifier
            probability: Probability of trading
            accuracy: Accuracy of price predictions
        """

        super().__init__(trader_id)
        self.probability = probability
        self.accuracy = accuracy

    
    def generate_orders(self, order_book: IOrderBook, market_state: MarketState, fair_value: float) -> None:
        """ Generate aggressive order based on information. """

        if random.random() > self.probability:
            return
        
        # Predict direction (with accuracy)
        true_direction = 1 if fair_value > market_state.mid_price else -1
        if random.random() > self.accuracy:
            true_direction *= -1 # Wrong prediction

        if true_direction > 0:
            # Expect price up - buy aggressively
            side = Side.BUY
            if market_state.best_ask:
                price = market_state.best_ask * 1.001 # Cross spread
            else:
                price = market_state.mid_price * 1.001
        else:
            # Expect price down - sell aggressively
            side = Side.SELL
            if market_state.best_bid:
                price = market_state.best_bid * 0.999 # Cross spread
            else:
                price = market_state.mid_price * 0.999

        quantity = random.randint(5, 15) * 100 # 500-1500 shares

        order = Order(
            order_id = 0,
            trader_id = self.trader_id,
            side = side,
            price = price,
            quantity = quantity,
            timestamp = 0,
            order_type = OrderType.LIMIT
        )

        order_id = order_book.add_order(order)
        if order_id > 0:
            self.trades_executed += 1
            self.total_volume += quantity

    
class NoiseTrader(MarketParticipant):
    """ Trader that trades randomly. These simulate
    retail flow and other uninformed trading that provides
    liquidity to the market. """

    def __init__(self, trader_id: str , probability: float, price_sensitivity: float):
        """ Initialize noise trader.
        
        Args:
            trader_id: Unique trader identifier
            probability: Probability of trading
            price_sensitivity: How far from mid price to trade
        """

        super().__init__(trader_id)
        self.probability = probability
        self.price_sensitivity = price_sensitivity


    def generate_orders(self, order_book: IOrderBook, market_state: MarketState, fair_value: float) -> None:
        """ Generate random orders. """

        if random.random() > self.probability:
            return
        
        # Random side
        side = Side.BUY if random.random() < 0.5 else Side.SELL

        # Random price around mid
        price_offset = np.random.normal(0, self.price_sensitivity * market_state.mid_price)

        # 30% chance to cross spread (aggressive)
        if random.random() < 0.3:
            if side == Side.BUY and market_state.best_ask:
                price = market_state.best_ask + abs(price_offset) * 0.1
            elif side == Side.SELL and market_state.best_bid:
                price = market_state.best_bid - abs(price_offset) * 0.1
            else:
                price = market_state.mid_price + price_offset
        
        else:
            # Passive order
            if side == Side.BUY:
                price = market_state.mid_price - abs(price_offset)
            else:
                price = market_state.mid_price + abs(price_offset)

        quantity = random.randint(1, 10) * 100 # 100-1000 shares

        order = Order(
            order_id = 0,
            trader_id = self.trader_id,
            side = side,
            price = price,
            quantity = quantity,
            timestamp = 0,
            order_type = OrderType.LIMIT
        )

        order_id = order_book.add_order(order)
        if order_id > 0:
            self.trades_executed += 1
            self.total_volume += quantity


class Arbitrageur(MarketParticipant):
    """ Keeps prices in line with fair value. These traders ensure
    price efficiency by trading when prices deviate from fair value. """

    def __init__(self, trader_id: str, threshold: float):
        """ Initialize arbitrageur.
        
        Args:
            trader_id: Unique trader identifier
            probability: Probability of trading
            threshold: Minimum profit threshold to trade
        """

        super().__init__(trader_id)
        self.threshold = threshold

    
    def generate_orders(self, order_book: IOrderBook, market_state: MarketState, fair_value: float) -> None:
        """ Trade when prices deviate from fair value. """

        # Check for arbitrage opportunities
        if market_state.best_ask and market_state.best_ask < fair_value * (1 - self.threshold):
            # Ask too low - buy
            order = Order(
                order_id = 0,
                trader_id = self.trader_id,
                side = Side.BUY,
                price = market_state.best_ask,
                quantity = random.randint(10, 20) * 100,
                timestamp = 0,
                order_type = OrderType.LIMIT
            )
            
            order_book.add_order(order)

        elif market_state.best_bid and market_state.best_bid > fair_value * (1 + self.threshold):
            # Bid too high - sell
            order = Order(
                order_id = 0,
                trader_id = self.trader_id,
                side = Side.SELL,
                price = market_state.best_bid,
                quantity = random.randint(10, 20) * 100,
                timestamp = 0,
                order_type = OrderType.LIMIT
            )
            
            order_book.add_order(order)


class MomentumTrader(MarketParticipant):
    """ Follows price trends. These traders can amplify price moves
    and create additional volatility. """

    def __init__(self, trader_id: str, lookback: int, threshold: float):
        """ Initialize momentum trader.
        
        Args:
            trader_id: Unique trader identifier
            lookback: Number of previous ticks to consider
            threshold: Minimum momentum to trade
        """

        super().__init__(trader_id)
        self.lookback = lookback
        self.threshold = threshold
        self.price_history = []

    
    def generate_orders(self, order_book: IOrderBook, market_state: MarketState, fair_value: float) -> None:
        """ Trade based on momentum. """

        # Update price history
        self.price_history.append(market_state.mid_price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

        # Need enought history
        if len(self.price_history) < self.lookback:
            return
        
        # Calculate momentum
        returns = [
            (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
            for i in range(1, len(self.price_history))
        ]

        momentum = np.mean(returns)

        # Trade if momentum strong enough
        if abs(momentum) > self.threshold:
            if momentum > 0:
                # Positive momentum - buy
                side = Side.BUY
                price = market_state.best_ask if market_state.best_ask else market_state.mid_price * 1.001
            else:
                # Negative momentum - sell
                side = Side.SELL
                price = market_state.best_bid if market_state.best_bid else market_state.mid_price * 0.999

            order = Order(
                order_id = 0,
                trader_id = self.trader_id,
                side = side,
                price = price,
                quantity = random.randint(5, 10) * 100,
                timestamp = 0,
                order_type = OrderType.LIMIT
            )

            order_book.add_order(order)


def create_participants(config: Config) -> List[MarketParticipant]:
    """ Create market participants from configuration. 
    
    Args:
        config: Full configuration

    Returns:
        List of market participants
    """

    participants = []

    # Create informed traders
    for i in range(config.participants.informed_traders.count):
        trader = InformedTrader(
            trader_id = f"INF_{i+1:02d}",
            probability = config.participants.informed_traders.probability,
            accuracy = config.participants.informed_traders.accuracy
        )
        participants.append(trader)

    # Create noise traders
    for i in range(config.participants.noise_traders.count):
        trader = NoiseTrader(
            trader_id = f"NOISE_{i+1:02d}",
            probability = config.participants.noise_traders.probability,
            price_sensitivity = config.participants.noise_traders.price_sensitivity
        )
        participants.append(trader)

    # Create arbitrageurs
    for i in range(config.participants.arbitrageurs.count):
        trader = Arbitrageur(
            trader_id = f"ARB_{i+1:02d}",
            threshold = config.participants.arbitrageurs.threshold
        )
        participants.append(trader)

    # Create momentum traders
    for i in range(config.participants.momentum_traders.count):
        trader = MomentumTrader(
            trader_id = f"MOM_{i+1:02d}",
            lookback = config.participants.momentum_traders.lookback_period,
            threshold = config.participants.momentum_traders.momentum_threshold
        )
        participants.append(trader)

    return participants
        
        