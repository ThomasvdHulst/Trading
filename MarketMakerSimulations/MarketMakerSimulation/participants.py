""" This file contains the participant classes for the simulation """

# Import used libraries
import random
import numpy as np
from order_book import Side


class MarketParticipant:
    """ Base class for all market participants """

    def __init__(self, trader_id, config):
        self.trader_id = trader_id
        self.config = config
        self.trades_executed = 0
        self.total_volume = 0


    def generate_order(self, order_book, market_state):
        """ Generate an order (to be implemented by subclasses) """
        raise NotImplementedError("Subclasses must implement this method")
    

class InformedTrader(MarketParticipant):
    """ Trader with information about future price movements """

    def __init__(self, trader_id, config):
        super().__init__(trader_id, config)
        self.information_accuracy = 0.7 # 70% accurate predictions
        self.aggressive_factor = 2 # How aggressive the trader is


    def generate_order(self, order_book, market_state):
        """ Generate an order based on our information """

        if random.random() > self.config.INFORMED_TRADER_PROB:
            return None # No order
        
        book_depth = order_book.get_book_depth()
        mid_price = book_depth['mid_price']

        # Simulate private information about future price
        if random.random() < self.information_accuracy:
            # Correct prediction
            true_direction = market_state.get('future_direction', 1)
        else:
            # Wrong prediction
            true_direction = -market_state.get('future_direction', 1)

        # Decide order parameters
        if true_direction > 0:
            # Expect price to go up, buy aggressively
            side = Side.BUY
            # Willing to pay about current ask
            if book_depth['best_ask']:
                price = book_depth['best_ask'] + self.config.TICK_SIZE * self.aggressive_factor
            else:
                price = mid_price * 1.001
        else:
            # Expect price to go down, sell aggressively
            side = Side.SELL
            # Willing to sell below current bid
            if book_depth['best_bid']:
                price = book_depth['best_bid'] - self.config.TICK_SIZE * self.aggressive_factor
            else:
                price = mid_price * 0.999

        # Random quantity
        quantity = random.randint(1, 5) * self.config.LOT_SIZE

        order_id = order_book.add_order(side, price, quantity, self.trader_id)

        if order_id > 0:
            self.trades_executed += 1
            self.total_volume += quantity

        return order_id
    

class NoiseTrader(MarketParticipant):
    """ Random liquidity trader """

    def __init__(self, trader_id, config):
        super().__init__(trader_id, config)
        self.price_sensitivity = 0.002 # How far from mid price to trade

    
    def generate_order(self, order_book, market_state):
        """ Generate an order based on market conditions """

        if random.random() > self.config.NOISE_TRADER_PROB:
            return None # No order
        
        book_depth = order_book.get_book_depth()
        mid_price = book_depth['mid_price']

        # Random side
        side = Side.BUY if random.random() < 0.5 else Side.SELL

        # Random price around mid - sometimes aggressive (crosses spread)
        price_offset = random.gauss(0, self.price_sensitivity * mid_price)
        
        # 30% chance to be aggressive (cross the spread)
        if random.random() < 0.3:
            if side == Side.BUY:
                # Aggressive buy: willing to pay above mid (hit ask)
                price = mid_price + abs(price_offset)
            else:
                # Aggressive sell: willing to sell below mid (hit bid)  
                price = mid_price - abs(price_offset)
        else:
            # Passive order: provide liquidity
            if side == Side.BUY:
                # Passive buy: bid below mid
                price = mid_price - abs(price_offset)
            else:
                # Passive sell: offer above mid
                price = mid_price + abs(price_offset)

        # Random quantity
        quantity = random.randint(1, 10) * self.config.LOT_SIZE

        order_id = order_book.add_order(side, price, quantity, self.trader_id)

        if order_id > 0:
            self.trades_executed += 1
            self.total_volume += quantity

        return order_id
    

class Arbitrageur(MarketParticipant):
    """ Keeps prices in line with fair value """

    def __init__(self, trader_id, config):
        super().__init__(trader_id, config)
        self.fair_value = config.INITIAL_PRICE
        self.threshold = 0.001 # Minimum profit to trade

    
    def generate_order(self, order_book, market_state):
        """ Trade when price is too far from fair value """

        book_depth = order_book.get_book_depth()

        # Update fair value
        if 'fair_value' in market_state:
            self.fair_value = market_state['fair_value']

        # Check for arbitrage opportunities
        if book_depth['best_ask'] and book_depth['best_ask'] < self.fair_value * (1 - self.threshold):
            # Ask is too low, buy
            side = Side.BUY
            price = book_depth['best_ask']
            quantity = random.randint(5, 15) * self.config.LOT_SIZE

            order_id = order_book.add_order(side, price, quantity, self.trader_id)

            if order_id > 0:
                self.trades_executed += 1
                self.total_volume += quantity
            
            return order_id
        
        elif book_depth['best_bid'] and book_depth['best_bid'] > self.fair_value * (1 + self.threshold):
            # Bid is too high, sell
            side = Side.SELL
            price = book_depth['best_bid']
            quantity = random.randint(5, 15) * self.config.LOT_SIZE

            order_id = order_book.add_order(side, price, quantity, self.trader_id)
            
            if order_id > 0:
                self.trades_executed += 1
                self.total_volume += quantity

            return order_id
        
        return None
    

class MomentumTrader(MarketParticipant):
    """ Follows the trend """

    def __init__(self, trader_id, config):
        super().__init__(trader_id, config)
        self.price_history = []
        self.lookback_period = 10 # Number of ticks to consider
        self.momentum_threshold = 0.0005 # Minimum return change to trigger trade


    def generate_order(self, order_book, market_state):
        """ Generate an order based on price history """

        book_depth = order_book.get_book_depth()
        mid_price = book_depth['mid_price']

        # Update price history
        self.price_history.append(mid_price)
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)

        # Need enough history
        if len(self.price_history) < self.lookback_period:
            return None
        
        # Calculate momentum
        returns = [(self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1] for i in range(1, len(self.price_history))]
        momentum = np.mean(returns)

        # Trade on strong momentum
        if abs(momentum) > self.momentum_threshold:
            if momentum > 0:
                # Positive momentum, buy
                side = Side.BUY
                if book_depth['best_ask']:
                    price = book_depth['best_ask']
                else:
                    price = mid_price * 1.001
            else:
                # Negative momentum, sell
                side = Side.SELL
                if book_depth['best_bid']:
                    price = book_depth['best_bid']
                else:
                    price = mid_price * 0.999
                    
            quantity = random.randint(2, 8) * self.config.LOT_SIZE

            order_id = order_book.add_order(side, price, quantity, self.trader_id)

            if order_id > 0:
                self.trades_executed += 1
                self.total_volume += quantity

            return order_id
        
        return None
        
        
        