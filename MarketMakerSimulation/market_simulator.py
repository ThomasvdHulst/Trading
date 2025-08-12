""" This file contains the market simulator class for the simulation """

# Import used libraries
import numpy as np
import random
from order_book import OrderBook
from market_maker import MarketMaker
from participants import InformedTrader, NoiseTrader, Arbitrageur, MomentumTrader


class MarketSimulator:
    """ Main class for running the market simulation """

    def __init__(self, config):
        self.config = config

        # Set seed for reproducibility
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        # Initialize order book
        self.order_book = OrderBook(config)

        # Initialize market maker
        self.market_maker = MarketMaker(config, trader_id="MM_01")

        # Initialize other participants
        self.participants = []
        for i in range(config.NUMB_INFORMED_TRADERS):
            self.participants.append(InformedTrader(f"INF_{i+1}", config))
        for i in range(config.NUMB_NOISE_TRADERS):
            self.participants.append(NoiseTrader(f"NOISE_{i+1}", config))
        for i in range(config.NUMB_ARBITRAGEURS):
            self.participants.append(Arbitrageur(f"ARB_{i+1}", config))
        for i in range(config.NUMB_MOMENTUM_TRADERS):
            self.participants.append(MomentumTrader(f"MOM_{i+1}", config))

        # Market state
        self.fair_value = config.INITIAL_PRICE
        self.volatility = config.VOLATILITY
        self.current_tick = 0

        # Tracking
        self.price_history = []
        self.spread_history = []
        self.volume_history = []
        self.mm_pnl_history = []
        self.mm_inventory_history = []

    
    def simulate_price_dynamics(self):
        """ Simulate underlying fair value changes """

        # Random walk with reversion to mean
        drift = -self.config.MEAN_REVERSION_SPEED * (self.fair_value - self.config.INITIAL_PRICE)
        diffusion = self.volatility * np.random.randn()

        self.fair_value += drift + diffusion
        self.fair_value = max(self.fair_value, self.config.INITIAL_PRICE * 0.5) # Floor at 50% of initial price
        self.fair_value = min(self.fair_value, self.config.INITIAL_PRICE * 1.5) # Ceiling at 150% of initial price

        # Occasional jumps (new events)
        if random.random() < self.config.EVENT_PROBABILITY:
            jump_size = random.choice([-0.02, -0.01, 0.01, 0.02]) * self.fair_value
            self.fair_value += jump_size


    def generate_market_state(self):
        """ Generate the current market state for participants """

        # Base on recent price movement
        recent_change = self.fair_value - self.config.INITIAL_PRICE
        momentum_prob = 0.5 + 0.1 * np.tanh(recent_change)  # Slight momentum bias
        future_direction = 1 if random.random() < momentum_prob else -1

        market_state = {
            'fair_value': self.fair_value,
            'volatility': self.volatility,
            'future_direction': future_direction,
            'tick': self.current_tick,
        }

        return market_state
    

    def process_participant_orders(self, market_state):
        """ Let each participant generate orders """

        # Shuffle order to avoid bias
        participants_shuffled = self.participants.copy()
        random.shuffle(participants_shuffled)

        for participant in participants_shuffled:

            participant.generate_order(self.order_book, market_state)



    def process_market_maker_trades(self):
        """ Process any trades that the market maker participated in """

        # Only check trades from current tick
        current_tick_trades = [trade for trade in self.order_book.trade_history 
                              if trade['timestamp'] == self.current_tick]
        for trade in current_tick_trades:
            if trade['buy_trader'] == self.market_maker.trader_id or trade['sell_trader'] == self.market_maker.trader_id:
                self.market_maker.on_trade(trade)

                
    def update_tracking(self):
        """ Update tracking variables """

        book_depth = self.order_book.get_book_depth()

        # Price and spread
        self.price_history.append(book_depth['mid_price'])
        self.spread_history.append(book_depth['spread'])

        # Volume (trades in this tick)
        tick_volume = sum(trade['quantity'] for trade in self.order_book.trade_history if trade['timestamp'] == self.current_tick)
        self.volume_history.append(tick_volume)

        # Market maker metrics
        self.market_maker.update_pnl(book_depth['mid_price'])
        mm_metrics = self.market_maker.get_metrics()
        self.mm_pnl_history.append(mm_metrics['total_pnl'])
        self.mm_inventory_history.append(mm_metrics['inventory'])


    def run_tick(self):
        """ Run a single tick of the simulation """

        # Advance time
        self.current_tick += 1
        self.order_book.tick()

        # Update underlying price dynamics
        self.simulate_price_dynamics()

        # Generate market state
        market_state = self.generate_market_state()

        # Market maker updates quotes (if past warmup period)
        if self.current_tick > self.config.WARMUP_PERIOD:
            self.market_maker.update_quotes(self.order_book, market_state)

        # Other participants generate orders
        self.process_participant_orders(market_state)

        # Process market maker trades
        self.process_market_maker_trades()

        # Update tracking
        self.update_tracking()


    def run_simulation(self):
        """ Run the full simulation """

        print(f"Starting market simulation for {self.config.SIMULATION_TICKS} ticks")
        print(f"Initial price: {self.config.INITIAL_PRICE:.2f}")
        print(f"Market maker capital: ${self.config.INITIAL_CAPITAL:.2f}")
        print("\n")

        for tick in range(self.config.SIMULATION_TICKS):
            self.run_tick()

            # Print progress
            if (tick + 1) % 1000 == 0:
                mm_metrics = self.market_maker.get_metrics()
                book_depth = self.order_book.get_book_depth()

                print(f"Tick {tick + 1}/{self.config.SIMULATION_TICKS}")
                print(f"Mid price: {book_depth['mid_price']:.2f}")
                print(f"Spread: {book_depth['spread']:.3f}")
                print(f"Best bid: {book_depth['best_bid']}, Best ask: {book_depth['best_ask']}")
                print(f"MM Inventory: {mm_metrics['inventory']}")
                print(f"MM PnL: ${mm_metrics['total_pnl']:,.2f}")
                print(f"Total trades: {len(self.order_book.trade_history)}")
                print(f"MM trades executed: {len(self.market_maker.trades_executed)}")
                print(f"Active buy orders: {mm_metrics['active_buy_orders']}, Active sell orders: {mm_metrics['active_sell_orders']}")
                print("\n")

        print("Simulation complete")

        return self.get_results()
    

    def get_results(self):
        """ Get simulation results """

        mm_metrics = self.market_maker.get_metrics()

        results = {
            'price_history': self.price_history,
            'spread_history': self.spread_history,
            'volume_history': self.volume_history,
            'mm_pnl_history': self.mm_pnl_history,
            'mm_inventory_history': self.mm_inventory_history,
            'mm_final_metrics': mm_metrics,
            'total_trades': len(self.order_book.trade_history),
            'trade_history': self.order_book.trade_history,
            'final_order_book': self.order_book.get_book_depth(),
        }

        return results