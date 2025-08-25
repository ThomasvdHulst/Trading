""" This file contains the simulator implementation. """

# Import used libraries
import numpy as np
import random
from typing import Dict, Any

from market_maker.core.market_maker import MarketMaker
from simulation.order_book import OrderBook
from simulation.participants import create_participants
from config.config import Config


class MarketSimulator:
    """ Run market simulation with market maker and participants. """

    def __init__(self, config: Config):
        """ Initialize simulator.
        
        Args:
            config: Full configuration
        """

        self.config = config

        # Set random seeds for reproducibility
        random.seed(config.simulation.random_seed)
        np.random.seed(config.simulation.random_seed)

        # Initialize components
        print("Initializing market simulation...")

        # Order book
        self.order_book = OrderBook(config.market)

        # Market maker
        self.market_maker = MarketMaker(config)

        # Other participants
        self.participants = create_participants(config)

        # Market dynamics
        self.fair_value = config.market.initial_price
        self.current_tick = 0

        # Tracking
        self.price_history = []
        self.spread_history = []
        self.volume_history = []
        self.mm_metrics_history = []

        print(f"Simulation initialized with {len(self.participants)} participants")


    def simulate_price_dynamics(self) -> None:
        """ Simulate underlying fair value changes. """

        # Random walk with mean reversion
        drift = -self.config.market_dynamics.mean_reversion_speed * (self.fair_value - self.config.market.initial_price)
        diffusion = self.config.market_dynamics.volatility * np.random.randn()

        self.fair_value += drift + diffusion

        # Occasionally jump
        if random.random() < self.config.market_dynamics.event_probability:
            jump = np.random.choice([-0.02, -0.01, 0.01, 0.02]) * self.fair_value
            self.fair_value += jump

        # Keep in reasonable range
        self.fair_value = np.clip(
            self.fair_value,
            self.config.market.initial_price * 0.5,
            self.config.market.initial_price * 1.5,
        )


    def generate_participant_orders(self) -> None:
        """ Let other participants generate orders. """

        # Shuffle to avoid bias
        participants = self.participants.copy()
        random.shuffle(participants)

        market_state = self.order_book.get_market_state()

        for participant in participants:
            # Each participant decides to trade or not
            participant.generate_orders(
                self.order_book,
                market_state,
                self.fair_value
            )


    def process_trades(self) -> None:
        """ Process any trades that occured this tick. """

        trades = self.order_book.get_trades_for_tick(self.current_tick)

        for trade in trades:
            # Notify market maker of trades
            self.market_maker.on_trade(trade)


    def update_tracking(self) -> None:
        """ Update tracking metrics. """

        market_state = self.order_book.get_market_state()

        # Price and spread
        self.price_history.append(market_state.mid_price)
        self.spread_history.append(market_state.spread)

        # Volume
        trades = self.order_book.get_trades_for_tick(self.current_tick)
        tick_volume = sum(trade.quantity for trade in trades)
        self.volume_history.append(tick_volume)

        # Market maker metrics
        mm_metrics = self.market_maker.get_metrics()
        self.mm_metrics_history.append({
            'tick': self.current_tick,
            'position': mm_metrics['position']['position'],
            'pnl': mm_metrics['position']['total_pnl'],
            'volatility': mm_metrics['volatility']['current_volatility'],
        })


    def run_tick(self) -> None:
        """ Run a single simulation tick. """
        self.current_tick += 1
        self.order_book.tick()

        # Update market dynamics
        self.simulate_price_dynamics()

        # Market maker updates quotes
        if self.current_tick > self.config.simulation.warmup_period:
            market_state = self.order_book.get_market_state()
            market_state.fair_value = self.fair_value
            self.market_maker.update(self.order_book, market_state)

        # Other participants generate orders
        self.generate_participant_orders()

        # Process trades
        self.process_trades()

        # Update tracking
        self.update_tracking()


    def run(self) -> Dict[str, Any]:
        """ Run full simulation.
        
        Returns:
            Dictionary of results
        """

        print(f"\nStarting simulation for {self.config.simulation.ticks} ticks...")
        print(f"Initial price: {self.config.market.initial_price}")
        print(f"Market maker capital: {self.config.market_maker.initial_capital}")

        # Run simulation
        for tick in range(self.config.simulation.ticks):
            self.run_tick()

            # Print progress
            if (tick + 1) % 1000 == 0:
                self._print_progress(tick+1)

        print("\nSimulation complete!")

        # Compile results
        results = self._compile_results()
        self._print_summary(results)

        return results
    

    def _print_progress(self, tick: int) -> None:
        """Print simulation progress."""
        market_state = self.order_book.get_market_state()
        mm_metrics = self.market_maker.get_metrics()
        
        print(f"Tick {tick}/{self.config.simulation.ticks}")
        print(f"Price: {market_state.mid_price:.2f} (Fair: {self.fair_value:.2f})")
        print(f"Spread: {market_state.spread:.3f}")
        print(f"MM Position: {mm_metrics['position']['position']}")
        print(f"MM PnL: ${mm_metrics['position']['total_pnl']:,.2f}")
        print(f"Total trades: {len(self.order_book.trade_history)}")
        

    def _compile_results(self) -> Dict[str, Any]:
        """Compile simulation results."""
        mm_metrics = self.market_maker.get_metrics()
        
        return {
            'price_history': self.price_history,
            'spread_history': self.spread_history,
            'volume_history': self.volume_history,
            'mm_metrics_history': self.mm_metrics_history,
            'final_mm_metrics': mm_metrics,
            'trade_history': self.order_book.trade_history,
            'total_trades': len(self.order_book.trade_history),
        }
        

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print results summary."""
        mm_final = results['final_mm_metrics']
        
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        
        print(f"\nMarket Maker Performance:")
        print(f"Total PnL: ${mm_final['position']['total_pnl']:,.2f}")
        print(f"Realized PnL: ${mm_final['position']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL: ${mm_final['position']['unrealized_pnl']:,.2f}")
        print(f"Total Volume: {mm_final['position']['total_volume']:,}")
        print(f"Number of Trades: {mm_final['position']['num_trades']}")
        print(f"Final Position: {mm_final['position']['position']}")
        
        print(f"\nRisk Metrics:")
        print(f"Max Drawdown: ${mm_final['risk']['max_drawdown']:,.2f}")
        print(f"Current Drawdown: ${mm_final['risk']['current_drawdown']:,.2f}")
        
        print(f"\nModel Performance:")
        print(f"GARCH Fits: {mm_final['volatility']['fit_count']}")
        print(f"Volatility Regime: {mm_final['volatility']['volatility_regime']}")
        print(f"Market Toxicity: {mm_final['adverse_selection']['market_toxicity']:.2f}")
        print(f"Toxic Traders: {mm_final['adverse_selection']['num_toxic_traders']}")
        
        print(f"\nMarket Statistics:")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Avg Spread: {np.mean(results['spread_history']):.3f}")
        print(f"Price Volatility: {np.std(results['price_history']):.3f}")