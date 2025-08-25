""" This file contains the core market maker implementation """

# Import used libraries
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass

from ..core.base import IMarketMaker, IOrderBook, MarketState, Trade, Signal, Position, QuotePair
from ..core.position import PositionTracker
from ..execution.orders import OrderManager
from ..execution.risk import RiskManager
from ..models.garch import GARCHVolatilityModel
from ..models.microprice import MicropriceModel
from ..models.adverse_selection import AdverseSelectionDetector
from ..strategies.quotes import QuoteStrategy
from ..strategies.spreads import SpreadCalculator

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import Config # Import the config classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import Config


@dataclass
class PendingTradeAnalysis:
    """ Track trades pending adverse selection analysis """
    trade: Trade
    entry_price: float
    entry_tick: int
    analyze_after_tick: int # When to measure price impact


class MarketMaker(IMarketMaker):
    """ Main market maker implementation. This class
    orchestrates all components but delegates specific
    responsibilities to specialized classes. """

    def __init__(self, config: Config):
        """ Initialize market maker with all components. 
        
        Args:
            config: Full configuration object.
        """

        self.config = config
        self.trader_id = config.market_maker.trader_id

        # Initialize components
        print(f"Initializing market maker: {self.trader_id}")

        # Core components
        self.position_tracker = PositionTracker(config.market_maker, config.fees)
        self.order_manager = OrderManager(config.market_maker)
        self.risk_manager = RiskManager(config.risk, config.market_maker)

        # Models
        self.volatility_model = GARCHVolatilityModel(config.models.garch)
        self.microprice_model = MicropriceModel(config.models.microprice)
        self.adverse_selection_detector = AdverseSelectionDetector(config.models.adverse_selection)

        # Adverse selection analysis
        self.pending_trade_analysis = deque()
        self.price_impact_window = 10 # Ticks to wait before measuring price impact
        self.price_history = deque(maxlen=100) # (tick, price)

        # Strategies
        self.spread_calculator = SpreadCalculator(config.strategy, config.market_maker)
        self.quote_strategy = QuoteStrategy(config.market_maker, config.market, self.spread_calculator)

        # State tracking
        self.current_tick = 0
        self.last_market_state: Optional[MarketState] = None
        self.signals: Dict[str, Signal] = {}

        # Performance tracking
        self.update_count = 0
        self.trades_count = 0

        print(f"Market maker initialized: {self.trader_id}")


    def update(self, order_book: IOrderBook, market_state: MarketState) -> None:
        """ Main update cycle - called every tick.
        Template method pattern:
        1. Gather information (signals)
        2. Make decisions (quotes)
        3. Take actions (orders)
        4. Update state

        Args:
            order_book: Current order book
            market_state: Current market state
        """

        self.current_tick = market_state.tick
        self.last_market_state = market_state
        self.update_count += 1

        # Track price history for adverse selection analysis
        self.price_history.append((self.current_tick, market_state.mid_price))

        # Step 1: Update models with new data
        self._update_models(market_state)

        # Step 2: Generate signals
        self.signals = self._generate_signals(order_book, market_state)

        # Step 3: Check risk limits
        position = self.position_tracker.get_position()
        #if not self.risk_manager.check_limits(position, market_state):
        #    # Risk limits exceeded - cancel all orders and stop
        #    self.order_manager.cancel_all_orders(order_book)
        #    print(f"Risk limits exceeded at tick {self.current_tick}. Cancelling all orders.")
        #    return
        
        # Step 4: Generate quotes
        quotes = self._generate_quotes(market_state, position)
        
        # Step 5: Place orders
        self.order_manager.update_quotes(quotes, order_book)

        # Step 6: Update position PnL
        self.position_tracker.calculate_pnl(market_state.mid_price)

        # Print progress periodically
        if self.update_count % 1000 == 0:
            self._print_status()


    def _update_models(self, market_state: MarketState) -> None:
        """ Update all models with new market data.
        
        Args:
            market_state: Current market state
        """

        # Update volatility model
        self.volatility_model.update(market_state.mid_price, market_state.tick)

        # Process pending adverse selection analyses
        self._process_pending_trade_analyses(market_state)


    def _process_pending_trade_analyses(self, market_state: MarketState) -> None:
        """ Process trades that are ready for adverse selection analysis. We wait
        N ticks after a trade to see if the price moved against us, indicating
        the counterparty has better information. 
        
        Args:
            market_state: Current market state
        """

        # Process all pending analyses that are ready
        while self.pending_trade_analysis:
            pending = self.pending_trade_analysis[0]

            # Check if enough time has passed
            if self.current_tick >= pending.analyze_after_tick:
                # Remove from queue
                self.pending_trade_analysis.popleft()

                # Get price after the delay
                price_after = self.last_market_state.mid_price

                # Update adverse selection detector with actual price movement
                self.adverse_selection_detector.update(pending.trade, price_after)
            
            else:
                # Not ready yet, stop processing (queue is ordered by tick)
                break

    
    def _generate_signals(self, order_book: IOrderBook, market_state: MarketState) -> Dict[str, Signal]:
        """ Generate trading signals from all models.
        
        Args:
            order_book: Current order book
            market_state: Current market state

        Returns:
            Dictionary of signals
        """

        signals = {}

        # Base spread signal
        signals['base_spread'] = Signal(
            name = 'base_spread',
            value = self.config.market_maker.base_spread,
            confidence = 1.0,
            timestamp = market_state.tick,
        )

        # Volatility signal
        current_vol = self.volatility_model.get_current_volatility()
        signals['volatility'] = Signal(
            name = 'volatility',
            value = float(current_vol),
            confidence = 0.8 if self.volatility_model.last_fit_success else 0.3,
            timestamp = market_state.tick
        )

        # Microprice signal (better fair value)
        book_snapshot = order_book.get_order_book_snapshot()
        microprice = self.microprice_model.calculate_microprice(book_snapshot)
        signals['microprice'] = Signal(
            name = 'microprice',
            value = float(microprice),
            confidence = 0.9,
            timestamp = market_state.tick
        )

        # Adverse selection signal
        market_toxicity = self.adverse_selection_detector.market_toxicity
        signals['toxicity'] = Signal(
            name = 'toxicity',
            value = market_toxicity,
            confidence = 0.7,
            timestamp = market_state.tick
        )

        # Book imbalance signal
        signals['book_imbalance'] = Signal(
            name = 'book_imbalance',
            value = market_state.bid_ask_imbalance,
            confidence = 0.9,
            timestamp = market_state.tick
        )

        return signals
    

    def _generate_quotes(self, market_state: MarketState, position: Position) -> List[QuotePair]:
        """ Generate quotes based on signals and position. 
        
        Args:
            market_state: Current market state
            position: Current position

        Returns:
            List of quote pairs
        """

        # Get fair value estimate (use microprice if available)
        fair_value = self.signals.get('microprice', Signal('mid', market_state.mid_price, 1.0, 0)).value

        # Calculate optimal spread
        spread = self.spread_calculator.calculate_spread(self.signals, position)

        # Generate quote ladder
        quotes = self.quote_strategy.generate_quotes(
            fair_value = fair_value,
            spread = spread,
            position = position,
            tick = self.current_tick
        )

        return quotes
    

    def on_trade(self, trade: Trade) -> None:
        """ Handle trade execution. 
        
        Args:
            trade: Executed trade
        """

        # Check if we're involved
        if trade.buy_trader != self.trader_id and trade.sell_trader != self.trader_id:
            return
                
        self.trades_count += 1

        # Update position
        self.position_tracker.update_position(trade)

        # Update order manager
        self.order_manager.on_trade(trade)

        # Schedule adverse selection analysis
        if self.last_market_state:
            # Schedule this trade for analysis after price_impact_window ticks
            pending = PendingTradeAnalysis(
                trade = trade,
                entry_price = trade.price, # Use trade price as entry price
                entry_tick = self.current_tick,
                analyze_after_tick = self.current_tick + self.price_impact_window
            )
            self.pending_trade_analysis.append(pending)


    def get_metrics(self) -> Dict[str, Any]:
        """ Get performance metrics.
        
        Returns:
            Dictionary of all metrics
        """

        position = self.position_tracker.get_position()

        metrics = {
            # Position metrics
            'position': self.position_tracker.get_metrics(),

            # Order metrics
            'orders': self.order_manager.get_metrics(),

            # Risk metrics
            'risk': self.risk_manager.get_risk_metrics(position),

            # Model metrics
            'volatility': self.volatility_model.get_metrics(),
            'adverse_selection': self.adverse_selection_detector.get_metrics(),

            # Strategy metrics
            'last_spread_components': self.spread_calculator.get_last_components(),

            # Overall metrics
            'update_count': self.update_count,
            'trades_count': self.trades_count,
            'current_tick': self.current_tick,
        }

        return metrics
    

    def reset(self) -> None:
        """ Reset strategy state """

        self.position_tracker.reset()
        self.update_count = 0
        self.trades_count = 0
        self.current_tick = 0
        self.signals.clear()

    
    def _print_status(self) -> None:
        """ Print current status. """

        if not self.last_market_state:
            return
        
        position = self.position_tracker.get_position()
        risk_metrics = self.risk_manager.get_risk_metrics(position)
        vol_metrics = self.volatility_model.get_metrics()
        adverse_metrics = self.adverse_selection_detector.get_metrics()

        print(f"\nTick {self.current_tick}:")
        print(f"Price: {self.last_market_state.mid_price:.4f}")
        print(f"Position: {position.quantity} @ ${position.average_price:.4f}")
        print(f"PnL: ${position.total_pnl:.2f} (Realized: ${position.realized_pnl:.2f}, Unrealized: ${position.unrealized_pnl:.2f})")
        print(f"Volatility: {vol_metrics['current_volatility']:.4f} ({vol_metrics['volatility_regime']})")
        print(f"Market toxicity: {adverse_metrics['market_toxicity']:.4f}")
        print(f"Toxic traders: {adverse_metrics['num_toxic_traders']}")
        print(f"Pending adverse analyses: {len(self.pending_trade_analysis)}")
        print(f"Orders: {self.order_manager.get_order_count()}")
        print(f"Trades: {self.trades_count}")
        print(f"Fees: ${self.position_tracker.get_position().total_fees:.2f}")

        # Show spread components
        components = self.spread_calculator.get_last_components()
        if components:
            print(f"Spread components: {components}")

        # Show risk metrics
        if risk_metrics['distance_to_stop'] < 5000:
            print(f"Warning: Close to stop loss! Distance: ${risk_metrics['distance_to_stop']:.2f}")
        
        