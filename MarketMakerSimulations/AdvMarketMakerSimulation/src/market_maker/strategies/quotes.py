""" This file contains the quote generation implementation """


# Import used libraries
from typing import List, Tuple

from ..core.base import Quote, QuotePair, Side, Position
from .spreads import SpreadCalculator

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import MarketMakerConfig, MarketConfig # Import the config classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import MarketMakerConfig, MarketConfig


class QuoteStrategy:
    """ Generate quote ladder (multiple levels of orders).
    We place orders at multiple price levels to:
    1. Capture different types of flow
    2. Reduce impact of adverse selection
    3. Provide consistent liquidity
    """

    def __init__(self, config: MarketMakerConfig, market_config: MarketConfig, spread_calculator: SpreadCalculator):
        """ Initialize quote strategy.
        
        Args:
            config: Market maker configuration
            market_config: Market configuration
            spread_calculator: Spread calculator instance (from spreads.py)
        """

        self.config = config
        self.market_config = market_config
        self.spread_calculator = spread_calculator

        # Quote generation parameters
        self.num_levels = config.num_quote_levels
        self.level_spacing = config.level_spacing_ticks * market_config.tick_size

        # Defensive mode state
        self.defensive_mode = False
        self.defensive_mode_until = 0


    def generate_quotes(self, fair_value: float, spread: float, position: Position, tick: int) -> List[QuotePair]:
        """ Generate multi-level quotes.
        
        Args:
            fair_value: Fair value of the asset
            spread: Current spread
            position: Current position
            tick: Current tick

        Returns:
            List of quote pairs (bid/ask at each level)
        """

        quotes = []

        # Check if we're in defensive mode
        if tick < self.defensive_mode_until:
            self.defensive_mode = True
        else:
            self.defensive_mode = False

        # Calculate base quotes (best level)
        half_spread = spread / 2
        base_bid_price = fair_value - half_spread
        base_ask_price = fair_value + half_spread

        # Apply inventory skew
        skew = self._calculate_inventory_skew(position, fair_value)
        base_bid_price -= skew
        base_ask_price -= skew

        # Generate quotes for each level
        for level in range(self.num_levels):
            bid_price = base_bid_price - (level * self.level_spacing)
            ask_price = base_ask_price + (level * self.level_spacing)

            # Calculate sizes for this level
            bid_size, ask_size = self._calculate_level_sizes(level, position, self.defensive_mode)

            # Round prices to tick size
            bid_price = round(bid_price / self.market_config.tick_size) * self.market_config.tick_size
            ask_price = round(ask_price / self.market_config.tick_size) * self.market_config.tick_size

            # Create quotes
            bid_quote = None
            ask_quote = None

            if bid_size > 0:
                bid_quote = Quote(
                    side = Side.BUY,
                    price = bid_price,
                    size = bid_size,
                    level = level
                )

            if ask_size > 0:
                ask_quote = Quote(
                    side = Side.SELL,
                    price = ask_price,
                    size = ask_size,
                    level = level
                )

            # Add to quote pair
            quotes.append(QuotePair(bid=bid_quote, ask=ask_quote))

        return quotes
    

    def _calculate_inventory_skew(self, position: Position, fair_value: float) -> float:
        """ Calculate price skew based on inventory. Skew prices to reduce inventory risk
        - Long position: lower prices (encourage selling)
        - Short position: higher prices (encourage buying)
        
        Args:
            position: Current position
            fair_value: Fair value of the asset

        Returns:
            Price skew
        """

        if position.quantity == 0:
            return 0.0
        
        # Skew proportional to inventory
        skew = position.quantity * self.config.inventory_skew_factor

        return skew
    

    def _calculate_level_sizes(self, level: int, position: Position, defensive_mode: bool) -> Tuple[int, int]:
        """ Calculate order sizes for each level.
        
        Args:
            level: Current level index (0 = best level, 1 = next level, etc.)
            position: Current position
            defensive_mode: Whether we're in defensive mode

        Returns:
            Tuple of bid and ask sizes
        """

        # Base size is quote_size
        base_size = self.config.quote_size

        # Size reduction per level
        level_multiplier = (0.8 ** level)

        # Defensive mode: smaller at better prices, larger at worse prices
        if defensive_mode:
            if level == 0:
                level_multiplier = 0.3
            elif level == 1:
                level_multiplier = 0.6
            else:
                level_multiplier = 1.0

        # Inventory-based sizing
        inventory_ratio = abs(position.quantity) / self.config.max_inventory

        if position.is_long:
            # Reduce bid size, increase ask size
            bid_multiplier = max(0.2, 1.0 - inventory_ratio * 0.8)
            ask_multiplier = min(1.5, 1.0 + inventory_ratio * 0.5)
        elif position.is_short:
            # Increase bid size, reduce ask size
            bid_multiplier = min(1.5, 1.0 + inventory_ratio * 0.5)
            ask_multiplier = max(0.2, 1.0 - inventory_ratio * 0.8)
        else:
            bid_multiplier = 1.0
            ask_multiplier = 1.0

        # Calculate final sizes
        bid_size = int(base_size * level_multiplier * bid_multiplier)
        ask_size = int(base_size * level_multiplier * ask_multiplier)
 
        # Round to lot size
        lot_size = self.market_config.lot_size
        bid_size = bid_size // lot_size * lot_size
        ask_size = ask_size // lot_size * lot_size

        # Respect position limits
        max_long = self.config.max_inventory - position.quantity
        max_short = self.config.max_inventory + position.quantity

        bid_size = min(bid_size, max(0, max_long))
        ask_size = min(ask_size, max(0, max_short))

        return bid_size, ask_size
    

    def set_defensive_mode(self, ticks: int, current_tick: int) -> None:
        """ Enable defensive mode for specified number of ticks.
        
        Args:
            ticks: Number of ticks to enable/stay defensive mode
            current_tick: Current tick
        """

        self.defensive_mode = True
        self.defensive_mode_until = current_tick + ticks