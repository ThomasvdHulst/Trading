""" This file contains the spread calculation implementation """


# Import used libraries
from typing import Dict

from ..core.base import ISpreadCalculator, Signal, Position

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import StrategyConfig, MarketMakerConfig # Import the config classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import StrategyConfig, MarketMakerConfig


class SpreadCalculator(ISpreadCalculator):
    """ Calculate the optimal spread based on multiple factors. 
    Combine multiple signals with configurable weights to allow for
    easy tuning and experimentation.
    """

    def __init__(self, strategy_config: StrategyConfig, market_maker_config: MarketMakerConfig):
        """ Initialize spread calculator.
        
        Args:
            strategy_config: Strategy configuration
            market_maker_config: Market maker configuration
        """
        self.strategy_config = strategy_config
        self.market_maker_config = market_maker_config
        self.components = strategy_config.spread_components

        # Track spread history for analysis
        self.spread_history = []
        self.last_components = {}


    def calculate_spread(self, signals: Dict[str, Signal], position: Position) -> float:
        """ Calculate optimal spread given current signals. 
        Spread needs to balance:
        1. Competitiveness (tighter = more trades)
        2. Profitability (wider = more profit per trade)
        3. Risk (adverse selection, inventory risk)

        Args:
            signals: Dictionary of trading signals
            position: Current position

        Returns:
            Calculated spread
        """

        # Start with base spread
        base_spread = signals.get('base_spread', Signal('base', 0.20, 1.0, 0)).value

        # Calculate component multipliers
        components = {
            'base': 1.0,
            'volatility': self._volatility_component(signals),
            'inventory': self._inventory_component(position),
            'adverse': self._adverse_selection_component(signals),
            'imbalance': self._book_imbalance_component(signals)
        }

        # Weighted average of components
        weights = {
            'base': self.components.base_weight,
            'volatility': self.components.volatility_weight,
            'inventory': self.components.inventory_weight,
            'adverse': self.components.adverse_weight,
            'imbalance': self.components.imbalance_weight
        }

        # Calculate weighted multiplier
        total_multiplier = sum(
            components[key] * weights[key]
            for key in components
        )

        # Apply to base spread
        final_spread = base_spread * total_multiplier

        # Store for analysis
        self.last_components = components
        self.spread_history.append({
            'spread': final_spread,
            'components': components.copy(),
            'multiplier': total_multiplier
        })
        
        return final_spread
    

    def _volatility_component(self, signals: Dict[str, Signal]) -> float:
        """ Calculate volatility component of spread.
        Higher volatility = wider spread (more risk).
        
        Args:
            signals: Dictionary of trading signals

        Returns:
            Volatility component
        """

        vol_signal = signals.get('volatility', Signal('volatility', 0.0002, 0.5, 0))

        # Normalize volatility to multiplier
        # Assuming baseline volatility of 0.0002
        baseline_vol = 0.0002
        vol_ratio = vol_signal.value / baseline_vol

        # Convert to multiplier (1.0 = no change)
        # Cap between 0.5 and 3.0
        multiplier = max(0.5, min(3.0, vol_ratio))

        return 1 # TO DO!!!!! Volatility component does not work....

        return multiplier
        

    def _inventory_component(self, position: Position) -> float:
        """ Calculate inventory component of spread.
        Higher inventory = wider spread (more risk).

        Args:
            position: Current position

        Returns:
            Inventory component
        """

        if position.quantity == 0:
            return 1.0
        
        # Calculate inventory pressure (0 to 1)
        max_inv = self.market_maker_config.max_inventory
        inventory_ratio = abs(position.quantity) / max_inv

        # Higher inventory = wider spread
        # But not too aggressive to avoid getting stuck
        multiplier = 1.0 + inventory_ratio * 0.5

        return multiplier
    

    def _adverse_selection_component(self, signals: Dict[str, Signal]) -> float:
        """ Calculate adverse selection component of spread.
        Higher toxicity = wider spread.
        
        Args:
            signals: Dictionary of trading signals

        Returns:
            Adverse selection component
        """

        # Get toxicity signal
        toxicity_signal = signals.get('toxicity', Signal('toxicity', 0.5, 0.5, 0))

        # Convert toxicity to multiplier
        # 0.5 = normal, >0.6 = toxic
        if toxicity_signal.value > 0.6:
            multiplier = 1.0 + (toxicity_signal.value - 0.5) * 2
        else:
            multiplier = 1.0

        return multiplier
    

    def _book_imbalance_component(self, signals: Dict[str, Signal]) -> float:
        """ Calculate book imbalance component of spread.
        Higher imbalance = adjust spread to lean against it.
        
        Args:
            signals: Dictionary of trading signals

        Returns:
            Book imbalance component
        """

        imbalance_signal = signals.get('book_imbalance', Signal('imbalance', 0.0, 0.5, 0))
        
        # Small imbalances do not matter much
        if abs(imbalance_signal.value) < 0.2:
            return 1.0
        
        # Larger imbalance = slightly wider spread
        multiplier = 1.0 + abs(imbalance_signal.value) * 0.6

        return multiplier
    

    def get_last_components(self) -> Dict[str, float]:
        """ Get last calculated components.
        
        Returns:
            Dictionary of last calculated components
        """
        return self.last_components