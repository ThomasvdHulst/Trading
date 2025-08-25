""" This file contains the risk management implementation """


# Import used libraries
from typing import Dict

from ..core.base import IRiskManager, Position, MarketState

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import RiskConfig, MarketMakerConfig # Import the config classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import RiskConfig, MarketMakerConfig


class RiskManager(IRiskManager):
    """ Manage risk exposure for the market maker. """

    def __init__(self, config: RiskConfig, market_maker_config: MarketMakerConfig):
        """ Initialize risk manager.

        Args:
            config: Risk configuration
            market_maker_config: Market maker configuration
        """

        self.config = config
        self.market_maker_config = market_maker_config

        # Risk state
        self.max_drawdown = 0.0
        self.high_water_mark = 0.0
        self.risk_events = []
        

    def check_limits(self, position: Position, market_state: MarketState) -> bool:
        """ Check if within risk limits.
        
        Args:
            position: Current position
            market_state: Current market state

        Returns:
            True if within limits, False otherwise
        """

        # Check stop loss
        if position.total_pnl < self.config.stop_loss_threshold:
            self._log_risk_event("STOP LOSS", position.total_pnl, market_state.tick)
            return False
        
        # Check position value limit
        position_value = abs(position.quantity * market_state.mid_price)
        if position_value > self.config.max_position_value:
            self._log_risk_event("POSITION LIMIT", position_value, market_state.tick)
            return False
        
        # Check inventory limit:
        inventory_ratio = abs(position.quantity) / self.market_maker_config.max_inventory
        if inventory_ratio > self.config.inventory_pct_limit:
            self._log_risk_event("INVENTORY WARNING", inventory_ratio, market_state.tick)
            # Continue trading but with reduced position sizing

        return True
    

    def get_risk_metrics(self, position: Position) -> Dict[str, float]:
        """ Calculate current risk metrics.
        
        Args:
            position: Current position

        Returns:
            Dictionary containing risk metrics
        """

        # Update drawdown
        if position.total_pnl > self.high_water_mark:
            self.high_water_mark = position.total_pnl

        current_drawdown = self.high_water_mark - position.total_pnl
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        return {
            'total_pnl': position.total_pnl,
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown,
            'position_value': abs(position.quantity * position.average_price),
            'inventory_ratio': abs(position.quantity) / self.market_maker_config.max_inventory,
            'distance_to_stop': position.total_pnl - self.config.stop_loss_threshold,
        }
    

    def should_stop_trading(self, position: Position) -> bool:
        """ Check if we should stop trading due to risk.
        
        Args:
            position: Current position

        Returns:
            True if we should stop trading, False otherwise
        """

        return position.total_pnl < self.config.stop_loss_threshold
    

    def _log_risk_event(self, event_type: str, value: float, tick: int) -> None:
        """ Log a risk event.
        
        Args:
            event_type: Type of event
            value: Value of the event
            tick: Tick at which the event occurred
        """

        self.risk_events.append({
            'type': event_type,
            'value': value,
            'tick': tick,
        })