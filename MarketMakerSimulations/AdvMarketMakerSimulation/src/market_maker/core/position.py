""" This file contains the position tracking implementation """

# Import used libraries
from typing import Tuple, Dict, List

from ..core.base import IPositionTracker, Position, Trade # Import the IPositionTracker, Position, and Trade interfaces

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import MarketMakerConfig, FeesConfig # Import the MarketMakerConfig and FeesConfig classes
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import MarketMakerConfig, FeesConfig


class PositionTracker(IPositionTracker):
    """ Track position, PnL and trading statistics. """

    def __init__(self, config: MarketMakerConfig, fees_config: FeesConfig):
        """ Initialize position tracker.
        
        Args:
            config: Market maker configuration
            fees_config: Fees configuration
        """
        self.config = config
        self.fees_config = fees_config

        # Initialize position tracking
        self.position: Position = Position(
            quantity = 0,
            average_price = 0.0,
            realized_pnl = 0.0,
            unrealized_pnl = 0.0,
            total_fees = 0.0,
            last_update = 0.0
        )

        # Cost basis tracking
        self.cost_basis = 0.0

        # Statistics
        self.total_volume = 0
        self.num_trades = 0
        self.max_position = 0
        self.min_position = 0

        # Trade history for analysis
        self.trade_history: List[Trade] = []


    def update_position(self, trade: Trade) -> None:
        """ Update position with executed trade.
        
        Args:
            trade: Executed trade
        """

        # Determine if we bought or sold
        we_bought = trade.buy_trader == self.config.trader_id
        we_sold = trade.sell_trader == self.config.trader_id

        if not (we_bought or we_sold):
            return # Not our trade
        
        # Calculate fees
        fee = trade.quantity * self.fees_config.exchange_fee
        self.position.total_fees += fee

        if we_bought:
            self._process_buy(trade, fee)
        else:
            self._process_sell(trade, fee)

        # Update statistics
        self.total_volume += trade.quantity
        self.num_trades += 1
        self.max_position = max(self.max_position, self.position.quantity)
        self.min_position = min(self.min_position, self.position.quantity)
        self.position.last_update = trade.timestamp

        # Store trade
        self.trade_history.append(trade)


    def _process_buy(self, trade: Trade, fee: float) -> None:
        """ Process a buy trade """

        total_cost = trade.price * trade.quantity + fee

        if self.position.quantity >= 0:
            # Adding to a long position or starting a new long position
            new_quantity = self.position.quantity + trade.quantity
            new_cost_basis = self.cost_basis + total_cost

            # Update average price
            if new_quantity > 0:
                self.position.average_price = new_cost_basis / new_quantity

            self.position.quantity = new_quantity
            self.cost_basis = new_cost_basis

        else:
            # Closing a short position
            if trade.quantity >= abs(self.position.quantity):
                # Fully closing short and potentially going long
                shares_to_close = abs(self.position.quantity)
                shares_going_long = trade.quantity - shares_to_close

                # Realized PnL on closed short position
                avg_short_price = self.cost_basis / abs(self.position.quantity) if self.position.quantity != 0 else 0.0
                
                # Calculate the short PnL, which consists of:
                # 1. The difference between the average short price and the closing price, times the number of shares closed ("wins" per share)
                # 2. The fees paid on the shares closed, calculated by taking the share of the total fees paid by the number of shares closed
                short_pnl = (avg_short_price - trade.price) * shares_to_close - fee * (shares_to_close / trade.quantity)
                self.position.realized_pnl += short_pnl

                # Reset for new long position
                self.position.quantity = shares_going_long
                if shares_going_long > 0:
                    long_cost = trade.price * shares_going_long + fee * (shares_going_long / trade.quantity)
                    self.cost_basis = long_cost
                    self.position.average_price = trade.price
                else:
                    self.cost_basis = 0.0
                    self.position.average_price = 0.0

            else:
                # Partially closing short
                cost_per_share = self.cost_basis / abs(self.position.quantity)
                short_pnl = (cost_per_share - trade.price) * trade.quantity - fee
                self.position.realized_pnl += short_pnl

                self.position.quantity += trade.quantity
                self.cost_basis -= cost_per_share * trade.quantity


    def _process_sell(self, trade: Trade, fee: float) -> None:
        """ Process a sell trade (similar to buy). """

        if self.position.quantity > 0:
            # Closing a long position
            if trade.quantity >= self.position.quantity:
                # Fully closing long and potentially going short
                shares_to_close = self.position.quantity
                shares_going_short = trade.quantity - shares_to_close

                # Realized PnL on closed long position
                avg_long_price = self.cost_basis / self.position.quantity if self.position.quantity != 0 else 0.0
                long_pnl = (trade.price - avg_long_price) * shares_to_close - fee * (shares_to_close / trade.quantity)
                self.position.realized_pnl += long_pnl

                # Reset for new short position
                self.position.quantity = -shares_going_short
                if shares_going_short > 0:
                    short_value = trade.price * shares_going_short + fee * (shares_going_short / trade.quantity)
                    self.cost_basis = short_value
                    self.position.average_price = trade.price
                else:
                    self.cost_basis = 0.0
                    self.position.average_price = 0.0

            else:
                # Partially closing long
                cost_per_share = self.cost_basis / self.position.quantity
                long_pnl = (trade.price - cost_per_share) * trade.quantity - fee
                self.position.realized_pnl += long_pnl

                self.position.quantity -= trade.quantity
                self.cost_basis -= cost_per_share * trade.quantity

        else:
            # Adding to a short position or starting a new short position
            new_quantity = self.position.quantity - trade.quantity
            new_cost_basis = self.cost_basis + trade.price * trade.quantity + fee

            # Update average price
            if new_quantity != 0:
                self.position.average_price = new_cost_basis / abs(new_quantity)

            self.position.quantity = new_quantity
            self.cost_basis = new_cost_basis

                
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """ Calculate current PnL.
        
        Args:
            current_price: Current market price

        Returns:
            Tuple of (realized PnL, unrealized PnL)
        """

        # Realized PnL is already tracked
        realized_pnl = self.position.realized_pnl

        # Calculate unrealized PnL
        if self.position.quantity != 0:
            # Account for exit fees in unrealized PnL
            exit_fee = abs(self.position.quantity) * self.fees_config.exchange_fee

            if self.position.quantity > 0:
                # Long position
                avg_price = self.cost_basis / self.position.quantity
                unrealized_pnl = (current_price - avg_price) * self.position.quantity - exit_fee
            else:
                # Short position
                avg_price = self.cost_basis / abs(self.position.quantity)
                unrealized_pnl = (avg_price - current_price) * abs(self.position.quantity) - exit_fee
        else:
            unrealized_pnl = 0.0

        self.position.unrealized_pnl = unrealized_pnl

        return realized_pnl, unrealized_pnl
    

    def get_position(self) -> Position:
        """ Get current position. """
        return self.position
    

    def reset(self) -> None:
        """ Reset position to flat. """
        self.position = Position(
            quantity = 0,
            average_price = 0.0,
            realized_pnl = 0.0,
            unrealized_pnl = 0.0,
            total_fees = 0.0,
            last_update = 0.0
        )
        self.cost_basis = 0.0
        self.total_volume = 0
        self.num_trades = 0
        self.max_position = 0
        self.min_position = 0
        self.trade_history.clear()


    def get_metrics(self) -> Dict:
        """ Get position tracking metrics. 
        
        Returns:
            Dictionary of metrics
        """

        return {
            'position': self.position.quantity,
            'average_price': self.position.average_price,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl,
            'total_pnl': self.position.total_pnl,
            'total_fees': self.position.total_fees,
            'total_volume': self.total_volume,
            'num_trades': self.num_trades,
            'max_position': self.max_position,
            'min_position': self.min_position,
        }