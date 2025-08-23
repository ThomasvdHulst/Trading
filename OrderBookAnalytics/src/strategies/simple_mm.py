""" This file contains a simple market maker strategy using order book analytics. """

# Import used libraries
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from ..core.order_book import OrderBook
from ..analytics.microstructure import MicrostructureAnalyzer
from ..analytics.toxicity import ToxicityAnalyzer


@dataclass
class Position:
    """ Track strategy position. """
    quantity: float = 0
    avg_cost: float = 0
    realized_pnl: float = 0
    unrealized_pnl: float = 0


@dataclass
class MMOrder:
    """ Market maker order. """
    side: str
    price: float
    quantity: float
    order_id: int


class SimpleMarketMaker:
    """ Simple market maker strategy using order book analytics. Place
    quotes on both sides and manages inventory risk. """

    def __init__(self, 
                 max_position: float = 1000,
                 min_spread: float = 0.02,
                 target_spread: float = 0.05,
                 max_spread: float = 0.20,
                 quote_size: float = 100,
                 toxicity_threshold: float = 0.7):
        
        # Strategy parameters
        self.max_position = max_position
        self.min_spread = min_spread
        self.target_spread = target_spread
        self.max_spread = max_spread
        self.quote_size = quote_size
        self.toxicity_threshold = toxicity_threshold

        # Position tracking
        self.position = Position()
        self.active_orders: Dict[int, MMOrder] = {}
        self.order_id_counter = 10000

        # Analytics
        self.microstructure = MicrostructureAnalyzer()
        self.toxicity = ToxicityAnalyzer()

        # Performance tracking
        self.trades = []
        self.pnl_history = []


    def calculate_quotes(self, order_book: OrderBook) -> Tuple[Optional[MMOrder], Optional[MMOrder]]:
        """ Calculate quotes for the market maker.
        
        Args:
            order_book: OrderBook object to calculate quotes from.

        Returns:
            Tuple containing the buy and sell quotes.
        """
        
        # Get current metrics
        metrics = self.microstructure.calculate_metrics(order_book)
        toxicity = self.toxicity.get_toxicity_metrics()

        # Get mid price
        mid = metrics.get('mid_price')
        if not mid:
            return None, None
        
        # Base spread calculation
        spread = self.target_spread

        # Adjust spread based on toxicity
        vpin = toxicity.get('vpin', 0)
        if vpin and vpin > self.toxicity_threshold:
            # Widen spread when toxicity is high
            spread = min(self.max_spread, spread * (1 + vpin))
        
        # Adjust for inventory risk
        inventory_ratio = abs(self.position.quantity) / self.max_position
        if inventory_ratio > 0.5:
            # Widen spread when inventory is high
            spread = min(self.max_spread, spread * (1 + inventory_ratio))

        # Skew quotes based on inventory
        inventory_skew = self.position.quantity / self.max_position
        bid_adjustment = max(0, inventory_skew * spread / 2) # Lower bid if long
        ask_adjustment = max(0, -inventory_skew * spread / 2) # Higher ask if short

        # Calculate quote prices
        half_spread = spread / 2
        bid_price = mid - half_spread - bid_adjustment
        ask_price = mid + half_spread + ask_adjustment

        # Adjust for book imbalance
        imbalance = metrics.get('order_book_imbalance', 0)
        if abs(imbalance) > 0.3:
            # Adjust prices based on order book pressure
            price_adjustment = imbalance * self.min_spread
            bid_price -= price_adjustment
            ask_price -= price_adjustment

        # Size adjustment based on conditions
        bid_size = self.quote_size
        ask_size = self.quote_size

        # Reduce size if position limits approached
        if self.position.quantity > self.max_position * 0.8:
            bid_size *= 0.5 # Reduce buying
        elif self.position.quantity < -self.max_position * 0.8:
            ask_size *= 0.5 # Reduce selling

        # Create orders
        bid_order = None
        ask_order = None

        if abs(self.position.quantity + bid_size) <= self.max_position:
            bid_order = MMOrder(
                side = 'BID',
                price = round(bid_price, 2),
                quantity = bid_size,
                order_id = self._get_order_id()
            )
        
        if abs(self.position.quantity - ask_size) <= self.max_position:
            ask_order = MMOrder(
                side = 'ASK',
                price = round(ask_price, 2),
                quantity = ask_size,
                order_id = self._get_order_id()
            )

        return bid_order, ask_order
    
    
    def on_trade(self, price: float, quantity: float, is_buy: bool) -> None:
        """ Process a trade. 
        
        Args:
            price: Trade price.
            quantity: Trade quantity.
            is_buy: Whether the trade is a buy.
        """

        if is_buy:
            # We sold (were hit on our ask)
            trade_quantity = -quantity
        else:
            # We bought (were hit on our bid)
            trade_quantity = quantity

        # Update position
        old_quantity = self.position.quantity
        new_quantity = old_quantity + trade_quantity

        # Calculate average cost
        if old_quantity == 0:
            self.position.avg_cost = price
        elif np.sign(old_quantity) != np.sign(new_quantity):
            # Position flipped, realize PnL
            if old_quantity > 0:
                # Was long, calculate PnL on closed portion
                self.position.realized_pnl += old_quantity * (price - self.position.avg_cost)
            else:
                # Was short
                self.position.realized_pnl += -old_quantity * (self.position.avg_cost - price)

            # Reset average cost for new position
            self.position.avg_cost = price
        else:
            # Position in same direction, update weighted average
            total_cost = old_quantity * self.position.avg_cost + trade_quantity * price
            self.position.avg_cost = total_cost / new_quantity

        self.position.quantity = new_quantity

        # Record trade
        self.trades.append({
            'timestamp': pd.Timestamp.now(),
            'price': price,
            'quantity': trade_quantity,
            'position': self.position.quantity,
            'avg_cost': self.position.avg_cost,
        })

        # Feed to toxicity analyzer
        self.toxicity.process_trade(price, quantity, is_buy, price)


    def update_unrealized_pnl(self, current_price: float) -> None:
        """ Update unrealized PnL based on current price. 
        
        Args:
            current_price: Current market price.
        """

        if self.position.quantity != 0:
            if self.position.quantity > 0:
                # Long position
                self.position.unrealized_pnl = self.position.quantity * (
                    current_price - self.position.avg_cost
                )
            else:
                # Short position
                self.position.unrealized_pnl = -self.position.quantity * (
                    self.position.avg_cost - current_price
                )
        else:
            self.position.unrealized_pnl = 0

        # Record PnL
        self.pnl_history.append({
            'timestamp': pd.Timestamp.now(),
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl,
            'total_pnl': self.position.realized_pnl + self.position.unrealized_pnl,
            'position': self.position.quantity
        })


    def get_performance_metrics(self) -> Dict:
        """ Get performance metrics. 
        
        Returns:
            Dictionary containing performance metrics.
        """

        if not self.pnl_history:
            return {}
        
        pnl_df = pd.DataFrame(self.pnl_history)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        metrics = {
            'total_pnl': self.position.realized_pnl + self.position.unrealized_pnl,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl,
            'current_position': self.position.quantity,
            'num_trades': len(self.trades),
        }

        if not trades_df.empty:
            metrics['avg_trade_size'] = trades_df['quantity'].abs().mean()
            metrics['max_position'] = trades_df['position'].abs().max()

        if len(pnl_df) > 1:
            # Calculate Sharpe ration (simplified)
            returns = pnl_df['total_pnl'].diff()
            if returns.std() > 0:
                metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()

            # Max drawdown
            cumulative = pnl_df['total_pnl']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max.abs().clip(lower=1)
            metrics['max_drawdown'] = drawdown.min()

        return metrics
    

    def _get_order_id(self) -> int:
        """ Get a new order ID. """
        order_id = self.order_id_counter
        self.order_id_counter += 1
        return order_id