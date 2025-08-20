""" This file contains the Microprice model """

# Import used libraries
import numpy as np
from typing import Dict, List, Tuple

# Suppress arch warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from ..core.base import IPricer, MarketState # Import the IPricer and MarketState interfaces

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import MicropriceConfig # Import the MicropriceConfig class
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import MicropriceConfig


class MicropriceModel(IPricer):
    """ Microprice model for better fair value estimation.
    
    The microprice is a weighted average of bid and ask prices where
    weights are based on their relative sizes. This provides a better
    estimate of the fair value than simple mid-price, especially when
    the order-book is imbalanced.
    """

    def __init__(self, config: MicropriceConfig):
        """ Initialize the microprice model.
        
        Args:
            config: Microprice configuration
        """
        self.config = config

    
    def calculate_fair_price(self, market_state: MarketState) -> float:
        """ Calculate fair price using microprice.
        
        Args:
            market_state: Current market state

        Returns:
            Fair price estimate
        """

        if market_state.best_bid and market_state.best_ask:
            # Simple microprice: weight by opposite side volume
            # Logic behind this is that the microprice should be closer to the side with more volume
            # This is because the side with more volume is more likely to be the side that will be executed
            # and the side with less volume is more likely to be the side that will be executed against 
            total_size = market_state.bid_volume + market_state.ask_volume
            if total_size > 0:
                microprice = (
                    market_state.best_bid * market_state.ask_volume +
                    market_state.best_ask * market_state.bid_volume
                ) / total_size

                return microprice
        
        return market_state.mid_price # Fallback to mid-price if no quotes available
    

    def calculate_microprice(self, order_book_snapshot: Dict) -> float:
        """ Calculate microprice from full order book. We use multiple levels with
        exponential decay to incorporate more information while giving more weight to
        top levels. 
        
        Args:
            order_book_snapshot: Order book snapshot with multiple levels

        Returns:
            Microprice estimate
        """

        bids = order_book_snapshot.get('bids', [])
        asks = order_book_snapshot.get('asks', [])

        if not bids or not asks:
            return order_book_snapshot.get('mid_price', 100.0)
        
        if self.config.use_multi_level:
            return self._calculate_multi_level_microprice(bids, asks)
        else:
            # Simple top-level microprice
            bid_price, bid_size = bids[0]
            ask_price, ask_size = asks[0]
            total_size = bid_size + ask_size
            if total_size > 0:
                return (bid_price * ask_size + ask_price * bid_size) / total_size
            
            return order_book_snapshot.get('mid_price', 100.0)
        
    
    def _calculate_multi_level_microprice(self, bids: List[Tuple[float, int]], asks: List[Tuple[float, int]]) -> float:
        """ Calculate microprice from multiple levels. 
        
        Args:
            bids: List of (price, size) tuples for bid side
            asks: List of (price, size) tuples for ask side

        Returns:
            Multi-level microprice
        """

        weighted_price = 0
        total_weight = 0

        # Process bid levels
        for i, (price, size) in enumerate(bids[:self.config.depth_levels]):
            # Exponential decay weight for deeper levels
            level_weight = size * np.exp(-i * self.config.depth_decay)
            weighted_price += price * level_weight
            total_weight += level_weight

        # Process ask levels
        for i, (price, size) in enumerate(asks[:self.config.depth_levels]):
            # Exponential decay weight for deeper levels
            level_weight = size * np.exp(-i * self.config.depth_decay)
            weighted_price += price * level_weight
            total_weight += level_weight

        if total_weight > 0:
            return weighted_price / total_weight
        
        # Fallback to simple mid if no valid levels
        return (bids[0][0] + asks[0][0]) / 2 if bids and asks else 100.0