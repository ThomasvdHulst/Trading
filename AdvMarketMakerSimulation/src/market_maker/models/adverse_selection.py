""" This file contains the Adverse Selection model """

# Import used libraries
from typing import Dict, List, Deque

# Defaultdict is a dictionary that will initialize with a default value if the key is not found
from collections import deque, defaultdict
from dataclasses import dataclass

from ..core.base import IAdverseSelectionDetector, Trade # Import the IAdverseSelectionDetector and Trade interfaces

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import AdverseSelectionConfig # Import the AdverseSelectionConfig class
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import AdverseSelectionConfig


@dataclass
class TradeOutcome:
    """ Track outcome of a trade for adverse selection detection """

    trade_id: int # Trade identifier
    trader_id: str # Trader identifier
    side: str # 'buy' or 'sell' from MM perspective
    price: float # Trade price
    size: int # Trade size
    price_after: float # Price after trade execution
    price_move: float # Price move from trade execution
    adverse: bool # Was this trade adverse for MM?
    timestamp: float # Timestamp of trade execution


class AdverseSelectionDetector(IAdverseSelectionDetector):
    """ Detect adverse selection (toxic flow) from trading patterns. 
    
    Market makers lose money to informed traders who have better information.
    By tracking which traders consistently trade in the right direction, we can
    identify toxic flows and adjust our strategy accordingly.
    """

    def __init__(self, config: AdverseSelectionConfig):
       """ Initiallize adverse selection detector.
       
       Args:
            config: Adverse selection configuration
       """

       self.config = config

       # Track trade outcomes by trader
       # This is a dictionary with the trader ID as the key and the value is a deque of TradeOutcome objects
       self.trader_history: Dict[str, Deque[TradeOutcome]] = defaultdict(
           lambda: deque(maxlen=config.lookback_trades)
       )

       # Toxicity scores (0 = benign, 1 = toxic)
       self.toxicity_scores: Dict[str, float] = {}

       # Overall market toxicity
       self.market_toxicity = 0.5 # Start neutral

       # Track our trades for analysis
       self.our_trades: Deque[Trade] = deque(maxlen=100)
       #self.pending_analysis: List[Tuple[Trade, float]] = [] # (trade, entry_tick)

    
    def update(self, trade: Trade, price_after: float) -> None:
        """ Update model with trade outcome.
        
        Args:
            trade: Executed trade
            price_after: Price after trade execution (some ticks later)
        """

        # Determine if we were buyer or seller
        our_side = None
        counterparty = None

        if trade.buy_trader == "MM_01":
            our_side = "buy"
            counterparty = trade.sell_trader
        elif trade.sell_trader == "MM_01":
            our_side = "sell"
            counterparty = trade.buy_trader
        else:
            return # Not our trade
        
        # Calculate price move
        price_move = (price_after - trade.price) / trade.price

        # Determine if trade was adverse for us
        # If we bought and price went down, or sold and price went up, it's adverse
        adverse = (our_side == "buy" and price_move < 0) or (our_side == "sell" and price_move > 0)

        # Create trade outcome record
        outcome = TradeOutcome(
            trade_id = trade.trade_id,
            trader_id = counterparty,
            side = our_side,
            price = trade.price,
            size = trade.quantity,
            price_after = price_after,
            price_move = price_move,
            adverse = adverse,
            timestamp = trade.timestamp
        )

        # Add to trader history
        self.trader_history[counterparty].append(outcome)

        # Update toxicity score for this trader
        self._update_toxicity_score(counterparty)

        # Update overall market toxicity
        self._update_market_toxicity()

    
    def _update_toxicity_score(self, trader_id: str) -> None:
        """ Update toxicity for a specific trader. Here we use
        exponential decay to weight recent trades more heavily. 
        
        Args:
            trader_id: Trader ID
        """

        # Get trader history
        history = self.trader_history[trader_id]

        if len(history) < self.config.min_trades_for_scoring:
            # Not enough data to score yet
            self.toxicity_scores[trader_id] = 0.5
            return
        
        # Calculate weighted adverse rate
        weights = []
        adverse_scores = []

        for i, outcome in enumerate(history):
            # Exponential decay weight (recent trades more heavily)
            weight = self.config.decay_factor ** (len(history) - i - 1)
            weights.append(weight)

            # Score based on both frequency and magnitude
            if outcome.adverse:
                # Adverse trade, score based on magnitude
                # Base penalty + magnitude bonus, capped at 1.0
                base_penalty = 0.5
                magnitude_bonus = abs(outcome.price_move) * 50  # Scale factor
                score = min(1.0, base_penalty + magnitude_bonus)
            else:
                # Beneficial trade, reduce score
                score = 0.0

            adverse_scores.append(score)

        # Weighted average
        if sum(weights) > 0:
            toxicity = sum(w * s for w, s in zip(weights, adverse_scores)) / sum(weights)
        else:
            toxicity = 0.5

        # Smooth update (momentum)
        old_score = self.toxicity_scores.get(trader_id, 0.5)
        self.toxicity_scores[trader_id] = 0.7 * toxicity + 0.3 * old_score


    def _update_market_toxicity(self) -> None:
        """ Update overall market toxicity based on recent activity. 
        It can be that the whole market becomes toxic, e.g. before major news. """

        # If we have no traders with identified toxicity scores, we can't update the market toxicity
        if not self.toxicity_scores:
            self.market_toxicity = 0.5
            return
        
        # Weight by sum of recent activity
        weighted_sum = 0
        weight_total = 0

        for trader_id, score in self.toxicity_scores.items():
            if trader_id in self.trader_history and self.trader_history[trader_id]:
                # Weight by number of recent trades
                recent_trades = len(self.trader_history[trader_id])
                weight = recent_trades
                weighted_sum += score * weight
                weight_total += weight

        if weight_total > 0:
            self.market_toxicity = weighted_sum / weight_total
        else:
            # No recent activity, stay neutral
            self.market_toxicity = 0.5


    def get_toxicity_score(self, trader_id: str) -> float:
        """ Get toxicity score for a specific trader.

        Args:
            trader_id: Trader identifier

        Returns:
            Toxicity score (0-1)
        """

        return self.toxicity_scores.get(trader_id, 0.5)
    

    def get_spread_adjustment(self) -> float:
        """ Get recommended spread adjustment based on toxicity.
        Widen spreads when facing toxic flow to compensate for adverse
        selection costs.
        
        Returns:
            Spread multiplier (1.0 = no adjustment)
        """

        if self.market_toxicity > self.config.toxicity_threshold:
            # High toxicity, widen spread significantly
            return 1.0 + (self.market_toxicity - 0.5) * 2.0
        elif self.market_toxicity > 0.55:
            # Moderate toxicity, widen spread moderately
            return 1.1
        else:
            # Low toxicity, normal spreads
            return 1.0
        

    def identify_toxic_traders(self) -> List[str]:
        """ Get list of traders identified as toxic.
        
        Returns:
            List of toxic trader IDs
        """

        toxic_traders = [
            trader_id for trader_id, score in self.toxicity_scores.items()
            if score > self.config.toxicity_threshold
        ]
        return toxic_traders
    
    
    def get_metrics(self) -> Dict:
        """ Get adverse selection metrics.
        
        Returns:
            Dictionary of metrics
        """

        toxic_traders = self.identify_toxic_traders()

        return {
            "market_toxicity": self.market_toxicity,
            "num_toxic_traders": len(toxic_traders),
            "toxic_traders": toxic_traders,
            "total_traders_tracked": len(self.toxicity_scores),
            "spread_adjustment": self.get_spread_adjustment(),
        }

