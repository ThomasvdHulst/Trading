""" Base classes for market maker simulation """

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """ Order side enumeration """
    BUY = 1
    SELL = -1


class OrderType(Enum):
    """ Order type enumeration """
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass
class Order:
    """ Order data structure """
    order_id: int
    trader_id: str
    side: Side
    price: float
    quantity: int
    timestamp: float
    order_type: OrderType = OrderType.LIMIT


@dataclass
class Trade:
    """ Trade data structure """
    trade_id: int
    price: float
    quantity: int
    buy_order_id: int
    sell_order_id: int
    buy_trader: str
    sell_trader: str
    timestamp: float


@dataclass
class Position:
    """ Position tracking data structure """
    quantity: int
    average_price: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    last_update: float

    @property
    def total_pnl(self) -> float:
        """ Total PnL including realized and unrealized """
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_long(self) -> bool:
        """ Check if position is long """
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """ Check if position is short """
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """ Check if position is flat """
        return self.quantity == 0
    

@dataclass
class MarketState:
    """" Market state information """
    mid_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: float
    bid_volume: float
    ask_volume: float
    last_trade_price: Optional[float]
    tick: int
    fair_value: float

    @property
    def bid_ask_imbalance(self) -> float:
        """ Calculate bid-ask volume imbalance """

        total = self.bid_volume + self.ask_volume
        if total > 0:
            return (self.bid_volume - self.ask_volume) / total
        return 0.0
    

@dataclass
class Quote:
    """ Quote (order to be placed) """
    side: Side
    price: float
    size: int
    level: int # 0 for best bid/ask, 1 for next level, etc.


@dataclass
class QuotePair:
    """ Bid and ask quotes at a single level """
    bid: Optional[Quote]
    ask: Optional[Quote]


@dataclass
class Signal:
    """ Trading signal with metadata """
    name: str
    value: float
    confidence: float # 0-1 confidence in the signal
    timestamp: float


class IOrderBook(ABC):
    """ Interface for order book operations """

    @abstractmethod
    def add_order(self, order: Order) -> int:
        """ Add order to order book, return order id """
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """ Cancel order, return success status """
        pass

    @abstractmethod
    def get_market_state(self) -> MarketState:
        """ Get current market state """
        pass

    @abstractmethod
    def get_order_book_snapshot(self, levels: int = 5) -> Dict[str, Any]:
        """ Get order book snapshot at specified levels """
        pass


class IVolatilityModel(ABC):
    """ Interface for volatility models """

    @abstractmethod
    def update(self, price: float, timestamp: float) -> None:
        """ Update model with new price data """
        pass

    @abstractmethod
    def get_forecast(self, horizon: int = 1) -> float:
        """ Get volatility forecast for next horizon ticks """
        pass

    @abstractmethod
    def get_current_volatility(self) -> float:
        """ Get current volatility estimate """
        pass

    @abstractmethod
    def should_refit(self) -> bool:
        """ Check if the model should be refit """
        pass

    @abstractmethod
    def fit(self) -> None:
        """ Fit model to recent data """
        pass


class IPricer(ABC):
    """ Interface for pricing models """

    @abstractmethod
    def calculate_fair_price(self, market_state: MarketState) -> float:
        """ Calculate fair price based on market state """
        pass

    @abstractmethod
    def calculate_microprice(self, order_book_snapshot: Dict) -> float:
        """ Calculate microprice from order book snapshot """
        pass


class ISpreadCalculator(ABC):
    """ Interface for spread calculators """

    @abstractmethod
    def calculate_spread(self, signals: Dict[str, Signal], position: Position) -> float:
        """ Calculate optimal spread given signals and position """
        pass


class IRiskManager(ABC):
    """ Interface for risk management """

    @abstractmethod
    def check_limits(self, position: Position, market_state: MarketState) -> bool:
        """ Check if position is within risk limits """
        pass

    @abstractmethod
    def get_risk_metrics(self, position: Position) -> Dict[str, float]:
        """ Calculate current risk metrics """
        pass

    @abstractmethod
    def should_stop_trading(self, position: Position) -> bool:
        """ Check if trading should be stopped """
        pass


class IAdverseSelectionDetector(ABC):
    """ Interface for adverse selection detection """

    @abstractmethod
    def update(self, trade: Trade, price_after: float) -> None:
        """ Update model with trade outcome """

    @abstractmethod
    def get_toxicity_score(self, trader_id: str) -> float:
        """ Get toxicity score for a trader """
        pass

    @abstractmethod
    def get_spread_adjustment(self) -> float:
        """ Get spread adjustment for adverse selection """
        pass


class IOrderManager(ABC):
    """ Interface for order management """

    @abstractmethod
    def update_quotes(self, quotes: List[QuotePair], order_book: IOrderBook) -> None:
        """ Update quotes based on order book """
        pass

    @abstractmethod
    def cancel_all_orders(self, order_book: IOrderBook) -> None:
        """ Cancel all active orders """
        pass

    @abstractmethod
    def get_active_orders(self) -> Dict[int, Order]:
        """ Get dictionary of active orders """
        pass

    @abstractmethod
    def on_trade(self, trade: Trade) -> None:
        """ Handle trade event """
        pass


class IPositionTracker(ABC):
    """ Interface for position tracking """

    @abstractmethod
    def update_position(self, trade: Trade) -> None:
        """ Update position with executed trade """
        pass

    @abstractmethod
    def get_position(self) -> Position:
        """ Get current position """
        pass

    @abstractmethod
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """ Calculate PnL for current position (realized and unrealized) """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Reset position to flat """
        pass


class IMarketMaker(ABC):
    """ Interface for market maker """

    @abstractmethod
    def update(self, order_book: IOrderBook, market_state: MarketState) -> None:
        """ Main update cycle """
        pass

    @abstractmethod
    def on_trade(self, trade: Trade) -> None:
        """ Handle trade event """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """ Get performance metrics """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Reset market maker to initial state """
        pass

    
        