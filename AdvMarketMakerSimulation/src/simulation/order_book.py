""" This file contains the order book implementation. """

# Import used libraries
import heapq
from typing import Dict, List, Any

from market_maker.core.base import IOrderBook, Order, Trade, MarketState, Side
from config.config import MarketConfig


class OrderBook(IOrderBook):
    """ Simulated order book with matching engine. """

    def __init__(self, config: MarketConfig):
        """ Initialize order book.
        
        Args:
            config: Market configuration
        """

        self.config = config

        # Order storage
        self.buy_orders = [] # Max heap (negative prices)
        self.sell_orders = [] # Min heap
        self.orders_dict: Dict[int, Order] = {} # Order ID -> Order

        # Order ID generation
        self.next_order_id = 1
        self.next_trade_id = 1

        # Current state
        self.current_timestamp = 0
        self.last_trade_price = config.initial_price

        # Trade history
        self.trade_history: List[Trade] = []


    def add_order(self, order: Order) -> int:
        """ Add order to book and attempt matching. 
        
        Args:
            order: Order to add

        Returns:
            Order ID if successful, -1 if failed
        """

        # Validate and adjust order
        price = round(order.price / self.config.tick_size) * self.config.tick_size
        quantity = (order.quantity // self.config.lot_size) * self.config.lot_size

        if quantity <= 0:
            return -1
        
        # Create internal order with generated ID
        internal_order = Order(
            order_id = self.next_order_id,
            trader_id = order.trader_id,
            side = order.side,
            price = price,
            quantity = quantity,
            timestamp = self.current_timestamp,
            order_type = order.order_type,
        )

        self.next_order_id += 1
        self.orders_dict[internal_order.order_id] = internal_order

        # Try to match immediately
        self._try_match_order(internal_order)

        # Add remaining quantity to book
        if internal_order.quantity > 0:
            if internal_order.side == Side.BUY:
                heapq.heappush(self.buy_orders, (-price, internal_order.timestamp, internal_order.order_id))
            else:
                heapq.heappush(self.sell_orders, (price, internal_order.timestamp, internal_order.order_id))

        return internal_order.order_id
    

    def cancel_order(self, order_id: int) -> bool:
        """ Cancel an order.
        
        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found
        """

        if order_id in self.orders_dict:
            del self.orders_dict[order_id]
            self._clean_heaps()
            return True
        
        return False
    

    def _try_match_order(self, order: Order) -> None:
        """ Try to match order against opposite side. 
        
        Args:
            order: Order to match
        """

        if order.side == Side.BUY:
            # Try to match with sells
            while self.sell_orders and order.quantity > 0:
                best_price, _, best_id = self.sell_orders[0]

                if best_id not in self.orders_dict:
                    heapq.heappop(self.sell_orders)
                    continue

                best_order = self.orders_dict[best_id]

                # Check if prices cross
                if order.price >= best_price:
                    # Execute trade
                    trade_quantity = min(order.quantity, best_order.quantity)
                    self._execute_trade(order, best_order, trade_quantity, best_price)

                    # Update order quantities
                    order.quantity -= trade_quantity
                    best_order.quantity -= trade_quantity

                    # Remove filled orders
                    if best_order.quantity == 0:
                        heapq.heappop(self.sell_orders)
                        del self.orders_dict[best_id]
                else:
                    break

        else: # Sell order
            # Try to match with buys
            while self.buy_orders and order.quantity > 0:
                neg_best_price, _, best_id = self.buy_orders[0]
                best_price = -neg_best_price

                if best_id not in self.orders_dict:
                    heapq.heappop(self.buy_orders)
                    continue

                best_order = self.orders_dict[best_id]

                # Check if prices cross
                if order.price <= best_price:
                    # Execute trade
                    trade_quantity = min(order.quantity, best_order.quantity)
                    self._execute_trade(best_order, order, trade_quantity, best_price)

                    # Update order quantities
                    order.quantity -= trade_quantity
                    best_order.quantity -= trade_quantity

                    # Remove filled orders
                    if best_order.quantity == 0:
                        heapq.heappop(self.buy_orders)
                        del self.orders_dict[best_id]

                else:
                    break

        
    def _execute_trade(self, buy_order: Order, sell_order: Order, quantity: int, price: float) -> None:
        """ Execute a trade between two orders. 
        
        Args:
            buy_order: Buy order
            sell_order: Sell order
            quantity: Trade quantity
            price: Trade price
        """

        # Create trade
        trade = Trade(
            trade_id = self.next_trade_id,
            price = price,
            quantity = quantity,
            buy_order_id = buy_order.order_id,
            sell_order_id = sell_order.order_id,
            buy_trader = buy_order.trader_id,
            sell_trader = sell_order.trader_id,
            timestamp = self.current_timestamp,
        )

        self.next_trade_id += 1
        self.trade_history.append(trade)
        self.last_trade_price = price


    def get_market_state(self) -> MarketState:
        """ Get current market state.
        
        Returns:
            Current market state
        """

        # Find best bid and ask
        best_bid = None
        best_ask = None
        bid_volume = 0
        ask_volume = 0

        # Clean and find best bid
        while self.buy_orders:
            neg_price, _, order_id = self.buy_orders[0]
            if order_id in self.orders_dict:
                best_bid = -neg_price
                bid_volume = self.orders_dict[order_id].quantity
                break
            else:
                heapq.heappop(self.buy_orders)

        # Clean and find best ask
        while self.sell_orders:
            price, _, order_id = self.sell_orders[0]
            if order_id in self.orders_dict:
                best_ask = price
                ask_volume = self.orders_dict[order_id].quantity
                break
            else:
                heapq.heappop(self.sell_orders)

        # Calculate mid and spread
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
        elif best_bid:
            mid_price = best_bid
            spread = 0.0
        elif best_ask:
            mid_price = best_ask
            spread = 0.0
        else:
            mid_price = self.last_trade_price
            spread = 0.0

        
        return MarketState(
            mid_price = mid_price,
            best_bid = best_bid,
            best_ask = best_ask,
            spread = spread,
            bid_volume = bid_volume,
            ask_volume = ask_volume,
            last_trade_price = self.last_trade_price,
            tick = self.current_timestamp,
            fair_value = mid_price,
        )
    

    def get_order_book_snapshot(self, levels: int = 5) -> Dict[str, Any]:
        """ Get order book snapshot.
        
        Args:
            levels: Number of levels to return

        Returns:
            Order book snapshot
        """

        # Aggregate by price level
        buy_levels = {}
        sell_levels = {}

        # Process buy orders
        for neg_price, _, order_id in self.buy_orders:
            if order_id in self.orders_dict:
                price = -neg_price
                order = self.orders_dict[order_id]
                if price not in buy_levels:
                    buy_levels[price] = 0
                buy_levels[price] += order.quantity

        # Process sell orders
        for price, _, order_id in self.sell_orders:
            if order_id in self.orders_dict:
                order = self.orders_dict[order_id]
                if price not in sell_levels:
                    sell_levels[price] = 0
                sell_levels[price] += order.quantity
                
        # Sort and limit
        sorted_bids = sorted(buy_levels.items(), reverse=True)[:levels]
        sorted_asks = sorted(sell_levels.items())[:levels]

        market_state = self.get_market_state()

        return {
            'bids': sorted_bids,
            'asks': sorted_asks,
            'mid_price': market_state.mid_price,
            'spread': market_state.spread,
            'timestamp': self.current_timestamp,
        }
    

    def _clean_heaps(self) -> None:
        """ Removed cancelled orders from heaps. """

        # Clean buy orders
        temp = []
        while self.buy_orders:
            item = heapq.heappop(self.buy_orders)
            if item[2] in self.orders_dict: # Order id is at index 2
                temp.append(item)
        for item in temp:
            heapq.heappush(self.buy_orders, item)


        # Clean sell orders
        temp = []
        while self.sell_orders:
            item = heapq.heappop(self.sell_orders)
            if item[2] in self.orders_dict: # Order id is at index 2
                temp.append(item)
        for item in temp:
            heapq.heappush(self.sell_orders, item)


    def tick(self) -> None:
        """ Advance time by 1 tick. """
        self.current_timestamp += 1


    def get_trades_for_tick(self, tick: int) -> List[Trade]:
        """ Get all trades that occurred at a specific tick.
        
        Args:
            tick: Tick number

        Returns:
            List of trades
        """

        return [trade for trade in self.trade_history if trade.timestamp == tick]