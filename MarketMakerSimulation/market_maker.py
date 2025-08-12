""" This file contains the Market Maker strategy implementation """

# Import used libraries
from order_book import Side
import numpy as np
from collections import deque


class MarketMaker:
    """ Basic Market Maker strategy """

    def __init__(self, config, trader_id="MM"):
        self.config = config
        self.trader_id = trader_id

        # Track our orders
        self.buy_orders = {} # order_id -> (price, quantity)
        self.sell_orders = {} # order_id -> (price, quantity)

        # Track inventory and PnL
        self.inventory = 0 # Inventory is the number of shares we hold
        self.cash = config.INITIAL_CAPITAL # Cash is the amount of money we have
        self.total_traded_volume = 0 # Total volume is the total number of shares traded

        self.realized_pnl = 0 # Realized PnL is the PnL from completed round-trip trades
        self.unrealized_pnl = 0 # Unrealized PnL is the PnL from current open position
        self.total_fees_paid = 0 # Total fees paid is the total amount of fees paid
        self.inventory_cost_basis = 0 # Total cost paid for current inventory (including fees)

        # Risk metrics
        self.max_inventory_held = 0 # Max inventory held is the maximum number of shares we have held
        self.min_inventory_held = 0 # Min inventory held is the minimum number of shares we have held

        # Performance tracking
        self.trades_executed = [] # Trades executed is the list of trades that have been executed

        # Market data
        self.current_mid_price = config.INITIAL_PRICE
        self.volatility_estimate = config.VOLATILITY

        # Adverse selection tracking
        self.trader_history = {} # trader_id -> list of trade outcomes
        self.trader_toxicity_scores = {} # trader_id -> toxicity score (0-1)
        self.lookback_trades = 20 # Number of trades to consider for toxicity score calculation
        self.toxicity_decay = 0.95 # Decay factor for toxicity score
        self.min_trades_for_scoring = 5 # Minimum number of trades required for toxicity score calculation

        # Price tracking for adverse selection
        self.post_trade_price_impacts = [] # Track price movements after trades
        self.adverse_selection_threshold = 0.6 # Threshold for adverse selection

        # Dynamic spread calculation
        self.price_history = deque(maxlen=100) # Track recent prices for volatility
        self.spread_history = deque(maxlen=50) # Track recent spreads
        self.trade_intensity_history = deque(maxlen=20) # Trade frequency
        self.volatility_window = 20 # Number of ticks to average volatility over

        # Spread components tracking
        self.base_spread = config.BASE_SPREAD
        self.min_spread = config.TICK_SIZE * 2 # Minimum viable spread
        self.max_spread = config.TICK_SIZE * 100 # Maximum spread cap

        # Market regime detection
        self.market_regime = "normal" # normal, volatile, trending, quiet
        self.regime_detection_window = 50

        # Real-time metrics
        self.realized_volatility = config.VOLATILITY
        self.trade_intensity = 0
        self.recent_pnl_trend = 0
        self.bid_ask_imbalance = 0


    
    def update_quotes(self, order_book, market_state=None):
        """ Update market maker quotes based on current market conditions """
        
        # Get tick
        if market_state and 'tick' in market_state:
            self.current_tick = market_state['tick']
        else:
            self.current_tick = 0

        # Store order book
        self.order_book = order_book

        # Cancel all existing orders first
        self.cancel_all_orders(order_book)

        # Get current market data
        book_depth = order_book.get_book_depth()
        self.current_mid_price = book_depth['mid_price']

        # Calculate quote prices
        bid_price, ask_price = self._calculate_quote_prices(book_depth)

        # Calculate quote sizes
        bid_size, ask_size = self._calculate_quote_sizes()

        #print(f"Bid price: {bid_price}, Ask price: {ask_price}")
        #print(f"Bid size: {bid_size}, Ask size: {ask_size}")
        #print(f"Mid price: {self.current_mid_price}")
       # print(f"Inventory: {self.inventory}")
        #print('--------------------------------')

        # Check risk limits before placing new orders
        # If we are at risk limits, we do not place any new orders
        #if self._check_risk_limits():
        if True:

            # Place buy order
            if bid_size > 0 and bid_price > 0:
                order_id = order_book.add_order(
                    side = Side.BUY,
                    price = bid_price,
                    quantity = bid_size,
                    trader_id = self.trader_id
                )

                # The add order function returns -1 if the order is invalid
                if order_id > 0:
                    self.buy_orders[order_id] = (bid_price, bid_size)

            # Place sell order
            if ask_size > 0 and ask_price > 0:
                order_id = order_book.add_order(
                    side = Side.SELL,
                    price = ask_price,
                    quantity = ask_size,
                    trader_id = self.trader_id
                )

                # The add order function returns -1 if the order is invalid
                if order_id > 0:
                    self.sell_orders[order_id] = (ask_price, ask_size)

        
    def _calculate_quote_prices(self, book_depth):
        """ Calculate bid and ask prices based on inventory and market conditions """


        mid_price = book_depth['mid_price']

        # Update market metrics first
        self.update_market_metrics(self.order_book)

        # Get dynamic spread instead of using base spread
        total_spread = self.calculate_dynamic_spread()
        half_spread = total_spread / 2

        # Inventory skew - adjust prices to reduce inventory
        inventory_skew = self.inventory * self.config.INVENTORY_SKEW_FACTOR

        # Additional skew based on book imbalance
        # If more bids than asks, markets wants to buy - we should sell higher
        imbalance_skew = self.bid_ask_imbalance * half_spread * 0.2

        # Calculate base quotes
        bid_price = mid_price - half_spread - inventory_skew - imbalance_skew
        ask_price = mid_price + half_spread - inventory_skew + imbalance_skew

        # Adjust for market regime
        #if self.market_regime == "trending":
        #    # In trending market, fade the trend
        #    trend_adjustment = (self.price_history[-1] - self.price_history[-10]) if len(self.price_history) > 10 else 0
        #    if trend_adjustment > 0: # Up trend
        #        # Be more willing to sell (lower ask), less willing to buy (higher bid)
        #        ask_price -= half_spread * 0.1
        #        bid_price -= half_spread * 0.2
        #    else: # Down trend
        #        # Be more willing to buy (higher bid), less willing to sell (lower ask)
        #        bid_price += half_spread * 0.1
        #        ask_price += half_spread * 0.2

        # Ensure minimum spread
        if ask_price - bid_price < self.min_spread:
            mid = (ask_price + bid_price) / 2
            bid_price = mid - self.min_spread / 2
            ask_price = mid + self.min_spread / 2

        # Round to tick size
        bid_price = round(bid_price / self.config.TICK_SIZE) * self.config.TICK_SIZE
        ask_price = round(ask_price / self.config.TICK_SIZE) * self.config.TICK_SIZE

        return bid_price, ask_price


    def _calculate_quote_sizes(self):
        """ Calculate quote sizes based on inventory and risk limits """

        base_size = self.config.QUOTE_SIZE

        # Reduce size if we're near risk limits
        inventory_ratio = abs(self.inventory) / self.config.MAX_INVENTORY
        size_multiplier = max(0.1, 1 - inventory_ratio * 0.8)

        # Adjust sizes based on inventory position
        if self.inventory > 0:
            # Long inventory - quote more aggressively on sell side
            bid_size = int(base_size * size_multiplier * 0.7)
            ask_size = int(base_size * size_multiplier * 1.3)
        elif self.inventory < 0:
            # Short inventory - quote more aggressively on buy side
            bid_size = int(base_size * size_multiplier * 1.3)
            ask_size = int(base_size * size_multiplier * 0.7)
        else:
            # Neutral inventory - quote evenly
            bid_size = int(base_size * size_multiplier)
            ask_size = int(base_size * size_multiplier)
            
        # Ensure we don't exceed position limits
        if self.inventory + bid_size > self.config.MAX_INVENTORY:
            bid_size = max(0, self.config.MAX_INVENTORY - self.inventory)

        if self.inventory - ask_size < -self.config.MAX_INVENTORY:
            ask_size = max(0, self.config.MAX_INVENTORY + self.inventory)

        # Round to lot size
        bid_size = int(bid_size / self.config.LOT_SIZE) * self.config.LOT_SIZE
        ask_size = int(ask_size / self.config.LOT_SIZE) * self.config.LOT_SIZE

        return bid_size, ask_size
    

    def _check_risk_limits(self):
        """ Check if we're at risk limits and adjust positions if needed """

        # Check PnL limits
        # If the realized PnL is less than the stop loss threshold, we are at risk
        if self.realized_pnl < self.config.STOP_LOSS_THRESHOLD:
            return False
        
        # Check position value limits
        # If the position value is greater than the maximum position value, we are at risk
        position_value = abs(self.inventory * self.current_mid_price)
        if position_value > self.config.MAX_POSITION_VALUE:
            return False
        
        # If we are not at risk, return True so we can place new orders
        return True
    

    def cancel_all_orders(self, order_book):
        """ Cancel all orders for this trader """

        # Cancel buy orders
        buy_order_ids = list(self.buy_orders.keys())  # Make a copy of keys
        for order_id in buy_order_ids:
            if order_book.cancel_order(order_id):
                del self.buy_orders[order_id]
        
        # Cancel sell orders  
        sell_order_ids = list(self.sell_orders.keys())
        for order_id in sell_order_ids:
            if order_book.cancel_order(order_id):
                del self.sell_orders[order_id]
        
        # Clear the dictionaries completely
        self.buy_orders.clear()
        self.sell_orders.clear()


    def on_trade(self, trade):
        """ Process a trade that we participated in """

        price = trade['price']
        quantity = trade['quantity']
        
        # Remove filled orders from our tracking
        if trade['buy_trader'] == self.trader_id:
            # Find and remove the buy order
            for order_id in list(self.buy_orders.keys()):
                if order_id == trade.get('buy_order_id'):
                    del self.buy_orders[order_id]
                    break
                    
        elif trade['sell_trader'] == self.trader_id:
            # Find and remove the sell order
            for order_id in list(self.sell_orders.keys()):
                if order_id == trade.get('sell_order_id'):
                    del self.sell_orders[order_id]
                    break

        # If we bought
        if trade['buy_trader'] == self.trader_id:
            fee = quantity * self.config.EXCHANGE_FEE
            total_cost = price * quantity + fee
            
            # Update cash and fees
            self.cash -= total_cost
            self.total_fees_paid += fee
            
            # Update inventory and cost basis
            if self.inventory >= 0:
                # Adding to long position or starting new long position
                self.inventory += quantity
                self.inventory_cost_basis += total_cost
            else:
                # Covering short position
                if quantity >= abs(self.inventory):
                    # Fully covering short and potentially going long
                    shares_covering_short = abs(self.inventory)
                    shares_going_long = quantity - shares_covering_short
                    
                    # Realize PnL from covering short
                    if self.inventory != 0:
                        avg_short_price = self.inventory_cost_basis / abs(self.inventory)
                        short_pnl = (avg_short_price - price) * shares_covering_short
                        self.realized_pnl += short_pnl
                    
                    # Reset for new long position  
                    self.inventory = shares_going_long
                    long_position_cost = (price * shares_going_long) + (fee * shares_going_long / quantity)
                    self.inventory_cost_basis = long_position_cost
                else:
                    # Partially covering short position
                    cost_per_share = self.inventory_cost_basis / abs(self.inventory)
                    short_pnl = (cost_per_share - price) * quantity
                    self.realized_pnl += short_pnl
                    
                    self.inventory += quantity  # Moving towards zero
                    self.inventory_cost_basis -= cost_per_share * quantity
        
            # Track the trade
            self.trades_executed.append({
                'timestamp': trade['timestamp'],
                'side': 'buy',
                'price': price,
                'quantity': quantity,
                'inventory_after': self.inventory
            })

        # If we sold
        elif trade['sell_trader'] == self.trader_id:
            fee = quantity * self.config.EXCHANGE_FEE
            net_proceeds = price * quantity - fee
            
            # Update cash and fees
            self.cash += net_proceeds
            self.total_fees_paid += fee
            
            # Update inventory and cost basis
            if self.inventory > 0:
                # Selling from long position
                if quantity >= self.inventory:
                    # Selling entire long position and potentially going short
                    shares_from_long = self.inventory
                    shares_going_short = quantity - self.inventory
                    
                    # Realize PnL from closing long position
                    if self.inventory_cost_basis > 0:
                        avg_long_price = self.inventory_cost_basis / self.inventory
                        long_pnl = (price - avg_long_price) * shares_from_long
                        self.realized_pnl += long_pnl
                    
                    # Reset for new short position
                    self.inventory = -shares_going_short
                    short_position_cost = (price * shares_going_short) + (fee * shares_going_short / quantity)
                    self.inventory_cost_basis = short_position_cost
                else:
                    # Partially selling long position
                    cost_per_share = self.inventory_cost_basis / self.inventory
                    long_pnl = (price - cost_per_share) * quantity
                    self.realized_pnl += long_pnl
                    
                    self.inventory -= quantity
                    self.inventory_cost_basis -= cost_per_share * quantity
            else:
                # Adding to short position or starting new short position
                self.inventory -= quantity
                total_short_value = price * quantity + fee
                self.inventory_cost_basis += total_short_value
            
            # Track the trade
            self.trades_executed.append({
                'timestamp': trade['timestamp'],
                'side': 'sell',
                'price': price,
                'quantity': quantity,
                'inventory_after': self.inventory
            })
            
        # Update inventory limits
        self.max_inventory_held = max(self.max_inventory_held, self.inventory)
        self.min_inventory_held = min(self.min_inventory_held, self.inventory)

        # Update volume
        self.total_traded_volume += quantity


    def update_pnl(self, current_price):
        """ Update PnL tracking 
        The realized PnL is tracked as trades are executed (already calculated in on_trade)
        The unrealized PnL is calculated from the current inventory position
        The total PnL is the sum of realized and unrealized PnL
        """

        # Unrealized PnL from current inventory position
        if self.inventory != 0:
            # Calculate average entry price from cost basis
            avg_entry_price = self.inventory_cost_basis / abs(self.inventory)
            
            # Calculate exit fees
            exit_fees = abs(self.inventory) * self.config.EXCHANGE_FEE
            
            if self.inventory > 0:
                # Long position: unrealized gain if current price > avg entry price
                self.unrealized_pnl = (current_price - avg_entry_price) * self.inventory - exit_fees
            else:
                # Short position: unrealized gain if current price < avg entry price
                self.unrealized_pnl = (avg_entry_price - current_price) * abs(self.inventory) - exit_fees
        else:
            self.unrealized_pnl = 0


    def track_trade_outcome(self, trade, price_after):
        """ Track the outcome of a trade to detect adverse selection """

        # Determine if we are the buyer or seller
        we_bought = (trade['buy_trader'] == self.trader_id)
        we_sold = (trade['sell_trader'] == self.trader_id)

        if not (we_bought or we_sold):
            return
        
        trade_price = trade['price']
        price_move = price_after - trade_price
        price_move_pct = price_move / trade_price

        # Determine counterparty
        counterparty = trade['sell_trader'] if we_bought else trade['buy_trader']

        # Initialize trader history if not exists
        if counterparty not in self.trader_history:
            self.trader_history[counterparty] = []
            self.trader_toxicity_scores[counterparty] = 0.5 # Start with neutral score

        # Determine if this trade was adverse for us
        if we_bought:
            # We bought - adverse if price went down
            adverse = (price_move_pct < 0)
            impact = -price_move_pct # Negative impact if price went down (bad for us...)
        else:
            # We sold - adverse if price went up
            adverse = (price_move_pct > 0)
            impact = price_move_pct # Positive impact if price went up (bad for us...)

        # Store trade outcome
        trade_outcome = {
            'timestamp': trade['timestamp'],
            'adverse': adverse,
            'impact': impact,
            'price_move': price_move_pct,
            'our_side': 'buy' if we_bought else 'sell',
        }

        self.trader_history[counterparty].append(trade_outcome)

        # Keep only the most recent trades
        if len(self.trader_history[counterparty]) > self.lookback_trades:
            self.trader_history[counterparty].pop(0)

        # Update toxicity score
        self._update_toxicity_score(counterparty)

    
    def _update_toxicity_score(self, trader_id):
        """ Update toxicity score for a trader based on their history """

        if trader_id not in self.trader_history:
            return
        
        trades = self.trader_history[trader_id]
        
        if len(trades) < self.min_trades_for_scoring:
            # Not enough trades to calculate score
            self.trader_toxicity_scores[trader_id] = 0.5
            return
        
        # Calculate weighted adverse selection rate (recent trades more important)
        weights = [self.toxicity_decay ** (len(trades) - i - 1) for i in range(len(trades))]
        weights_sum = sum(weights)

        # Calculate weighted average rate and average impact
        adverse_score = 0
        impact_score = 0

        for i, trade in enumerate(trades):
            # Calculate weight for this trade
            weight = weights[i] / weights_sum

            # If the trade was adverse, add to the adverse score
            adverse_score += weight * (1 if trade['adverse'] else 0)
            impact_score += weight * abs(trade['impact'])

        # Combine adverse frequency and impact magnitude
        # 70% weight on how often the trader is toxic, 30% weight on the magnitude of the impact
        toxicity = 0.7 * adverse_score + 0.3 * min(impact_score*50, 1)
        # Cap at 1.0
        toxicity = min(toxicity, 1.0)

        # Update score with smoothing
        old_score = self.trader_toxicity_scores.get(trader_id, 0.5)
        self.trader_toxicity_scores[trader_id] = 0.7 * toxicity + 0.3 * old_score


    def get_trader_adjustment(self, book_depth):
        """ Get spread adjustment based on trader toxicity scores """

        # Look at recent trades to see who is active
        recent_toxic_activity = 0
        total_recent_trades = 0

        for trader_id, score in self.trader_toxicity_scores.items():
            if trader_id in self.trader_history and self.trader_history[trader_id]:
                # Check if this trader has been recently active
                recent_trades = [t for t in self.trader_history[trader_id] if t['timestamp'] > self.current_tick - 50]

                if recent_trades:
                    # If the trader has been active, add to the total recent trades and recent toxic activity
                    total_recent_trades += len(recent_trades)
                    recent_toxic_activity += score * len(recent_trades)

        # Calculate the average toxicity of recent activity
        avg_toxicity = recent_toxic_activity / total_recent_trades if total_recent_trades > 0 else 0.5

        # Adjust spreads based on toxicity
        if avg_toxicity > self.adverse_selection_threshold:
            # High toxicity - widen spreads significantly
            spread_multiplier = 1.5 + (avg_toxicity - 0.5) * 2
            size_multiplier = 0.5
        elif avg_toxicity > 0.55:
            # Moderate toxicity - widen spreads moderately
            spread_multiplier = 1.2
            size_multiplier = 0.8
        else:
            # Low toxicity - normal spreads
            spread_multiplier = 1
            size_multiplier = 1

        return spread_multiplier, size_multiplier
    

    def get_trader_specific_quotes(self, trader_id, base_bid, base_ask):
        """ Adjust quotes based on trader toxicity scores """

        if trader_id not in self.trader_toxicity_scores:
            return base_bid, base_ask
        
        toxicity = self.trader_toxicity_scores[trader_id]

        if toxicity > self.adverse_selection_threshold:
            # This trader is toxic - widen spreads significantly
            spread = base_ask - base_bid
            extra_spread = spread * (toxicity - 0.5) * 2

            mid = (base_bid + base_ask) / 2
            new_bid = mid - (spread + extra_spread) / 2
            new_ask = mid + (spread + extra_spread) / 2

            return new_bid, new_ask

        return base_bid, base_ask


    def update_market_metrics(self, order_book):
        """ Update real-time market metrics for spread calculation """

        book_depth = order_book.get_book_depth()
        current_price = book_depth['mid_price']

        # Update price history
        self.price_history.append(current_price)

        # Update spread history
        if book_depth['spread'] > 0:
            self.spread_history.append(book_depth['spread'])

        # Calculate realized volatility
        if len(self.price_history) >= self.volatility_window:
            returns = []
            for i in range(1, self.volatility_window):
                ret = (self.price_history[-i] - self.price_history[-i-1]) / self.price_history[-i-1]
                returns.append(ret)
            self.realized_volatility = np.std(returns) if returns else self.config.VOLATILITY

        # Calculate trade intensity (trades per tick)
        recent_trades = len([t for t in order_book.trade_history if t['timestamp'] > self.current_tick - 20])
        self.trade_intensity = recent_trades / 20 if self.current_tick > 20 else 0
        self.trade_intensity_history.append(self.trade_intensity)

        # Calculate bid-ask imbalance
        self.bid_ask_imbalance = self._calculate_book_imbalance(book_depth)

        # Detect market regime
        self.market_regime = self._detect_market_regime()

        # Calculate recent PnL trend
        if len(self.trades_executed) >= 5:
            recent_trades = self.trades_executed[-5:]
            recent_pnl = sum([(t['price'] - current_price) * (1 if t['side'] == 'sell' else -1) for t in recent_trades])
            self.recent_pnl_trend = recent_pnl


    def _calculate_book_imbalance(self, book_depth):
        """ Calculate bid-ask imbalance """

        # Obtain the volumes for both sides
        bid_volume = sum([qty for price, qty in book_depth['bids']])
        ask_volume = sum([qty for price, qty in book_depth['asks']])

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            # Calculate imbalance
            imbalance = (bid_volume - ask_volume) / total_volume
            return np.clip(imbalance, -1, 1) # Clip between -1 and 1 (though it should never be outside of this range)
        return 0
    

    def _detect_market_regime(self):
        """ Detect current market regime based on recent activity """

        if len(self.price_history) < self.regime_detection_window:
            return "normal"

        # Calculate metrics for regime detection
        recent_prices = list(self.price_history)[-self.regime_detection_window:]
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, len(recent_prices))]

        volatility = np.std(returns) if returns else 0
        avg_volatiltiy = self.config.VOLATILITY

        # Trend detection
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        abs_trend = abs(price_change)
        
        # Trade intensity
        avg_trade_intensity = np.mean(self.trade_intensity_history) if self.trade_intensity_history else 0

        # Classify regimes
        if volatility > avg_volatiltiy * 2: # High volatility
            return "volatile"
        elif abs_trend > 0.002: # 0.2% move (significant trend)
            return "trending"
        elif avg_trade_intensity < 0.1: # Very few trades (low activity)
            return "quiet"
        else:
            return "normal"
        

    def calculate_dynamic_spread(self):
        """ Calculate dynamic spread based on market regime and recent activity """

        # Start with base spread
        dynamic_spread = self.base_spread

        # 1. Volatility adjustment
        vol_multiplier = 1
        if self.realized_volatility > 0:
            # Higher volatility -> wider spread
            vol_ratio = self.realized_volatility / self.config.VOLATILITY
            vol_multiplier = 0.5 + np.clip(vol_ratio, 0.5, 3)

        # 2. Trade intensity adjustment
        intensity_multiplier = 1
        if self.trade_intensity > 0:
            # More traders, more competition -> tighter spreads
            if self.trade_intensity > 2: 
                intensity_multiplier = 0.9
            elif self.trade_intensity < 0.5: # Low activity
                intensity_multiplier = 1.2

        # 3. Inventory risk adjustment
        inventory_multiplier = 1
        inventory_ratio = abs(self.inventory) / self.config.MAX_INVENTORY
        if inventory_ratio > 0.7:
            # High inventory -> wider spread
            inventory_multiplier = 1.3
        elif inventory_ratio > 0.5:
            inventory_multiplier = 1.1

        # 4. Recent PnL trend adjustment
        pnl_multiplier = 1
        if self.recent_pnl_trend < 0:
            # Losing money -> wider spread
            pnl_multiplier = 1.2
        elif self.recent_pnl_trend > 0:
            # Making money -> tighter spreads
            pnl_multiplier = 0.95

        # 5. Market regime adjustment
        regime_multipliers = {
            'normal': 1,
            'volatile': 1.5,
            'trending': 1.3,
            'quiet': 0.8
        }
        regime_multiplier = regime_multipliers.get(self.market_regime, 1)

        # 6. Adverse selection adjustment
        toxicity_spread_mult, toxicity_size_mult = self.get_trader_adjustment(None)
   
        # Combine all multipliers
        total_multiplier = (vol_multiplier * intensity_multiplier * inventory_multiplier * pnl_multiplier * regime_multiplier * toxicity_spread_mult)

        # Weight the factors (volatility and toxicity are more important)
        weighted_multiplier = (0.3 * vol_multiplier +
                               0.2 * toxicity_spread_mult +
                               0.15 * inventory_multiplier +
                               0.15 * regime_multiplier +
                               0.1 * intensity_multiplier +
                               0.1 * pnl_multiplier)

        # Calculate final spread
        dynamic_spread = self.base_spread * weighted_multiplier
        dynamic_spread = np.clip(dynamic_spread, self.min_spread, self.max_spread)

        # Store components for analysis
        self.spread_components = {
            'base': self.base_spread,
            'volatility_mult': vol_multiplier,
            'intensity_mult': intensity_multiplier,
            'inventory_mult': inventory_multiplier,
            'pnl_mult': pnl_multiplier,
            'regime_mult': regime_multiplier,
            'toxicity_mult': toxicity_spread_mult,
            'final_spread': dynamic_spread,
        }

        return dynamic_spread


    def get_metrics(self):
        """ Get current market maker metrics """

        total_pnl = self.realized_pnl + self.unrealized_pnl

        toxic_traders = [tid for tid, score in self.trader_toxicity_scores.items() if score > self.adverse_selection_threshold]
        avg_toxicity = sum(self.trader_toxicity_scores.values()) / len(self.trader_toxicity_scores) if self.trader_toxicity_scores else 0.5

        return {
            'inventory': self.inventory,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': total_pnl,
            'total_fees': self.total_fees_paid,
            'total_volume': self.total_traded_volume,
            'num_trades': len(self.trades_executed),
            'max_inventory': self.max_inventory_held,
            'min_inventory': self.min_inventory_held,
            'active_buy_orders': len(self.buy_orders),
            'active_sell_orders': len(self.sell_orders),
            'toxic_traders': toxic_traders,
            'avg_toxicity': avg_toxicity,
            'current_spread': self.spread_components.get('final_spread', self.base_spread) if hasattr(self, 'spread_components') else self.base_spread,
            'market_regime': self.market_regime if hasattr(self, 'market_regime') else 'normal',
            'realized_volatility': self.realized_volatility if hasattr(self, 'realized_volatility') else 0,
            'trade_intensity': self.trade_intensity if hasattr(self, 'trade_intensity') else 0,
            'book_imbalance': self.bid_ask_imbalance if hasattr(self, 'bid_ask_imbalance') else 0,
        }
    

    def reset(self):
        """ Reset market maker state """

        self.buy_orders = {}
        self.sell_orders = {}
        self.inventory = 0
        self.cash = self.config.INITIAL_CAPITAL
        self.total_traded_volume = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_fees_paid = 0
        self.trades_executed = []
        self.max_inventory_held = 0
        self.min_inventory_held = 0
        self.inventory_cost_basis = 0