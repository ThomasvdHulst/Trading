""" This file contains the Market Maker strategy implementation """

# Import used libraries
from order_book import Side


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

    
    def update_quotes(self, order_book, market_state=None):
        """ Update market maker quotes based on current market conditions """
        # Note to self: market_state is currently not used, but it is a placeholder for future use
        # It is used to pass in the current market state to the market maker, but it is not used in the current implementation
        # We do use it currently for other participants (participants.py)

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
        """ Calculate bid and ask prices based on inventory and market conditions 
        Example:
            Initial conditions
                mid_price = 100.00
                BASE_SPREAD = 0.10          # 10 cents total spread
                inventory = 500             # Long 500 shares
                INVENTORY_SKEW_FACTOR = 0.0002
                volatility_estimate = 0.015  # 1.5% volatility

            Calculations
                half_spread = 0.10 / 2 = 0.05
                inventory_skew = 500 * 0.0002 = 0.10
                volatility_adjustment = 0.015 * 100 = 1.50

            Final prices
                bid_price = 100.00 - 0.05 - 0.10 - 1.50 = 98.35
                ask_price = 100.00 + 0.05 - 0.10 + 1.50 = 101.45

            Result: Quotes are $98.35 / $101.45
              - Wide spread due to high volatility
              - Skewed down due to long inventory position
        """


        mid_price = book_depth['mid_price']

        # Base spread (half on each side)
        half_spread = self.config.BASE_SPREAD / 2

        # Inventory skew - adjust prices to reduce inventory
        inventory_skew = self.inventory * self.config.INVENTORY_SKEW_FACTOR

        # Volatiltiy adjustment - widen spread based on volatility
        volatility_adjustment = self.volatility_estimate * 100

        # Calculate quotes
        bid_price = mid_price - half_spread - inventory_skew - volatility_adjustment
        ask_price = mid_price + half_spread - inventory_skew + volatility_adjustment

        # Ensure we maintain minimum spread
        if ask_price - bid_price < self.config.TICK_SIZE:
            ask_price = bid_price + self.config.TICK_SIZE

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
        for order_id in list(self.buy_orders.keys()):
            if order_book.cancel_order(order_id):
                del self.buy_orders[order_id]

        # Cancel sell orders
        for order_id in list(self.sell_orders.keys()):
            if order_book.cancel_order(order_id):
                del self.sell_orders[order_id]


    def on_trade(self, trade):
        """ Process a trade that we participated in """

        price = trade['price']
        quantity = trade['quantity']

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


    def get_metrics(self):
        """ Get current market maker metrics """

        total_pnl = self.realized_pnl + self.unrealized_pnl

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