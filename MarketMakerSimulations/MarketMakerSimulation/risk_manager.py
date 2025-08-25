""" This file contains the risk management for the market maker """

# Import used libraries
import numpy as np

# Deque is a double-ended queue, which is a list-like container with fast appends and pops on either end
# Also, it has a rolling window feature, which is useful for tracking history, so if the maxlen is 100
# then the deque will only keep the last 100 items, and when a new item is added, the oldest item is removed
from collections import deque

class RiskManager:
    """ Manages risk limits for the market maker """

    def __init__(self, config):
        self.config = config

        # Risk limits
        self.max_inventory = config.MAX_INVENTORY
        self.max_position_value = config.MAX_POSITION_VALUE
        self.stop_loss_threshold = config.STOP_LOSS_THRESHOLD

        # Risk metrics tracking
        self.inventory_history = deque(maxlen=100)
        self.pnl_history = deque(maxlen=100)
        self.trade_history = deque(maxlen=100)

        # Adverse selection tracking
        self.trades_by_counterparty = {}
        self.toxic_flow_indicators = deque(maxlen=50)

        # Risk state
        self.risk_state = "normal" # normal, caution, stop
        self.inventory_imbalance_time = 0


    def assess_risk(self, market_maker, order_book):
        """ Assess risk levels """

        mm_metrics = market_maker.get_metrics()
        book_depth = order_book.get_book_depth()

        risk_assessment = {
            'inventory_risk': self._assess_inventory_risk(mm_metrics),
            'pnl_risk': self._assess_pnl_risk(mm_metrics),
            'adverse_selection_risk': self._assess_adverse_selection(market_maker),
            'market_risk': self._assess_market_conditions(book_depth),
            'overall_risk': 'low'
        }

        # Determine overall risk
        risk_levels = [risk_assessment[key] for key in risk_assessment if key != 'overall_risk']
        if 'high' in risk_levels:
            risk_assessment['overall_risk'] = 'high'
            self.risk_state = 'stop'
        elif risk_levels.count('medium') >= 2:
            risk_assessment['overall_risk'] = 'medium'
            self.risk_state = 'caution'
        else:
            risk_assessment['overall_risk'] = 'low'
            self.risk_state = 'normal'

        return risk_assessment
    

    def _assess_inventory_risk(self, mm_metrics):
        """ Assess inventory risk """

        inventory = mm_metrics['inventory']
        inventory_ratio = abs(inventory) / self.max_inventory

        # Track how long we've been imbalanced
        if inventory_ratio > 0.7:
            self.inventory_imbalance_time += 1
        else:
            self.inventory_imbalance_time = 0

        # Determine risk level
        if inventory_ratio > 0.9:
            return 'high'
        elif inventory_ratio > 0.7 or self.inventory_imbalance_time > 50:
            return 'medium'
        else:
            return 'low'
        

    def _assess_pnl_risk(self, mm_metrics):
        """ Assess PnL risk """

        current_pnl = mm_metrics['total_pnl']

        # Check against stop loss
        if current_pnl < self.stop_loss_threshold:
            return 'high'
        elif current_pnl < self.stop_loss_threshold * 0.5:
            return 'medium'
    
        # Check drawdown if we have a history
        if len(self.pnl_history) > 0:
            max_pnl = max(self.pnl_history)
            drawdown = (max_pnl - current_pnl) / abs(max_pnl) if max_pnl != 0 else 0

            if drawdown > 0.2: # 20% drawdown
                return 'medium'
            
        return 'low'
    

    def _assess_adverse_selection(self, market_maker):
        """ Assess adverse selection risk """

        if len(market_maker.trades_executed) < 10:
            return 'low' # Not enough data
        
        # Look at recent trades
        recent_trades = market_maker.trades_executed[-10:]

        # Simple adverse selection check
        # Are we consistently buying before price drops or selling before price rises?
        adverse_trades = 0
        for i, trade in enumerate(recent_trades[:-1]):
            next_trade = recent_trades[i + 1]

            if trade['side'] == 'buy' and next_trade['price'] < trade['price']:
                adverse_trades += 1
            elif trade['side'] == 'sell' and next_trade['price'] > trade['price']:
                adverse_trades += 1

        adverse_ratio = adverse_trades / len(recent_trades)

        if adverse_ratio > 0.7:
            return 'high'
        elif adverse_ratio > 0.5:
            return 'medium'
        else:
            return 'low'
        

    def _assess_market_conditions(self, book_depth):
        """ Assess market conditions """

        spread = book_depth['spread']

        # Wide spread incidates uncertainty
        if spread > self.config.BASE_SPREAD * 3:
            return 'high'
        elif spread > self.config.BASE_SPREAD * 2:
            return 'medium'
        
        # Check if book is one-sided (potential price pressure)
        if book_depth['best_bid'] is None or book_depth['best_ask'] is None:
            return 'medium'
        
        return 'low'
    

    def get_risk_adjusted_parameters(self, risk_assessment):
        """ Get risk-adjusted parameters """

        params = {
            'spread_multiplier': 1.0,
            'size_multiplier': 1.0,
            'should_trade': True
        }

        overall_risk = risk_assessment['overall_risk']

        if overall_risk == 'high':
            params['should_trade'] = False
        elif overall_risk == 'medium':
            params['spread_multiplier'] = 1.5 # Wider spread
            params['size_multiplier'] = 0.5 # Smaller size
        
        # Adjust for specific risks
        if risk_assessment['inventory_risk'] == 'high':
            params['size_multiplier'] *= 0.3
            params['spread_multiplier'] *= 1.2

        if risk_assessment['adverse_selection_risk'] == 'high':
            params['spread_multiplier'] *= 1.3


        return params
    

    def update_history(self, mm_metrics, trades):
        """ Update risk tracking history"""

        self.inventory_history.append(mm_metrics['inventory'])
        self.pnl_history.append(mm_metrics['total_pnl'])

        for trade in trades:
            self.trade_history.append(trade)


    def calculate_var(self, confidence_level=0.95):
        """ Calculate Value at Risk (VaR) """

        if len(self.pnl_history) < 20:
            return 0
        
        returns = []
        for i in range(1, len(self.pnl_history)):
            returns.append(self.pnl_history[i] - self.pnl_history[i - 1])

        if returns:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return abs(var)
        
        return 0
    

    def get_risk_metrics(self):
        """ Get risk metrics """

        metrics = {
            'risk_state': self.risk_state,
            'inventory_imbalance_time': self.inventory_imbalance_time,
            'var_95': self.calculate_var(0.95),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }

        return metrics
    

    def _calculate_max_drawdown(self):
        """ Calculate max drawdown from PnL history """

        if len(self.pnl_history) < 2:
            return 0
        
        peak = self.pnl_history[0]
        max_dd = 0

        for pnl in self.pnl_history:
            if pnl > peak:
                peak = pnl

            drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd
    

    def _calculate_sharpe_ratio(self):
        """ Calculate Sharpe ratio from returns """

        if len(self.pnl_history) < 2:
            return 0
        
        returns = []
        for i in range(1, len(self.pnl_history)):
            returns.append(self.pnl_history[i] - self.pnl_history[i - 1])

        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return > 0:
                # Annualized Sharpe (assuming tick frequency)
                sharpe = mean_return / std_return * np.sqrt(252 * 6.5 * 60 * 60 / self.config.TICK_FREQUENCY_MS * 1000)
                return sharpe
            
        return 0
        