""" This file contains trade classification algorithms to determine trade direction. """

# Import used libraries
import pandas as pd
from typing import Dict, Optional, List


class TradeClassifier:
    """
    Implements various trade classification algorithms.
    These algorithms infer trade direction when not directly observable.
    """

    def __init__(self):
        self.classification_history = []

    
    def lee_ready_algorithm(self, trade_price: float, mid_price: float, prev_trade_price: Optional[float] = None) -> str:
        """ Lee-Ready algorithm for trade classification.
        
        Rules:
        1. If trade price > mid price: BUY
        2. If trade price < mid price: SELL
        3. If trade price = mid price: use tick test
            - If current price > prev price: BUY
            - If current price < prev price: SELL
            - If current price = prev price: UNKNOWN

        Args:
            trade_price: The price of the trade.
            mid_price: The mid price of the order book.
            prev_trade_price: The price of the previous trade.

        Returns:
            'BUY', 'SELL', or 'UNKNOWN'
        """

        # Quote test
        if trade_price > mid_price:
            classification = 'BUY'
        elif trade_price < mid_price:
            classification = 'SELL'
        else:
            # Tick test for traders at mid
            if prev_trade_price is not None:
                if trade_price > prev_trade_price:
                    classification = 'BUY'
                elif trade_price < prev_trade_price:
                    classification = 'SELL'
                else:
                    classification = 'UNKNOWN'
            else:
                classification = 'UNKNOWN'

        # Store classification
        self.classification_history.append({
            'trade_price': trade_price,
            'mid_price': mid_price,
            'classification': classification,
            'algorithm': 'lee_ready'
        })

        return classification
    

    def tick_test(self, trade_price: float, prev_trade_price: float) -> str:
        """ Simple tick test for traders at mid.
        
        Rules:
        1. If current price > prev price: BUY
        2. If current price < prev price: SELL
        3. If current price = prev price: UNKNOWN

        Args:
            trade_price: The price of the trade.
            prev_trade_price: The price of the previous trade.

        Returns:
            'BUY', 'SELL', or 'UNKNOWN'
        """

        if trade_price > prev_trade_price:
            return 'BUY'
        elif trade_price < prev_trade_price:
            return 'SELL'
        else:
            return 'UNKNOWN'
        

    def quote_rule(self, trade_price: float, best_bid: float, best_ask: float) -> str:
        """ Quote rule for trade classification. Based on position relative to quotes. 
        
        Args:
            trade_price: The price of the trade.
            best_bid: The best bid price.
            best_ask: The best ask price.

        Returns:
            'BUY', 'SELL', or 'UNKNOWN'
        """

        mid = (best_bid + best_ask) / 2

        if trade_price > mid:
            return 'BUY'
        elif trade_price < mid:
            return 'SELL'
        else:
            return 'UNKNOWN'
        

    def bulk_volume_classification(self, trades: List[Dict]) -> pd.DataFrame:
        """
        Bulk volume classification algorithm. Considers volume distribution.

        Args:
            trades: List of trade dictionaries.

        Returns:
            DataFrame with trade classification.
        """

        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)

        # Calculate mid prices
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2

        # Initial classification using Lee-Ready algorithm
        df['lr_classification'] = df.apply(
            lambda row: self.lee_ready_algorithm(
                row['price'],
                row['mid_price'],
                df['price'].shift(1).loc[row.name] if row.name > 0 else None
            ),
            axis=1
        )

        # Volume weighted adjustments
        df['volume_weight'] = df['quantity'] / df['quantity'].rolling(window=10).mean()

        # Adjust classification for large trades
        df['final_classification'] = df['lr_classification']

        # Large trades (>2x average volume) at mid get special treatment
        large_at_mid = (df['volume_weight'] > 2) & (df['mid_price'] == df['price'])

        # Use order book imbalance for large trades at mid
        for idx in df[large_at_mid].index:
            if idx > 0:
                bid_vol = df.loc[idx, 'bid_volume_1']
                ask_vol = df.loc[idx, 'ask_volume_1']

                if bid_vol > ask_vol * 1.5:
                    df.loc[idx, 'final_classification'] = 'SELL' # Hit the bid
                elif ask_vol > bid_vol * 1.5:
                    df.loc[idx, 'final_classification'] = 'BUY' # Lift the offer

        return df
    

    def calculate_accuracy_metrics(self, true_labels: List[str]) -> Dict:
        """ Calculate classification accuracy if true labels are known.
        Useful for backtesting and validation.

        Args:
            true_labels: List of true trade directions ('BUY', 'SELL', 'UNKNOWN').

        Returns:
            Dictionary with accuracy metrics.
        """

        if len(true_labels) != len(self.classification_history):
            return {}
        
        correct = 0
        total = 0

        for true_label, classification in zip(true_labels, self.classification_history):
            if classification['classification'] != 'UNKNOWN':
                total += 1
                if classification['classification'] == true_label:
                    correct += 1

        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0

        return {
            'accuracy': accuracy,
            'total_classified': total,
            'unknown_count': len(self.classification_history) - total
        }
    

    def get_signed_volume(self) -> pd.Series:
        """ Calculate signed volume based on trade classification.
        
        Returns:
            Series with signed volume.
        """

        if not self.classification_history:
            return pd.Series()
        

        signed_volumes = []

        for record in self.classification_history:
            if record['classification'] == 'BUY':
                signed_volumes.append(record.get('quantity', 1))
            elif record['classification'] == 'SELL':
                signed_volumes.append(-record.get('quantity', 1))
            else:
                signed_volumes.append(0)

        return pd.Series(signed_volumes)
