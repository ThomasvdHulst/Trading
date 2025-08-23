""" This file contains the unit tests for the core order book functionality. """

# Import used libraries
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import OrderBook, Message, MessageType, EventProcessor

class TestOrderBook(unittest.TestCase):
    """ Test order book operations. """

    def setUp(self):
        """ Set up test fixtures. """
        self.order_book = OrderBook("TEST")

    def test_add_order(self):
        """ Test adding orders to the order book. """
        self.order_book.add_bid(99.5, 100)
        self.order_book.add_ask(100.5, 100)

        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()

        self.assertEqual(best_bid[0], 99.5)
        self.assertEqual(best_bid[1], 100)
        self.assertEqual(best_ask[0], 100.5)
        self.assertEqual(best_ask[1], 100)

    
    def test_remove_orders(self):
        """ Test removing orders from the order book. """

        self.order_book.add_bid(99.5, 100)
        self.order_book.remove_bid(99.5, 50)

        best_bid = self.order_book.get_best_bid()
        self.assertEqual(best_bid[1], 50)

        self.order_book.remove_bid(99.5, 50)
        best_bid = self.order_book.get_best_bid()
        self.assertIsNone(best_bid)


    def test_spread_calculation(self):
        """ Test spread and mid-price calculation. """

        self.order_book.add_bid(99.5, 100)
        self.order_book.add_ask(100.5, 100)

        spread = self.order_book.get_spread()
        mid = self.order_book.get_mid_price()

        self.assertEqual(spread, 1.0)
        self.assertEqual(mid, 100.0)


    def test_book_depth(self):
        """ Test book depth retrieval. """

        # Add multiple levels
        for i in range(5):
            self.order_book.add_bid(99.5 - i * 0.1, 100 * (i + 1))
            self.order_book.add_ask(100.5 + i * 0.1, 100 * (i + 1))

        depth = self.order_book.get_book_depth(3)

        self.assertEqual(len(depth['bids']), 3)
        self.assertEqual(len(depth['asks']), 3)
        self.assertEqual(depth['bids'][0][0], 99.5) # Best bid
        self.assertEqual(depth['asks'][0][0], 100.5) # Best ask


    def test_vwap_calculation(self):
        """ Test VWAP calculation. """
        
        self.order_book.add_ask(100.0, 50)
        self.order_book.add_ask(100.1, 100)
        self.order_book.add_ask(100.2, 150)

        vwap = self.order_book.get_vwap('buy', 300)

        # VWAP = (100*50 + 100.1*100 + 100.2*150) / (50 + 100 + 150)
        expected_vwap = (100*50 + 100.1*100 + 100.2*150) / (50 + 100 + 150)

        self.assertAlmostEqual(vwap, expected_vwap, places=4)


class TestMessageProcessing(unittest.TestCase):
    """ Test message parsing and processing. """

    def setUp(self):
        """ Set up test fixtures. """
        self.order_book = OrderBook("TEST")
        self.processor = EventProcessor(self.order_book)


    def test_process_add_order(self):
        """ Test processing an add order message. """
        msg = Message.create_add_order(1, 'BID', 99.5, 100)
        success = self.processor.process_message(msg)

        self.assertTrue(success)
        self.assertEqual(self.processor.processed_count, 1)

        best_bid = self.order_book.get_best_bid()
        self.assertEqual(best_bid[0], 99.5)


    def test_process_trade(self):
        """ Test processing trade message """

        # Add liquidity first
        self.order_book.add_ask(100.0, 100)

        # Process trade
        msg = Message.create_trade(100.0, 50, is_buy_aggressor=True)
        success = self.processor.process_message(msg)

        self.assertTrue(success)
        self.assertEqual(len(self.order_book.trade_history), 1)
        self.assertEqual(self.order_book.last_trade_price, 100.0)


    def test_process_batch(self):
        """ Test processing a batch of messages. """

        messages = [
            Message.create_add_order(1, 'BID', 99.5, 100),
            Message.create_add_order(2, 'ASK', 100.5, 100),
            Message.create_trade(100.0, 50, True)
        ]

        processed = self.processor.process_batch(messages)

        self.assertEqual(processed, 3)
        self.assertEqual(self.processor.processed_count, 3)


if __name__ == '__main__':
    unittest.main()