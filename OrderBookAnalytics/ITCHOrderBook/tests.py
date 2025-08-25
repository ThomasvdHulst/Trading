""" This file contains the unit tests for the order book. """

# Import used libraries
import unittest
from data_strucs import Order, PriceLevel
from itch_parser import ITCHParser
from order_book import ITCHOrderBook


class TestOrder(unittest.TestCase):
    """ Test order operations. """

    def test_order_id(self):
        """ Test order ID. """
        order = Order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        self.assertEqual(order.order_id, 1)


    def test_hash(self):
        """ Test order hash. """
        order = Order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        self.assertEqual(hash(order), 1)


    def test_eq(self):
        """ Test order equality. """
        order1 = Order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order2 = Order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        self.assertEqual(order1, order2)


class TestPriceLevel(unittest.TestCase):
    """ Test price level operations. """

    def test_add_order(self):
        """ Test adding order to price level. """
        price_level = PriceLevel(price=100, side='BID')
        price_level.add_order(order_id=1, shares=100)
        self.assertEqual(price_level.total_volume, 100)


    def test_remove_order(self):
        """ Test removing order from price level. """
        price_level = PriceLevel(price=100, side='BID')
        price_level.add_order(order_id=1, shares=100)
        price_level.remove_order(order_id=1, shares=100)
        self.assertEqual(price_level.total_volume, 0)


    def test_get_queue_position(self):
        """ Test getting queue position of order. """
        price_level = PriceLevel(price=100, side='BID')
        price_level.add_order(order_id=1, shares=100)
        self.assertEqual(price_level.get_queue_position(order_id=1), 1)


class TestITCHParser(unittest.TestCase):
    """ Test ITCH parser operations. """

    def test_parse_message(self):
        """ Test parsing message. """
        parser = ITCHParser(10000)
        message = parser.create_add_order_message(order_id=1, side='BID', price=100, shares=100)
        result = parser.parse_message(message)
        self.assertEqual(result['type'], 'ADD_ORDER')


    def test_parse_execute_message(self):
        """ Test parsing execute message. """
        parser = ITCHParser(10000)
        message = parser.create_execute_message(order_id=1, shares=100, match_number=1)
        result = parser.parse_message(message)
        self.assertEqual(result['type'], 'ORDER_EXECUTED')


    def test_parse_cancel_message(self):
        """ Test parsing cancel message. """
        parser = ITCHParser(10000)
        message = parser.create_cancel_message(order_id=1, shares=100)
        result = parser.parse_message(message)
        self.assertEqual(result['type'], 'CANCEL_ORDER')


    def test_parse_delete_message(self):
        """ Test parsing delete message. """
        parser = ITCHParser(10000)
        message = parser.create_delete_message(order_id=1)
        result = parser.parse_message(message)
        self.assertEqual(result['type'], 'ORDER_DELETE')


    def test_get_parsing_stats(self):
        """ Test getting parsing statistics. """
        parser = ITCHParser(10000)
        parser.parse_message(parser.create_add_order_message(order_id=1, side='BID', price=100, shares=100))
        parser.parse_message(parser.create_execute_message(order_id=1, shares=100, match_number=1))
        parser.parse_message(parser.create_cancel_message(order_id=1, shares=100))
        parser.parse_message(parser.create_delete_message(order_id=1))

        stats = parser.get_parsing_stats()
        self.assertIn('messages_parsed', stats)
        self.assertIn('avg_parse_time_ns', stats)
        self.assertIn('p50_parse_time_ns', stats)
        self.assertIn('p99_parse_time_ns', stats)


class TestITCHOrderBook(unittest.TestCase):
    """ Test ITCH order book operations. """

    def test_add_order(self):
        """ Test adding order to order book. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        self.assertEqual(order_book.orders[1].order_id, 1)


    def test_execute_order(self):
        """ Test executing order. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.execute_order(order_id=1, executed_shares=50)
        self.assertEqual(order_book.orders[1].shares, 50)


    def test_cancel_order(self):
        """ Test cancelling order. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.cancel_order(order_id=1, cancelled_shares=50)
        self.assertEqual(order_book.orders[1].shares, 50)


    def test_delete_order(self):
        """ Test deleting order. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.delete_order(order_id=1)
        self.assertNotIn(1, order_book.orders)


    def test_get_best_bid(self):
        """ Test getting best bid. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='BID', price=90, shares=100)
        self.assertEqual(order_book.get_best_bid(), (100/100, 100))


    def test_get_best_ask(self):
        """ Test getting best ask. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='ASK', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='ASK', price=110, shares=100)
        self.assertEqual(order_book.get_best_ask(), (100/100, 100))


    def test_get_spread(self):
        """ Test getting spread. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='ASK', price=110, shares=100)
        self.assertEqual(order_book.get_spread(), 10/100)


    def test_get_mid_price(self):
        """ Test getting mid price. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='ASK', price=110, shares=100)
        self.assertEqual(order_book.get_mid_price(), 105/100)


    def test_get_queue_position(self):
        """ Test getting queue position. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='BID', price=100, shares=100)
        self.assertEqual(order_book.get_queue_position(order_id=1), 1)


    def test_get_book_depth(self):
        """ Test getting book depth. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=3, timestamp=1000000000, side='ASK', price=110, shares=100)
        order_book.add_order(order_id=4, timestamp=1000000000, side='ASK', price=110, shares=100)

        depth = order_book.get_book_depth(levels=2)
        self.assertEqual(depth['bids'], [(100/100, 200, 2)])
        self.assertEqual(depth['asks'], [(110/100, 200, 2)])
        self.assertEqual(depth['bid_levels'], 1)
        self.assertEqual(depth['ask_levels'], 1)
        self.assertEqual(depth['total_orders'], 4)


    def test_get_performance_stats(self):
        """ Test getting performance statistics. """
        order_book = ITCHOrderBook()
        order_book.add_order(order_id=1, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=2, timestamp=1000000000, side='BID', price=100, shares=100)
        order_book.add_order(order_id=3, timestamp=1000000000, side='ASK', price=110, shares=100)
        order_book.add_order(order_id=4, timestamp=1000000000, side='ASK', price=110, shares=100)

        stats = order_book.get_performance_stats()
        self.assertIn('total_messages', stats)
        self.assertIn('active_orders', stats)
        self.assertIn('bid_levels', stats)
        self.assertIn('ask_levels', stats)
        self.assertIn('total_volume_traded', stats)


if __name__ == "__main__":
    unittest.main()