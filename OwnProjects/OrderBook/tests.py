""" This file contains the unit tests for the order book. """

# Import used libraries
import unittest
from order_book import OrderBook
from message_parser import Message, MessageParser, MessageType
from event_processor import EventProcessor
from data import DataGenerator


class TestOrderBook(unittest.TestCase):
    """ Test order book operations. """

    def setUp(self):
        """ Set up test fixtures. """
        self.order_book = OrderBook()

    
    def test_add_order(self):
        """ Test adding order to order book. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_ask(101, 100)

        self.assertEqual(self.order_book.bids[100].quantity, 100)
        self.assertEqual(self.order_book.asks[101].quantity, 100)

    
    def test_remove_order(self):
        """ Test removing order from order book. """
        self.order_book.add_bid(100, 100)
        self.order_book.remove_bid(100, 50)

        self.assertEqual(self.order_book.bids[100].quantity, 50)

    
    def test_spread_calculation(self): 
        """ Test spread calculation. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_ask(101, 100)

        self.assertEqual(self.order_book.get_spread(), 1)

    
    def test_mid_price_calculation(self):
        """ Test mid price calculation. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_ask(101, 100)

        self.assertEqual(self.order_book.get_mid_price(), 100.5)

    
    def test_get_best_bid(self):
        """ Test getting best bid. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_bid(101, 100)
        self.assertEqual(self.order_book.get_best_bid(), (101, 100))

    
    def test_get_best_ask(self):
        """ Test getting best ask. """
        self.order_book.add_ask(100, 100)
        self.order_book.add_ask(101, 100)
        self.assertEqual(self.order_book.get_best_ask(), (100, 100))

    
    def test_book_depth(self):
        """ Test getting book depth. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_bid(101, 100)
        self.order_book.add_ask(102, 100)
        self.order_book.add_ask(103, 100)
        self.assertEqual(self.order_book.get_book_depth(2), {'bids': [(101, 100), (100, 100)], 'asks': [(102, 100), (103, 100)]})
        
    
    def test_vwap_calculation(self):
        """ Test VWAP calculation. """
        self.order_book.add_bid(100, 100)
        self.order_book.add_bid(101, 100)
        self.assertEqual(self.order_book.get_vwap(side='sell', target_qty=200), 100.5)
        


class TestMessage(unittest.TestCase):
    """ Test message creation. """


    def test_create_add_order(self):
        """ Test creating add order message. """
        msg = Message.create_add_order(1, 'BID', 100, 100)
        self.assertEqual(msg.msg_type, MessageType.ADD_ORDER)
        self.assertEqual(msg.order_id, 1)
        self.assertEqual(msg.side, 'BID')
        self.assertEqual(msg.price, 100)
        self.assertEqual(msg.quantity, 100)

    
    def test_create_cancel_order(self):
        """ Test creating cancel order message. """
        msg = Message.create_cancel_order(1, 'BID', 100, 100)
        self.assertEqual(msg.msg_type, MessageType.CANCEL_ORDER)
        self.assertEqual(msg.order_id, 1)
        self.assertEqual(msg.side, 'BID')
        self.assertEqual(msg.price, 100)
        self.assertEqual(msg.quantity, 100)
        
    
    def test_create_trade(self):
        """ Test creating trade message. """
        msg = Message.create_trade(100, 100, True)
        self.assertEqual(msg.msg_type, MessageType.TRADE)
        self.assertEqual(msg.price, 100)
        self.assertEqual(msg.quantity, 100)
        self.assertEqual(msg.is_aggressor, True)

    
class TestMessageParser(unittest.TestCase):
    """ Test message parser. """

    def setUp(self):
        """ Set up test fixtures. """
        self.message_parser = MessageParser()
        
        
    def test_parse_message(self):
        """ Test parsing message. """
        msg = {'msg_type': 'ADD', 'order_id': 1, 'side': 'BID', 'price': 100, 'quantity': 100}
        parsed_msg = self.message_parser.parse(msg)
        self.assertEqual(parsed_msg.msg_type, MessageType.ADD_ORDER)
        self.assertEqual(parsed_msg.order_id, 1)
        self.assertEqual(parsed_msg.side, 'BID')
        self.assertEqual(parsed_msg.price, 100)
        self.assertEqual(parsed_msg.quantity, 100)


    def test_stats(self):
        """ Test getting stats. """
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 1, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 2, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 3, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 4, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 5, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 6, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.message_parser.parse({'msg_type': 'ADD', 'order_id': 7, 'side': 'BID', 'price': 100, 'quantity': 100})
        self.assertEqual(self.message_parser.get_stats(), {'message_count': 7, 'error_count': 0, 'error_rate': 0})


class TestEventProcessor(unittest.TestCase):
    """ Test event processor. """

    def setUp(self):
        """ Set up test fixtures. """
        self.event_processor = EventProcessor(OrderBook())


    def test_process_message(self):
        """ Test processing message. """
        msg = Message.create_add_order(1, 'BID', 100, 100)
        self.assertTrue(self.event_processor.handle_message(msg))


    def test_process_batch(self):
        """ Test processing batch. """
        msg1 = Message.create_add_order(1, 'BID', 100, 100)
        msg2 = Message.create_add_order(2, 'BID', 100, 100)
        msg3 = Message.create_add_order(3, 'BID', 100, 100)
        msg4 = Message.create_cancel_order(4, 'BID', 100, 100)
        msg5 = Message.create_trade(100, 100, True)
        self.assertTrue(self.event_processor.process_batch([msg1, msg2, msg3, msg4, msg5]))


class TestDataGenerator(unittest.TestCase):
    """ Test data generator. """

    def setUp(self):
        """ Set up test fixtures. """
        self.data_generator = DataGenerator()


    def test_generate_opening_book(self):
        """ Test generating opening book. """
        self.assertEqual(len(self.data_generator.generate_opening_book()), 20)


    def test_generate_random_walk(self):
        """ Test generating random walk. """
        self.assertEqual(len(self.data_generator.generate_random_walk(1000)), 1000 + 20)


if __name__ == '__main__':
    unittest.main()