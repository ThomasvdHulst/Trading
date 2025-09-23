""" This file contains the event processor for the order book. 
It processes the messages and updates the order book accordingly.
"""

# Import used libraries
from message_parser import Message, MessageParser, MessageType
from order_book import OrderBook
import time
from typing import List
import numpy as np


class EventProcessor:
    """ This class processes the messages and updates the order book accordingly. """
    
    def __init__(self, order_book: OrderBook):
        """ Initialize the event processor.

        Args:
            order_book: The order book to update.
        """

        self.order_book = order_book
        self.processed_count = 0

        # Track processing performance
        self.processing_times = []

    
    def handle_message(self, message: Message) -> bool:
        """ Handle a single message and update the order book.

        Args:
            message: The message to handle.

        Returns:
            True if the message was handled successfully, False otherwise.
        """

        # Start timer
        start_time = time.perf_counter()

        try:
            if message.msg_type == MessageType.ADD_ORDER:
                self._handle_add_order(message)
            elif message.msg_type == MessageType.CANCEL_ORDER:
                self._handle_cancel_order(message)
            elif message.msg_type == MessageType.TRADE:
                self._handle_trade(message)

            self.processed_count += 1

            # Track processing time
            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)
            
            return True

        except Exception as e:
            print(f"Error handling message: {e}")
            return False


    def process_batch(self, messages: List[Message]) -> int:
        """ Process a batch of messages.
        
        Args:
            messages: The list of messages to process.
            
        Returns:
            The number of messages processed.
        """

        success_count = 0

        for message in messages:
            if self.handle_message(message):
                success_count += 1

        return success_count


    def _handle_add_order(self, message: Message) -> None:
        """ Handle an add order message.
        
        Args:
            message: The add order message to process.
        """
        
        if message.side == 'BID':
            self.order_book.add_bid(message.price, message.quantity)
        elif message.side == 'ASK':
            self.order_book.add_ask(message.price, message.quantity)


    def _handle_cancel_order(self, message: Message) -> None:
        """ Handle a cancel order message.

        Args:
            message: The cancel order message to process.
        """

        if message.side == 'BID':
            self.order_book.remove_bid(message.price, message.quantity)
        elif message.side == 'ASK':
            self.order_book.remove_ask(message.price, message.quantity)

    
    def _handle_trade(self, message: Message) -> None:
        """ Handle a trade message.
        
        Args:
            message: The trade message to process.
        """
        
        self.order_book.execute_trade(message.price, message.quantity, message.is_aggressor)


    def get_performance_stats(self) -> dict:
        """ Get performance statistics. 
        
        Returns:
            A dictionary containing performance statistics.
        """

        if not self.processing_times:
            return {}

        time_us = np.array(self.processing_times) * 1e6 # Convert to microseconds
        
        return {
            'messages_processed': self.processed_count,
            'avg_latency_us': np.mean(time_us),
            'p50_latency_us': np.percentile(time_us, 50),
            'p99_latency_us': np.percentile(time_us, 99),
            'max_latency_us': np.max(time_us),
        }
