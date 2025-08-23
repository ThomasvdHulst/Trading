""" This file processes messages and updates the order book accordingly. 
This is the bridge between raw messages and the order book state.
"""

from typing import List
from .order_book import OrderBook
from .message_parser import Message, MessageType
import time
import numpy as np


class EventProcessor:
    """
    Processes normalized messages and updates the order book state.
    """

    def __init__(self, order_book: OrderBook):
        """
        Args:
            order_book: The order book to update.
        """

        self.order_book = order_book
        self.processed_count = 0
        self.last_sequence = 0

        # Track processing performance
        self.processing_times = []

    
    def process_message(self, message: Message) -> bool:
        """ Process a single message and update the order book.
        
        Args:
            message: The message to process.

        Returns:
            True if the message was processed successfully, False otherwise.
        """

        start_time = time.perf_counter()

        try:
            if message.msg_type == MessageType.ADD_ORDER:
                self._handle_add_order(message)
            elif message.msg_type == MessageType.CANCEL_ORDER:
                self._handle_cancel_order(message)
            elif message.msg_type == MessageType.TRADE:
                self._handle_trade(message)
            elif message.msg_type == MessageType.CLEAR:
                self.order_book.clear()

            self.processed_count += 1

            # Track processing time
            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)

            return True
        
        except Exception as e:
            print(f"Error processing message: {e}")
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
            if self.process_message(message):
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

        self.order_book.execute_trade(
            message.price,
            message.quantity,
            message.is_aggressor,
        )


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
