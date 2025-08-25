""" This file contains the message parser for ITCH messages."""

# Import used libraries
import struct
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional
from message_defs import ITCH_FORMATS


class ITCHParser:
    """ ITCH message parser. Uses struct for binary parsing and pre-allocated buffers. """

    def __init__(self, buffer_size: int = 1000000):
        """ Initialize parser with buffer size. 
        
        Args:
            buffer_size: Size of the buffer to use for parsing
        """

        # Pre-allocated numpy array for zero-copy message storage
        self.message_buffer = np.zeros(buffer_size, dtype=[ # buffer_size number of rows, and each row consists of:
            ('timestamp', 'u8'), # 8 bytes for the integer timestamp
            ('order_id', 'u8'), # 8 bytes for the integer order ID
            ('message_type', 'U1'), # 1 Unicode string for the message type (e.g. 'A' for Add Order)
            ('side', 'U3'), # 3 Unicode string for the side ('BID' or 'ASK')
            ('price', 'u4'), # 4 bytes for the integer price in cents
            ('shares', 'u4') # 4 bytes for the integer number of shares
        ])
        self.buffer_pos = 0

        # Message parsing statistics
        self.parse_times = deque(maxlen=10000)
        self.message_counts = defaultdict(int)


    def create_add_order_message(self, order_id: int, side: str, price: float, shares: int) -> bytes:
        """ 
        Create binary ITCH Add Order message.
        In production, this would come from the exchange.
        """

        # Convert price to cents
        price_int = int(price * 100)

        # Pack into binary format
        # Using simplified format (ITCH has more fields)
        message = struct.pack(
            '>cHQ6sQsI8sI',
            b'A', # Message type
            0, # Stock locate
            order_id, # Order ID / tracking number
            b'\x00' * 6, # Tracking number padding
            int(time.time_ns()), # Timestamp in nanoseconds
            b'B' if side == 'BID' else b'S', # Side
            shares, # Number of shares
            b'TEST    ', # Symbol, padded to 8 bytes
            price_int, # Price in cents
        )

        return message
    

    def create_execute_message(self, order_id: int, shares: int, match_number: int) -> bytes:
        """ Create binary Order Executed message """

        message = struct.pack(
            '>cHH6sQQIsQ',
            b'E', # Message type
            0, # Stock locate
            0, # Tracking number
            b'\x00' * 6, # Tracking number padding
            int(time.time_ns()), # Timestamp in nanoseconds
            order_id, # Order ID
            shares, # Number of shares
            b' ', # Printable flag
            match_number, # Match number
        )

        return message
    

    def create_cancel_message(self, order_id: int, shares: int) -> bytes:
        """ Create binary Cancel Order message """

        message = struct.pack(
            '>cHH6sQQI',
            b'X', # Message type
            0, # Stock locate
            0, # Tracking number
            b'\x00' * 6, # Tracking number padding
            int(time.time_ns()), # Timestamp in nanoseconds
            order_id, # Order ID
            shares # Number of shares
        )    

        return message
    

    def create_delete_message(self, order_id: int) -> bytes:
        """ Create binary Delete Order message """

        message = struct.pack(
            '>cHH6sQQ',
            b'D', # Message type
            0,
            0,
            b'\x00' * 6,
            int(time.time_ns()),
            order_id
        )

        return message
    

    def parse_message(self, raw_message: bytes) -> Optional[Dict]:
        """
        Parse binary ITCH message into dictionary.
        """

        start_time = time.perf_counter_ns()

        # Extract message type and update count
        message_type = raw_message[0:1]
        self.message_counts[message_type] += 1

        result = None

        try:
            if message_type == b'A':
                # Add Order
                # Skip the message type when unpacking
                unpacked = struct.unpack(ITCH_FORMATS[b'A'], raw_message[1:])
                result = {
                    'type': 'ADD_ORDER',
                    'timestamp': unpacked[3],
                    'order_id': unpacked[1],
                    'side': 'BID' if unpacked[4] == b'B' else 'ASK',
                    'shares': unpacked[5],
                    'symbol': unpacked[6].decode('ascii').strip(),
                    'price': unpacked[7], # In cents
                }

            elif message_type == b'E':
                # Order Executed
                unpacked = struct.unpack(ITCH_FORMATS[b'E'], raw_message[1:])

                result = {
                    'type': 'ORDER_EXECUTED',
                    'timestamp': unpacked[3],
                    'order_id': unpacked[4],
                    'executed_shares': unpacked[5],
                    'match_number': unpacked[7]
                }

            elif message_type == b'X':
                # Cancel Order
                unpacked = struct.unpack(ITCH_FORMATS[b'X'], raw_message[1:])
                result = {
                    'type': 'CANCEL_ORDER',
                    'timestamp': unpacked[3],
                    'order_id': unpacked[4],
                    'cancelled_shares': unpacked[5]
                }

            elif message_type == b'D':
                # Order Deleted
                unpacked = struct.unpack(ITCH_FORMATS[b'D'], raw_message[1:])
                result = {
                    'type': 'ORDER_DELETE',
                    'timestamp': unpacked[3],
                    'order_id': unpacked[4]
                }
        
        except struct.error as e:
            print(f"Error parsing message of type {message_type}: {e}")
            return None
        
        # Track parsing performance
        elapsed = time.perf_counter_ns() - start_time
        self.parse_times.append(elapsed)

        return result
    

    def get_parsing_stats(self) -> Dict:
        """ Get parsing performance statistics """

        if not self.parse_times:
            return {}
        
        times_ns = np.array(self.parse_times)

        return {
            'messages_parsed': sum(self.message_counts.values()),
            'avg_parse_time_ns': np.mean(times_ns),
            'p50_parse_time_ns': np.percentile(times_ns, 50),
            'p99_parse_time_ns': np.percentile(times_ns, 99),
            'message_type_counts': dict(self.message_counts)
        }
    

