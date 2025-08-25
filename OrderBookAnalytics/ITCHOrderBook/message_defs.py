""" This file contains the definitions for the messages that are sent to the order book. """

# Import used libraries
from enum import Enum

class ITCHMessageType(Enum):
    """NASDAQ ITCH 5.0 message types"""
    SYSTEM = b'S' # System event message
    STOCK_DIRECTORY = b'R' # Stock directory message
    TRADING_ACTION = b'H' # Trading action message
    ADD_ORDER = b'A' # Add order message
    ADD_ORDER_MPID = b'F' # Add order message with MPID (market participant ID visible)
    ORDER_EXECUTED = b'E' # Order executed message
    ORDER_EXECUTED_PRICE = b'C' # Order executed price message
    ORDER_CANCEL = b'X' # Order cancel message
    ORDER_DELETE = b'D' # Order delete message
    ORDER_REPLACE = b'U' # Order replace message
    TRADE = b'P' # Trade message
    CROSS_TRADE = b'Q' # Cross trade message


# Message format specifications (struct format strings)
ITCH_FORMATS = {
    b'A': '>HQ6sQsI8sI',      # Add Order: 36 bytes
    b'D': '>HH6sQQ',           # Order Delete: 20 bytes
    b'E': '>HH6sQQIsQ',        # Order Executed: 31 bytes
    b'X': '>HH6sQQI',          # Order Cancel: 23 bytes
    b'U': '>HH6sQQQII',        # Order Replace: 35 bytes
    b'P': '>HH6sQsQI8sIQ',     # Trade (Non-Cross): 44 bytes
}
