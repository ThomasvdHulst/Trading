""" This file contains the message parser for normalized exchange messages. This
simulates the type of messages that would be received from a real exchange (although simplified). """

# Import used libraries
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time


class MessageType(Enum):
    """ Types of messages in our normalized format. """

    ADD_ORDER = "ADD"
    CANCEL_ORDER = "CANCEL"
    MODIFY_ORDER = "MODIFY"
    TRADE = "TRADE"
    CLEAR = "CLEAR"


@dataclass
class Message:
    """
    Normalized message format that could come from any exchange.
    In production, this would be parsed from for example ITCH or FIX messages.
    """

    timestamp: float
    msg_type: MessageType
    order_id: Optional[int] = None
    side: Optional[str] = None # 'BID' or 'ASK'
    price: Optional[float] = None
    quantity: Optional[float] = None
    old_quantity: Optional[float] = None # For modify messages
    is_aggressor: Optional[bool] = None # For trade messages

    @classmethod
    def create_add_order(cls, order_id: int, side: str, price: float, quantity: float) -> 'Message':
        """ Factory method for add order messages. """

        return cls(
            timestamp = time.time(),
            msg_type = MessageType.ADD_ORDER,
            order_id = order_id,
            side = side,
            price = price,
            quantity = quantity,
        )
    

    @classmethod
    def create_cancel_order(cls, order_id: int, side: str, price: float, quantity: float) -> 'Message':
        """ Factory method for cancel order messages. """

        return cls(
            timestamp = time.time(),
            msg_type = MessageType.CANCEL_ORDER,
            order_id = order_id,
            side = side,
            price = price,
            quantity = quantity,
        )
    

    @classmethod
    def create_trade(cls, price: float, quantity: float, is_buy_aggressor: bool) -> 'Message':
        """ Factory method for trade messages. """

        return cls(
            timestamp = time.time(),
            msg_type = MessageType.TRADE,
            price = price,
            quantity = quantity,
            is_aggressor = is_buy_aggressor,
        )
    
    
class MessageParser:
    """
    Parses and validates incoming messages.
    In production, this would handle binary protocols and error correction.
    """

    def __init__(self):
        self.message_count = 0
        self.error_count = 0

    
    def parse(self, raw_message: dict) -> Optional[Message]:
        """
        Parse a raw message dictionary into a message object.
        In reality, this would parse binary data from the exchange.
        """

        try:
            msg_type = MessageType(raw_message.get('type'))

            message = Message(
                timestamp = raw_message.get('timestamp', time.time()),
                msg_type = msg_type,
                order_id = raw_message.get('order_id'),
                side = raw_message.get('side'),
                price = raw_message.get('price'),
                quantity = raw_message.get('quantity'),
                is_aggressor = raw_message.get('is_aggressor'),
            )

            self.message_count += 1
            return message
        
        except (KeyError, ValueError) as e:
            self.error_count += 1
            print(f"Error parsing message: {e}")
            return None
        

    def get_stats(self) -> dict:
        """ Get parser statistics. """

        return {
            'messages_processed': self.message_count,
            'errors': self.error_count,
            'error_rate': self.error_count / self.message_count if self.message_count > 0 else 0
        }
        