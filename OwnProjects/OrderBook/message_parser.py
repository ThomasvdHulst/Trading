""" This file contains the code around the messages and how they are parsed to the order book. """

# Import used libraries
from order_book import OrderBook
from enum import Enum
from typing import Optional
import time


class MessageType(Enum):
    """ This class contains the different message types. """

    ADD_ORDER = "ADD"
    CANCEL_ORDER = "CANCEL"
    TRADE = "TRADE"


class Message:
    """ This class creates the message object for the different message types. """

    def __init__(self, 
                timestamp: float,
                msg_type: MessageType, # The type of the message, see MessageType class
                order_id: Optional[int] = None, # The order ID
                side: Optional[str] = None, # The side, 'BID' or 'ASK'
                price: Optional[float] = None,
                quantity: Optional[float] = None,
                is_aggressor: Optional[bool] = None): # For trade messages

        self.timestamp = timestamp
        self.msg_type = msg_type
        self.order_id = order_id
        self.side = side
        self.price = price
        self.quantity = quantity
        self.is_aggressor = is_aggressor


    @classmethod
    def create_add_order(cls, order_id: int, side: str, price: float, quantity: float) -> 'Message':
        """ This method creates an add order message. 
        
        Args:
            order_id: The order ID
            side: The side, 'BID' or 'ASK'
            price: The price
            quantity: The quantity

        Returns:
            Message: The message object
        """

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
        """ This method creates a cancel order message. 
        
        Args:
            order_id: The order ID
            side: The side, 'BID' or 'ASK'
            price: The price
            quantity: The quantity

        Returns:
            Message: The message object
        """

        return cls(
            timestamp = time.time(),
            msg_type = MessageType.CANCEL_ORDER,
            order_id = order_id,
            side = side,
            price = price,
            quantity = quantity,
        )

    
    @classmethod
    def create_trade(cls, price: float, quantity: float, is_aggressor: bool) -> 'Message':
        """ This method creates a trade message. 
        
        Args:
            price: The price
            quantity: The quantity
            is_aggressor: Whether the trade is a buy or sell

        Returns:
            Message: The message object
        """

        return cls(
            timestamp = time.time(),
            msg_type = MessageType.TRADE,
            price = price,
            quantity = quantity,
            is_aggressor = is_aggressor,
        )

    
class MessageParser:
    """ This class parses the messages and creates the message object. """

    def __init__(self):
        self.message_count = 0
        self.error_count = 0

    
    def parse(self, message: dict) -> Optional[Message]:
        """ This method parses the message and creates the message object. 
        
        Args:
            message: The message to parse

        Returns:
            Message: The message object
        """

        try:
            msg_type = MessageType(message.get('msg_type'))

            message = Message(
                timestamp = message.get('timestamp'),
                msg_type = msg_type,
                order_id = message.get('order_id'),
                side = message.get('side'),
                price = message.get('price'),
                quantity = message.get('quantity'),
                is_aggressor = message.get('is_aggressor'),
            )

            self.message_count += 1
            return message

        except Exception as e:
            self.error_count += 1
            print(f"Error parsing message: {e}")
            return None

        
    def get_stats(self) -> dict:
        """ This method returns the statistics of the message parser. 
        
        Returns:
            dict: The statistics of the message parser
        """

        return {
            'message_count': self.message_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / (self.message_count+self.error_count) if self.message_count > 0 else 0,
        }



    