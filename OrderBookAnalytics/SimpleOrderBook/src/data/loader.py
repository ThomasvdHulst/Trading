""" This file loads the data for various data formats. """

# Import used libraries
import pandas as pd
from typing import List
from ..core.message_parser import Message, MessageParser
from .generator import MarketDataGenerator


class DataLoader:
    """ Loads data from various formats. 
    Currently supports CSV and DataFrame inputs.
    """

    def __init__(self):
        self.parser = MessageParser()
        self.data = None


    def load_csv(self, file_path: str) -> List[Message]:
        """ Load messages from a CSV file.
        
        Args:
            file_path: Path to the CSV file.

        Returns:
            List of messages.
        """
        df = pd.read_csv(file_path)
        return self.load_from_dataframe(df)
    

    def load_from_dataframe(self, df: pd.DataFrame) -> List[Message]:
        """ Load messages from a DataFrame.
        
        Args:
            df: DataFrame containing the data.

        Returns:
            List of messages.
        """
        messages = []

        for _, row in df.iterrows():
            raw_message = row.to_dict()
            message = self.parser.parse(raw_message)

            if message:
                messages.append(message)

        self.data = df
        return messages
    
    def load_from_generator(self) -> List[Message]:
        """ Load messages from a generator. 
        
        Returns:
            List of messages.
        """

        generator = MarketDataGenerator()
        return generator.generate_random_walk(1000)
    

    def get_summary_statistics(self) -> dict:
        """ Get summary statistics of the loaded data.
        
        Returns:
            Dictionary containing the statistics.
        """

        if self.data is None:
            return {}
        
        return {
            'total_messages': len(self.data),
            'message_types': self.data['type'].value_counts().to_dict(),
            'time_range': (self.data['timestamp'].min(), self.data['timestamp'].max()),
            'unique_orders': self.data['order_id'].nunique(),
            'price_range': (self.data['price'].min(), self.data['price'].max()),
            'total_volume': self.data['quantity'].sum(),
        }
        
    