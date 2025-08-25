import struct
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from enum import Enum
import random
import heapq
from data_strucs import Order, PriceLevel
from message_defs import ITCHMessageType, ITCH_FORMATS
from itch_parser import ITCHParser
from order_book import ITCHOrderBook
from itch_simulator import ITCHSimulator


def main():
    """ Main function, performance test of the ITCH order book. """

    print("ITCH Order Book Performance Test!")

    # Initialize components
    book = ITCHOrderBook("TEST")
    parser = ITCHParser()
    simulator = ITCHSimulator()

    # Test different message volumes
    test_sizes = [100, 1000, 10000]

    for n_messages in test_sizes:
        print(f"\nTesting with {n_messages:,} messages:")

        # Generate messages
        print(f"Generating {n_messages:,} messages...")
        start_time = time.perf_counter()
        messages = simulator.generate_messages(n_messages)
        gen_time = time.perf_counter() - start_time
        print(f"Generated in {gen_time:.3f} seconds")

        # Reset book
        book = ITCHOrderBook("TEST")
        parser = ITCHParser()

        # Process messages
        print("Processing messages...")
        start_time = time.perf_counter()

        successful = 0
        failed = 0

        for msg in messages:
            # Parse binary message
            parsed = parser.parse_message(msg)

            if parsed:
                # Process based on type
                if parsed['type'] == 'ADD_ORDER':
                    success = book.add_order(
                        parsed['order_id'],
                        parsed['timestamp'],
                        parsed['side'],
                        parsed['price'],
                        parsed['shares']
                    )
                elif parsed['type'] == 'ORDER_EXECUTED':
                    success = book.execute_order(
                        parsed['order_id'],
                        parsed['executed_shares']
                    )
                elif parsed['type'] == 'CANCEL_ORDER':
                    success = book.cancel_order(
                        parsed['order_id'],
                        parsed['cancelled_shares']
                    )
                elif parsed['type'] == 'ORDER_DELETE':
                    success = book.delete_order(parsed['order_id'])
                else:
                    success = False


                if success:
                    successful += 1
                else:
                    failed += 1
        
        process_time = time.perf_counter() - start_time

        # Calculate throughput
        throughput = n_messages / process_time

        # Get final book state
        depth = book.get_book_depth(levels=5)
        best_bid = book.get_best_bid()
        best_ask = book.get_best_ask()
        spread = book.get_spread()

        # Print results
        print(f"\nResults:")
        print(f"Processing time: {process_time:.3f} seconds")
        print(f"Throughput: {throughput:.2f} messages/second")
        print(f"Successful operations: {successful:,}")
        print(f"Failed operations: {failed:,}")

        print(f"\nParser statistics:")
        parser_stats = parser.get_parsing_stats()
        print(f"Avg parse time: {parser_stats['avg_parse_time_ns']:.0f} ns")
        print(f"50th percentile: {parser_stats['p50_parse_time_ns']:.0f} ns")
        print(f"99th percentile: {parser_stats['p99_parse_time_ns']:.0f} ns")
        print(f"Message type counts: {parser_stats['message_type_counts']}")

        print(f"\nBook statistics:")
        book_stats = book.get_performance_stats()
        print(f"Active orders: {book_stats['active_orders']:,}")
        print(f"Bid levels: {book_stats['bid_levels']:,}")
        print(f"Ask levels: {book_stats['ask_levels']:,}")
        print(f"Total volume traded: {book_stats['total_volume_traded']:.2f}")

        # Print operation latencies
        print(f"\n Operation latencies:")
        for op in ['add_order', 'execute_order', 'cancel_order', 'delete_order']:
            avg_key = f'{op}_avg_ns'
            p99_key = f'{op}_p99_ns'
            if avg_key in book_stats:
                print(f"{op} - Avg: {book_stats[avg_key]:.0f} ns, 99th: {book_stats[p99_key]:.0f} ns")

        print(f"\nFinal book state:")
        if best_bid and best_ask:
            print(f"Best bid: {best_bid[0]:.2f} @ {best_bid[1]:,} shares")
            print(f"Best ask: {best_ask[0]:.2f} @ {best_ask[1]:,} shares")
            print(f"Spread: {spread:.2f}")
            print(f"Mid price: {book.get_mid_price():.2f}")

        print(f"\nTop 5 levels:")
        print(f"Bids:")
        for price, volume, count in depth['bids']:
            print(f"  {price:.2f} @ {volume:,} shares ({count:,} orders)")
        print(f"Asks:")
        for price, volume, count in depth['asks']:
            print(f"  {price:.2f} @ {volume:,} shares ({count:,} orders)")
            

            
    # Queue position test
    print("\nQueue position test:")
    # Add some orders and check queue position
    test_book = ITCHOrderBook("TEST")
    order_ids = []

    # Add 5 orders at the same price
    for i in range(5):
        order_id = 9000000 + i
        test_book.add_order(order_id, time.time_ns(), 'BID', 9999, 100)
        order_ids.append(order_id)

    print("\nQueue positions for orders at $99.99:")
    for order_id in order_ids:
        position = test_book.get_queue_position(order_id)
        print(f"Order {order_id}: {'Front' if position == 1 else f'{position}th'} in queue")

    # Execute first order partially
    test_book.execute_order(order_ids[0], 50)
    print(f"\nAfter partial execution of order {order_ids[0]}:")
    for order_id in order_ids:
        position = test_book.get_queue_position(order_id)
        print(f"Order {order_id}: {'Front' if position == 1 else f'{position}th'} in queue")

    print("Test complete!")
    

if __name__ == "__main__":
    main()