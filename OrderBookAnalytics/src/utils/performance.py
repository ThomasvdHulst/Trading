""" This file contains monitoring utilities for the order book system.
Tracks latencies and throughput.
"""

# Import used libraries
import time
import numpy as np
from typing import Dict, List
from contextlib import contextmanager
from dataclasses import dataclass, field
import random
from ..core.order_book import OrderBook

@dataclass
class PerformanceStats:
    """ Container for performance metrics. """
    operation: str
    count: int = 0
    total_time: float = 0
    min_time: float = float('inf')
    max_time: float = 0
    times: List[float] = field(default_factory=list)

    def add_measurement(self, elapsed: float) -> None:
        """ Add a timing measurement. 
        
        Args:
            elapsed: Time taken for the operation.
        """
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    def get_stats(self) -> Dict:
        """ Get summary statistics. 
        
        Returns:
            Dictionary containing performance metrics.
        """
        if self.count == 0:
            return {}
        
        times_ms = np.array(self.times) * 1000 # Convert to milliseconds

        return {
            'operation': self.operation,
            'count': self.count,
            'avg_ms': np.mean(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'p50_ms': np.percentile(times_ms, 50),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'total_seconds': self.total_time
        }
    

class PerformanceMonitor:
    """ Monitors performance of order book operations. """

    def __init__(self):
        self.stats: Dict[str, PerformanceStats] = {}
        self.enabled = True


    @contextmanager
    def timer(self, operation: str):
        """ Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed.
        """

        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start

            if operation not in self.stats:
                self.stats[operation] = PerformanceStats(operation)
            self.stats[operation].add_measurement(elapsed)

  
    def get_report(self) -> Dict:
        """ Get performance report. 
        
        Returns:
            Dictionary containing performance metrics.
        """
        
        report = {}

        for operation, stats in self.stats.items():
            report[operation] = stats.get_stats()

        # Calculate throughput
        total_time = sum(s.total_time for s in self.stats.values())
        total_operations = sum(s.count for s in self.stats.values())

        if total_time > 0:
            report['overall'] = {
                'total_operations': total_operations,
                'total_time_seconds': total_time,
                'throughput_per_second': total_operations / total_time
            }

        return report
    

    def reset(self) -> None:
        """ Reset all statistics. """
        self.stats.clear()


    def benchmark_order_book_operations(self, order_book: OrderBook, n_operations: int = 10000) -> Dict:
        """ Benchmark standard order book operations. 
        
        Args:
            order_book: Order book instance.
            n_operations: Number of operations to benchmark.

        Returns:
            Dictionary containing performance metrics.
        """

        # Benchmark add operations
        with self.timer('add_order'):
            for _ in range(n_operations):
                price = 100 + random.uniform(-1, 1)
                quantity = random.uniform(10, 100)
                side = 'BID' if random.random() < 0.5 else 'ASK'

                if side == 'BID':
                    order_book.add_bid(price, quantity)
                else:
                    order_book.add_ask(price, quantity)

        # Benchmark best price lookups
        with self.timer('get_best_price'):
            for _ in range(n_operations):
                order_book.get_best_bid()
                order_book.get_best_ask()
                
        # Benchmark spread calculations
        with self.timer('calculate_spread'):
            for _ in range(n_operations):
                order_book.get_spread()
                order_book.get_mid_price()

        # Benchmark depth queries
        with self.timer('get_depth'):
            for _ in range(n_operations // 10): # Little less frequent
                order_book.get_book_depth(10)

        return self.get_report()

