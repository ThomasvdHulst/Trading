""" Main simulation runner demostrating the order book analytics system. """

# Import used libraries
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import OrderBook, EventProcessor
from src.data import MarketDataGenerator
from src.analytics import MicrostructureAnalyzer, ToxicityAnalyzer, TradeClassifier
from src.visualization import OrderBookVisualizer
from src.strategies import SimpleMarketMaker
from src.utils import PerformanceMonitor
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation():
    """ Run complete order book simulation with analytics. """
    
    print('Order Book Analytics Simulation!')

    # Initialize components
    order_book = OrderBook("TEST")
    processor = EventProcessor(order_book)
    generator = MarketDataGenerator(base_price=100.0)

    # Analytics
    micro_analyzer = MicrostructureAnalyzer()
    toxicity_analyzer = ToxicityAnalyzer(volume_bucket_size=500)
    trade_classifier = TradeClassifier()

    # Strategy
    market_maker = SimpleMarketMaker()

    # Performance monitoring
    perf_monitor = PerformanceMonitor()

    # Visualization
    visualizer = OrderBookVisualizer(figsize=(15, 10))

    print('')
    print("1. Generating market data")
    messages = generator.generate_random_walk(2000)
    print(f"Generated {len(messages)} messages")

    print('')
    print("2. Processing messages and calculating analytics")

    # Process messages
    for i, message in enumerate(messages):
        with perf_monitor.timer('message_processing'):
            processor.process_message(message)

        # Calculate analytics every 10 messages
        if i % 10 == 0:
            with perf_monitor.timer('analytics'):
                metrics = micro_analyzer.calculate_metrics(order_book)

                # Process trades for toxicity
                if message.msg_type.value == 'TRADE':
                    mid = order_book.get_mid_price()
                    if mid:
                        toxicity_analyzer.process_trade(
                            message.price,
                            message.quantity,
                            message.is_aggressor,
                            mid
                        )

                        # Trade classification
                        prev_price = messages[i-1].price if i > 0 else None
                        classification = trade_classifier.lee_ready_algorithm(
                            message.price, mid, prev_price
                        )

                # Market maker strategy
                bid_quote, ask_quote = market_maker.calculate_quotes(order_book)

                # Simulate fills (simplified)
                if bid_quote and message.msg_type.value == 'TRADE' and not message.is_aggressor:
                    if message.price <= bid_quote.price:
                        market_maker.on_trade(message.price, message.quantity, False)
                    
                if ask_quote and message.msg_type.value == 'TRADE' and message.is_aggressor:
                    if message.price >= ask_quote.price:
                        market_maker.on_trade(message.price, message.quantity, True)

                # Update PnL
                mid = order_book.get_mid_price()
                if mid:
                    market_maker.update_unrealized_pnl(mid)

        # Process indicator
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(messages)} messages")


    print('')
    print("3. Generating visualizations")

    # Create visualizations
    # 1. Final order book snapshot
    book_fig = visualizer.plot_book_snapshot(order_book, levels=15)
    book_fig.savefig('images/run_simulation/order_book_snapshot.png')

    # 2. Metrics history
    metrics_df = micro_analyzer.get_metrics_df()
    if not metrics_df.empty:
        metrics_fig = visualizer.plot_metrics_history(metrics_df)
        metrics_fig.savefig('images/run_simulation/metrics_history.png')

    # 3. Toxicity analysis
    toxicity_metrics = toxicity_analyzer.get_toxicity_metrics()
    toxicity_fig = visualizer.plot_toxicity_analysis(
        toxicity_metrics,
        list(toxicity_analyzer.volume_buckets)
    )
    toxicity_fig.savefig('images/run_simulation/toxicity_analysis.png')

    print('')
    print("4. Performance Report")
    perf_report = perf_monitor.get_report()

    for operation, stats in perf_report.items():
        if operation != 'overall':
            print(f"Operation: {operation}")
            print(f"Count: {stats.get('count', 0):,}")
            print(f"Avg latency: {stats.get('avg_ms', 0):.3f}ms")
            print(f"P99 latency: {stats.get('p99_ms', 0):.3f}ms")
            print('')

    if 'overall' in perf_report:
        overall = perf_report['overall']
        print("Overall:")
        print(f"Total operations: {overall['total_operations']:,}")
        print(f"Throughput: {overall['throughput_per_second']:,.0f} ops/sec")


    print('')
    print("5. Market Maker Strategy Performance")
    mm_metrics = market_maker.get_performance_metrics()

    print(f"Total PnL: ${mm_metrics.get('total_pnl', 0):.2f}")
    print(f"Realized PnL: ${mm_metrics.get('realized_pnl', 0):.2f}")
    print(f"Unrealized PnL: ${mm_metrics.get('unrealized_pnl', 0):.2f}")
    print(f"Number of trades: {mm_metrics.get('num_trades', 0)}")
    print(f"Current position: {mm_metrics.get('current_position', 0):.0f}")
    
    if 'sharpe_ratio' in mm_metrics:
        print(f"Sharpe ratio: {mm_metrics['sharpe_ratio']:.2f}")
    if 'max_drawdown' in mm_metrics:
        print(f"Max drawdown: {mm_metrics['max_drawdown']:.2%}")

    print('')
    print("6. Toxicity Metrics")
    print(f"VPIN: {toxicity_metrics.get('vpin', 0):.3f}")
    print(f"Kyle's Lambda: {toxicity_metrics.get('kyle_lambda', 0):.6f}")
    print(f"Mean Adverse Selection: {toxicity_metrics.get('mean_adverse_selection', 0):.4f}")
    print(f"Positive Selection Rate: {toxicity_metrics.get('positive_selection_rate', 0):.2%}")
    
    print('')
    print("7. Order Book Final State")
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    
    if best_bid and best_ask:
        print(f"Best Bid: ${best_bid[0]:.2f} x {best_bid[1]:.0f}")
        print(f"Best Ask: ${best_ask[0]:.2f} x {best_ask[1]:.0f}")
        print(f"Spread: ${order_book.get_spread():.3f}")
        print(f"Mid Price: ${order_book.get_mid_price():.2f}")
    
    depth = order_book.get_book_depth(5)
    print(f"Top 5 Bid Levels:")
    for price, qty in depth['bids']:
        print(f"${price:.2f}: {qty:.0f}")
    
    print(f"Top 5 Ask Levels:")
    for price, qty in depth['asks']:
        print(f"${price:.2f}: {qty:.0f}")

if __name__ == '__main__':
    run_simulation()