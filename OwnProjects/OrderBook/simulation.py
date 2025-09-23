""" This file contains the simulation of the order book. """

# Import used libraries
from order_book import OrderBook
from data import DataGenerator
from event_processor import EventProcessor
from metrics import Metrics
from visualization import Visualizer


def run_simulation():
    """ This function contains the simulation of the order book. """
    
    print('Order book simulation!')

    # Initialize components
    order_book = OrderBook()
    event_processor = EventProcessor(order_book)
    generator = DataGenerator()
    metrics = Metrics()
    visualizer = Visualizer()

    # Run simulation
    messages = generator.generate_random_walk(10000)
    print('All messages are generated!')
    print('Processing messages...')

    # If you want to process messages in batches, you can use the following code:
    #event_processor.process_batch(messages)

    # Instead, we process messages one by one to obtain snapshots of the order book
    for i, message in enumerate(messages):

        # Process the message
        event_processor.handle_message(message)

        # Update the realized volatility
        metrics.calculate_realized_volatility(order_book)

        # Calculate metrics every X messages
        if (i+1) % 10 == 0:
            metrics_data = metrics.calc_metrics(order_book)
            
            if (i+1) % 100 == 0:
                print(f'Processed {i} messages')
                for metric, value in metrics_data.items():
                    print(f'{metric}: {value}')

                print(f'Realized volatility: {metrics.realized_volatility[-1]}')
                print('--------------------------------')


    # Generate visualizations
    print('Generating visualizations...')
    
    # Print summary statistics
    visualizer.print_summary_stats(metrics, order_book)
    
    # Create individual plots
    visualizer.plot_price_evolution(metrics)
    visualizer.plot_order_book_depth(order_book)
    visualizer.plot_realized_volatility(metrics)
    
    print('Visualization complete!')


if __name__ == '__main__':
    run_simulation()
