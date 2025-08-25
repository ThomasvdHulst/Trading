import sys
import os

# Add the parent directory to Python path so we can import config and order_book
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import Config
from order_book import OrderBook, Side


def test_basic_order_operations():
    """Test basic order book operations: add, cancel, modify"""
    print("=== Testing Basic Order Operations ===")
    
    config = Config()
    book = OrderBook(config)
    
    # Test adding orders
    print("\n1. Adding orders...")
    buy_id1 = book.add_order(Side.BUY, 99.50, 100, "trader1")
    buy_id2 = book.add_order(Side.BUY, 99.75, 200, "trader2")
    sell_id1 = book.add_order(Side.SELL, 100.25, 150, "trader3")
    sell_id2 = book.add_order(Side.SELL, 100.50, 100, "trader4")
    
    print(f"Added buy orders: {buy_id1}, {buy_id2}")
    print(f"Added sell orders: {sell_id1}, {sell_id2}")
    print(f"Best bid: ${book.best_bid}, Best ask: ${book.best_ask}")
    print(f"Mid price: ${book.mid_price:.2f}, Spread: ${book.spread:.2f}")
    
    # Test canceling an order
    print("\n2. Canceling an order...")
    success = book.cancel_order(buy_id1)
    print(f"Canceled order {buy_id1}: {success}")
    print(f"Best bid: ${book.best_bid}, Best ask: ${book.best_ask}")
    
    # Test modifying an order
    print("\n3. Modifying an order...")
    success = book.modify_order(sell_id1, 75)  # Reduce quantity
    modified_order = book.orders_dict.get(sell_id1)
    print(f"Modified order {sell_id1}: {success}")
    print(f"New quantity: {modified_order.quantity if modified_order else 'Order not found'}")


def test_order_matching():
    """Test order matching and trade execution"""
    print("\n=== Testing Order Matching ===")
    
    config = Config()
    book = OrderBook(config)
    book.tick()  # Advance timestamp
    
    # Add some resting orders
    print("\n1. Adding resting orders...")
    book.add_order(Side.BUY, 99.75, 100, "buyer1")
    book.add_order(Side.BUY, 99.50, 200, "buyer2")
    book.add_order(Side.SELL, 100.25, 150, "seller1")
    book.add_order(Side.SELL, 100.50, 100, "seller2")
    
    print(f"Initial state - Best bid: ${book.best_bid}, Best ask: ${book.best_ask}")
    print(f"Trade history length: {len(book.trade_history)}")
    
    # Add a buy order that crosses the spread (should match)
    print("\n2. Adding aggressive buy order...")
    book.tick()
    aggressive_buy = book.add_order(Side.BUY, 100.30, 100, "aggressive_buyer")
    
    print(f"After aggressive buy - Best bid: ${book.best_bid}, Best ask: ${book.best_ask}")
    print(f"Trade history length: {len(book.trade_history)}")
    if book.trade_history:
        trade = book.trade_history[-1]
        print(f"Last trade: {trade['quantity']} shares at ${trade['price']}")
    
    # Add a sell order that crosses the spread
    print("\n3. Adding aggressive sell order...")
    book.tick()
    aggressive_sell = book.add_order(Side.SELL, 99.60, 150, "aggressive_seller")
    
    print(f"After aggressive sell - Best bid: ${book.best_bid}, Best ask: ${book.best_ask}")
    print(f"Trade history length: {len(book.trade_history)}")
    if len(book.trade_history) > 1:
        trade = book.trade_history[-1]
        print(f"Last trade: {trade['quantity']} shares at ${trade['price']}")


def test_price_time_priority():
    """Test price-time priority in order matching"""
    print("\n=== Testing Price-Time Priority ===")
    
    config = Config()
    book = OrderBook(config)
    
    # Add orders with different prices and times
    print("\n1. Adding orders with price-time priority...")
    book.tick()
    id1 = book.add_order(Side.BUY, 99.75, 100, "trader1")  # Earlier time, lower price
    
    book.tick()
    id2 = book.add_order(Side.BUY, 99.80, 100, "trader2")  # Later time, higher price
    
    book.tick()
    id3 = book.add_order(Side.BUY, 99.75, 100, "trader3")  # Later time, same price as first
    
    print(f"Best bid should be $99.80 from trader2: ${book.best_bid}")
    
    # Add a sell order that will match
    book.tick()
    sell_id = book.add_order(Side.SELL, 99.70, 250, "seller")  # Will match all buy orders
    
    print(f"\nAfter matching sell order:")
    print(f"Number of trades: {len(book.trade_history)}")
    
    # Print trade details to verify order of execution
    for i, trade in enumerate(book.trade_history):
        print(f"Trade {i+1}: {trade['quantity']} @ ${trade['price']} "
              f"(buy: {trade['buy_trader']}, sell: {trade['sell_trader']})")


def test_book_depth():
    """Test order book depth functionality"""
    print("\n=== Testing Book Depth ===")
    
    config = Config()
    book = OrderBook(config)
    
    # Add multiple orders at different price levels
    print("\n1. Adding orders at multiple price levels...")
    
    # Buy side
    book.add_order(Side.BUY, 99.75, 100, "buyer1")
    book.add_order(Side.BUY, 99.75, 200, "buyer2")  # Same price level
    book.add_order(Side.BUY, 99.50, 150, "buyer3")
    book.add_order(Side.BUY, 99.25, 100, "buyer4")
    
    # Sell side
    book.add_order(Side.SELL, 100.25, 100, "seller1")
    book.add_order(Side.SELL, 100.25, 150, "seller2")  # Same price level
    book.add_order(Side.SELL, 100.50, 200, "seller3")
    book.add_order(Side.SELL, 100.75, 100, "seller4")
    
    # Get book depth
    depth = book.get_book_depth(levels=3)
    
    print("\nBook Depth (3 levels):")
    print("BIDS:")
    for price, quantity in depth['bids']:
        print(f"  ${price:.2f}: {quantity} shares")
    
    print("ASKS:")
    for price, quantity in depth['asks']:
        print(f"  ${price:.2f}: {quantity} shares")
    
    print(f"\nBest bid: ${depth['best_bid']}")
    print(f"Best ask: ${depth['best_ask']}")
    print(f"Mid price: ${depth['mid_price']:.2f}")
    print(f"Spread: ${depth['spread']:.2f}")


def test_trader_orders():
    """Test trader-specific order tracking"""
    print("\n=== Testing Trader Order Tracking ===")
    
    config = Config()
    book = OrderBook(config)
    
    # Add orders from different traders
    print("\n1. Adding orders from multiple traders...")
    book.add_order(Side.BUY, 99.75, 100, "alice")
    book.add_order(Side.SELL, 100.25, 150, "alice")
    book.add_order(Side.BUY, 99.50, 200, "bob")
    book.add_order(Side.SELL, 100.50, 100, "charlie")
    book.add_order(Side.BUY, 99.25, 300, "alice")
    
    # Get orders for specific trader
    alice_orders = book.get_trader_orders("alice")
    bob_orders = book.get_trader_orders("bob")
    
    print(f"\nAlice's orders ({len(alice_orders)}):")
    for order in alice_orders:
        side_str = "BUY" if order.side == Side.BUY else "SELL"
        print(f"  Order {order.order_id}: {side_str} {order.quantity} @ ${order.price}")
    
    print(f"\nBob's orders ({len(bob_orders)}):")
    for order in bob_orders:
        side_str = "BUY" if order.side == Side.BUY else "SELL"
        print(f"  Order {order.order_id}: {side_str} {order.quantity} @ ${order.price}")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Testing Edge Cases ===")
    
    config = Config()
    book = OrderBook(config)
    
    # Test invalid quantities
    print("\n1. Testing invalid quantities...")
    result = book.add_order(Side.BUY, 100.0, 0, "trader1")  # Zero quantity
    print(f"Zero quantity order result: {result}")
    
    result = book.add_order(Side.BUY, 100.0, -50, "trader1")  # Negative quantity
    print(f"Negative quantity order result: {result}")
    
    # Test price/quantity rounding
    print("\n2. Testing price/quantity rounding...")
    id1 = book.add_order(Side.BUY, 99.736, 157, "trader1")  # Should round to 99.74, 100
    if id1 != -1:
        order = book.orders_dict[id1]
        print(f"Rounded order: ${order.price} for {order.quantity} shares")
    
    # Test canceling non-existent order
    print("\n3. Testing invalid operations...")
    result = book.cancel_order(9999)  # Non-existent order
    print(f"Cancel non-existent order: {result}")
    
    result = book.modify_order(9999, 100)  # Non-existent order
    print(f"Modify non-existent order: {result}")


def main():
    """Run all order book tests"""
    print("📈 ORDER BOOK TESTING SUITE 📈")
    print("=" * 50)
    
    try:
        test_basic_order_operations()
        test_order_matching()
        test_price_time_priority()
        test_book_depth()
        test_trader_orders()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ ERROR DURING TESTING: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()