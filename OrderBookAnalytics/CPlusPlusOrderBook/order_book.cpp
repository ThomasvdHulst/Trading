// Order book in C++

#include <iostream>
#include <map>
#include <unordered_map>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

// Using declarations to avoid std:: prefix
using namespace std;
using namespace std::chrono;

// Type aliases for cleaner code
using Price = int; // Price in cents
using Quantity = int; // Number of shares
using OrderId = uint64_t; // Unique order identifier. uint64_t is a fixed-width integer type (u = unsigned (only 0 and positive number), int = integer, 64 = 64 bits)

// ===== Order Structure =====
// This represents a single order in the order book.
// Structure is public by default
struct Order {
    OrderId id;
    Price price;
    Quantity quantity;
    bool is_buy; // True if the order is a buy order, false if it is a sell order.

    // Constructor - initializes an order
    Order(OrderId id_, Price p, Quantity q, bool buy)
        : id(id_), price(p), quantity(q), is_buy(buy) {}
};


// ===== Price Level =====
// Each price level contains all orders at that price.
class PriceLevel {
    private:
        Price price_;
        Quantity total_quantity_;

        // deque is a double-ended queue - used for FIFO order matching
        // O(1) add/remove at both ends
        // So we create a deque that contains pointers to Order objects, where pointers are used to avoid copying the Order object
        // And we use shared_ptr instead of Order* to avoid manual memory management
        deque<shared_ptr<Order>> orders_;

    public:
        // Constructor - intializes a price level with a given price
        PriceLevel(Price p) : price_(p), total_quantity_(0) {}

        // Add order to this price level (back of queue)
        void add_order(shared_ptr<Order> order) {
            orders_.push_back(order);
            total_quantity_ += order->quantity;
        }

        // Remove order from this price level (for cancellations)
        bool remove_order(OrderId order_id) {
            // Find the order with matching ID
            // find_if is a library function that searches through a container
            // orders_.begin() is the beginning of the deque, and orders_.end() is the end of the deque
            // We then have a Lambda function:
            //  [order_id] is a capture clause, captures the order_id variable from outside
            //  (const shared_ptr<Order>& o) is the parameter, each order in the deque
            //  { return o->id == order_id;} is the body, return true if this is the order we're looking for
            // We then check if we found it, where if 'it' is the end of the deque, we didn't find it
            auto it = find_if(orders_.begin(), orders_.end(),
                [order_id](const shared_ptr<Order>& o) {
                    return o->id == order_id;
                });

            if (it != orders_.end()) {
                total_quantity_ -= (*it)->quantity; // We have to dereference the iterator to get the order object
                orders_.erase(it);
                return true;
            }
            return false;
        }


        // Execute quantity from front of queue (FIFO matching)
        int execute_quantity(int qty_to_execute) {
            int executed = 0;

            while (!orders_.empty() && qty_to_execute > 0) {
                // Using & to avoid copying the order object
                auto& front_order = orders_.front();

                if (front_order->quantity <= qty_to_execute) {
                    // Fully execute this order
                    executed += front_order->quantity;
                    qty_to_execute -= front_order->quantity;
                    total_quantity_ -= front_order->quantity;
                    orders_.pop_front(); // Remove fully executed order
                } else {
                    // Partially execute
                    front_order->quantity -= qty_to_execute;
                    total_quantity_ -= qty_to_execute;
                    executed += qty_to_execute;
                    qty_to_execute = 0;
                }
            }

            return executed;
        }


        // Getters
        Price get_price() const { return price_; }
        Quantity get_total_quantity() const { return total_quantity_; }
        size_t get_order_count() const { return orders_.size(); } // Use size_t for unsigned integer as order size could be bigger than int
        bool is_empty() const { return orders_.empty(); }

};


// ===== Order Book =====
class OrderBook {

    private:
        // map keeps prices sorted automatically
        // For bids: highest price first (greater<Price>)
        // For asks: lowest price first (default less<Price>)
        map<Price, unique_ptr<PriceLevel>, greater<Price>> bid_levels_;
        map<Price, unique_ptr<PriceLevel>> ask_levels_;

        // Hash map for O(1) order lookup by ID
        // Maps order ID to pair of (is_buy, price)
        unordered_map<OrderId, pair<bool, Price>> order_location_;

        // Statistics
        uint64_t total_messages_ = 0;
        uint64_t total_volume_traded_ = 0;

        // For order ID generation
        OrderId next_order_id_ = 1000;

    public:
        // Add a new order to the order book
        bool add_order(OrderId id, bool is_buy, Price price, Quantity quantity) {
            total_messages_++;

            // Check if order ID already exists
            if (order_location_.find(id) != order_location_.end()) { // If the order ID is found in the map
                return false; // Duplicate order ID
            }

            // Create the order
            // make_shared creates a shared_ptr to the new Order object
            // auto is used to automatically deduce the type of the variable
            auto order = make_shared<Order>(id, price, quantity, is_buy);

            if (is_buy) {
                /// Add to bid side
                // If price level doesn't exist, create it
                if (bid_levels_.find(price) == bid_levels_.end()) {
                    bid_levels_[price] = make_unique<PriceLevel>(price);
                }
                bid_levels_[price]->add_order(order);
            } else {
                // Add to ask side
                if (ask_levels_.find(price) == ask_levels_.end()) {
                    ask_levels_[price] = make_unique<PriceLevel>(price);
                }
                ask_levels_[price]->add_order(order);
            }

            // Track order location for fast cancellation
            order_location_[id] = {is_buy, price};
            return true;
        }

        
        // Cancel an existing order
        bool cancel_order(OrderId id) {
            total_messages_++;

            // Find order location
            auto it = order_location_.find(id);
            if (it == order_location_.end()) {
                return false; // Order not found
            }

            // Get the is_buy and price from the second element of the pair, which is a pair of bool and Price
            bool is_buy = it->second.first;
            Price price = it->second.second;

            // Remove from appropriate side
            if (is_buy) {
                // Check if the price level exists
                if (bid_levels_.find(price) == bid_levels_.end()) {
                    order_location_.erase(it); // Clean up the orphaned entry
                    return false;
                }
                if (bid_levels_[price]->remove_order(id)) {
                    // If level is now empty, remove it
                    if (bid_levels_[price]->is_empty()) {
                        bid_levels_.erase(price);
                    }
                    order_location_.erase(it);
                    return true;
                } 
            } else {
                // Check if the price level exists
                if (ask_levels_.find(price) == ask_levels_.end()) {
                    order_location_.erase(it); // Clean up the orphaned entry
                    return false;
                }
                if (ask_levels_[price]->remove_order(id)) {
                    // If level is now empty, remove it
                    if (ask_levels_[price]->is_empty()) {
                        ask_levels_.erase(price);
                    }
                    order_location_.erase(it);
                    return true;
                }
            }

            return false; // Should never reach here
        }


        // Execute a market order (aggressive order that crosses spread for simplicity)
        int execute_market_order(bool is_buy, Quantity quantity) {
            total_messages_++;
            int executed = 0;

            if (is_buy) {
                // Buy market order - match against best ask
                while (!ask_levels_.empty() && quantity > 0) {
                    // Get best ask (lowest price)
                    auto& [price, level] = *ask_levels_.begin();

                    int level_executed = level->execute_quantity(quantity);
                    quantity -= level_executed;
                    executed += level_executed;
                    total_volume_traded_ += level_executed;

                    // Remove if level is empty
                    if (level->is_empty()) {
                        ask_levels_.erase(ask_levels_.begin());
                    }
                }
            } else {
                // Sell market order - match against best bid
                while (!bid_levels_.empty() && quantity > 0) {
                    // Get best bid (highest price)
                    auto& [price, level] = *bid_levels_.begin();

                    int level_executed = level->execute_quantity(quantity);
                    quantity -= level_executed;
                    executed += level_executed;
                    total_volume_traded_ += level_executed;

                    // Remove if level is empty
                    if (level->is_empty()) {
                        bid_levels_.erase(bid_levels_.begin());
                    }
                }   
            }

            return executed;
        }


        // Get the best bid price and quantity
        pair<Price, Quantity> get_best_bid() const {
            if (bid_levels_.empty()) {
                return {0, 0}; // No bids
            }

            // We use const auto& to avoid copying the price and level
            const auto& [price, level] = *bid_levels_.begin();
            return {price, level->get_total_quantity()};
        }

        // Get the best ask price and quantity
        pair<Price, Quantity> get_best_ask() const {
            if (ask_levels_.empty()) {
                return {INT32_MAX, 0}; // No asks
            }

            // We use const auto& to avoid copying the price and level
            const auto& [price, level] = *ask_levels_.begin();
            return {price, level->get_total_quantity()};
        }

        // Calculate the spread
        double get_spread() const {
            // As the pair are just two int values, & is not needed as these values are copyable
            auto [bid_price, bid_qty] = get_best_bid();
            auto [ask_price, ask_qty] = get_best_ask();

            if (bid_qty == 0 || ask_qty == 0) {
                return 0.0; // No spread
            }

            return (ask_price - bid_price) / 100.0; // Convert to dollars
        }

        // Calculate mid price
        double get_mid_price() const {
            auto [bid_price, bid_qty] = get_best_bid();
            auto [ask_price, ask_qty] = get_best_ask();

            if (bid_qty == 0 || ask_qty == 0) {
                return 0.0; // No mid price
            }

            return (bid_price + ask_price) / 200.0; // Convert to dollars
        }


        // Display order book state
        void display(int levels = 5) const {
            cout << "\nOrder book!" << endl;
            cout << "Messages processed: " << total_messages_ << endl;
            cout << "Total volume traded: " << total_volume_traded_ << " shares" << endl;

            // Display asks (in reverse order)
            vector<pair<Price, Quantity>> asks_to_show;
            int count = 0;
            for (const auto& [price, level] : ask_levels_) {
                if (count++ >= levels) break;
                asks_to_show.push_back({price, level->get_total_quantity()});
            }

            // Print asks from highest to lowest
            for (auto it = asks_to_show.rbegin(); it != asks_to_show.rend(); ++it) {
                cout << "Ask: $" << fixed << setprecision(2) << it->first / 100.0 << " x " << it->second << endl;
            }

            // Display bids
            count = 0;
            for (const auto& [price, level] : bid_levels_) {
                if (count++ >= levels) break;
                cout << "Bid: $" << fixed << setprecision(2) << price / 100.0 << " x " << level->get_total_quantity() << endl;
            }

            // Print spread
            cout << "Spread: $" << fixed << setprecision(2) << get_spread() << endl;

            cout << "===============\n" << endl;
        }

        // Get statistics
        void print_stats() const {
            cout << "\nPerformance stats!" << endl;
            cout << "Messages processed: " << total_messages_ << endl;
            cout << "Active orders: " << order_location_.size() << endl;
            cout << "Bid levels: " << bid_levels_.size() << endl;
            cout << "Ask levels: " << ask_levels_.size() << endl;
            cout << "Volume traded: " << total_volume_traded_ << " shares" << endl;

            auto [bid_price, bid_qty] = get_best_bid();
            auto [ask_price, ask_qty] = get_best_ask();

            if (bid_qty > 0) {
                cout << "Best bid: $" << fixed << setprecision(2) << bid_price / 100.0 << " x " << bid_qty << endl;
            }

            if (ask_qty > 0) {
                cout << "Best ask: $" << fixed << setprecision(2) << ask_price / 100.0 << " x " << ask_qty << endl;
            }

            cout << "Spread: $" << fixed << setprecision(2) << get_spread() << endl;
            cout << "Mid price: $" << fixed << setprecision(2) << get_mid_price() << endl;
            cout << "===============\n" << endl;
        }

        // Generate next order ID
        OrderId generate_order_id() {
            return next_order_id_++;
        }

};


// ===== Message Types =====
// Simple message structure
struct Message {
    enum Type {ADD, CANCEL, EXECUTE};
    Type type;
    OrderId order_id;
    bool is_buy;
    Price price;
    Quantity quantity;
};


// ===== Simulator =====
class MarketSimulator {
    private:
        OrderBook& book_;
        mt19937 rng_; // Random number generator
        uniform_real_distribution<> price_dist_; // Price distribution
        uniform_int_distribution<int> quantity_dist_; // Quantity distribution
        uniform_real_distribution<> action_dist_; // Action distribution

        Price base_price_ = 10000; // $100.00
        vector<OrderId> active_orders_; // Track orders we can cancel

    
    public:
        MarketSimulator(OrderBook& book)
            : book_(book),
              rng_(chrono::steady_clock::now().time_since_epoch().count()),
              price_dist_(0.01, 0.20), // Price offset from base price
              quantity_dist_(100, 1000), // Quantity range
              action_dist_(0.0, 1.0) {}

        // Generate and process random messages
        void simulate(int num_messages) {
            cout << "Simulating " << num_messages << " messages..." << endl;

            // First, populate the book with initial orders
            for (int i = 0; i < 10; ++i) {
                // Add bids
                Price bid_price = base_price_ - (i + 1) * 10; // 10 cents per level
                OrderId bid_id = book_.generate_order_id();
                Quantity bid_qty = quantity_dist_(rng_);
                if (book_.add_order(bid_id, true, bid_price, bid_qty)) {
                    active_orders_.push_back(bid_id);
                }

                // Add asks
                Price ask_price = base_price_ + (i + 1) * 10;
                OrderId ask_id = book_.generate_order_id();
                Quantity ask_qty = quantity_dist_(rng_);
                if (book_.add_order(ask_id, false, ask_price, ask_qty)) {
                    active_orders_.push_back(ask_id);
                }
            }

            cout << "Initial messages processed!" << endl;

            // Generate random messages
            for (int i = 0; i < num_messages; ++i) {
                double action = action_dist_(rng_);

                if (action < 0.5) {
                    // Add new order (50% chance)
                    add_random_order();
                } else if (action < 0.7) {
                    // Execute market order (20% chance)
                    execute_random_market_order();
                } else if (action < 0.9 && !active_orders_.empty()) {
                    // Cancel random order (20% chance)
                    cancel_random_order();
                } else {
                    // Add new order (10% chance)
                    add_random_order();
                }

                // Periodically display the order book
                //if ((i + 1) % 100 == 0) {
                //    cout << "Processed " << i + 1 << " messages..." << endl;
                //}
            }
        }


    private:

        void add_random_order() {
            bool is_buy = action_dist_(rng_) < 0.5;

            // Price around the base price
            Price price;
            if (is_buy) {
                price = base_price_ - static_cast<Price>(price_dist_(rng_) * 100); // 1-20% below base
                price = max(price, 1); // Ensure positive price
            } else {
                price = base_price_ + static_cast<Price>(price_dist_(rng_) * 100); // 1-20% above base
            }

            Quantity quantity = quantity_dist_(rng_);
            OrderId id = book_.generate_order_id();

            bool success = book_.add_order(id, is_buy, price, quantity);
            if (success) {
                active_orders_.push_back(id);
            }
        }

        void execute_random_market_order() {
            bool is_buy = action_dist_(rng_) < 0.5;
            Quantity quantity = quantity_dist_(rng_) / 2; // Smaller market orders

            int executed = book_.execute_market_order(is_buy, quantity);

            // Update base price based on executed quantity
            if (executed > 0) {
                if (is_buy) {
                    base_price_ += 1; // Price goes up on buying
                } else {
                    base_price_ -= 1; // Price goes down on selling
                }
            }
        }

        void cancel_random_order() {
            if (active_orders_.empty()) return;

            // Pick random order from active orders
            size_t size = active_orders_.size();
            if (size == 0) {
                return; // Double check
            }
            
            uniform_int_distribution<size_t> idx_dist(0, size - 1);
            size_t idx = idx_dist(rng_);
            
            // Bounds check
            if (idx >= active_orders_.size()) return;
            
            OrderId id = active_orders_[idx];

            bool cancel_success = book_.cancel_order(id);
            
            if (cancel_success) {
                // Remove from active orders - use safer method
                active_orders_.erase(active_orders_.begin() + idx);
            } else {
                // Order wasn't in book, but might still be in our list - remove it anyway
                active_orders_.erase(active_orders_.begin() + idx);
            }
        }
};


// ===== Performance Test =====
void performance_test(int num_orders) {
    cout << "\nPerformance test!" << endl;

    OrderBook book;

    // Measure time for adding orders
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_orders; ++i) {
        book.add_order(i, i % 2 == 0, 10000 + (i % 100), 100 + (i % 500));
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Added " << num_orders << " orders in " << duration.count() << " microseconds" << endl;
    cout << "Average time per order: " << static_cast<double>(duration.count()) / num_orders << " microseconds per order" << endl;
    cout << "Throughput: " << (static_cast<double>(num_orders) * 1000000 / duration.count()) << " orders per second" << endl;

    // Test market orders
    start = high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        book.execute_market_order(i % 2 == 0, 50);
    }

    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);

    cout << "Executed 100 market orders in " << duration.count() << " microseconds" << endl;

    book.print_stats();
}


// ===== Main Function =====
int main() {
    cout << "C++ Order Book Simulation" << endl;

    // Create order book
    OrderBook book;

    // Create and run simulator
    cout << "Creating simulator..." << endl;
    MarketSimulator simulator(book);
    cout << "Starting simulation..." << endl;
    simulator.simulate(1000000);
    cout << "Simulation complete!" << endl;

    // Display final state
    cout << "Displaying book..." << endl;
    book.display();
    cout << "Printing stats..." << endl;
    book.print_stats();
    cout << "Done!" << endl;

    // Run performance test
    performance_test(10000000);

    return 0;
}
