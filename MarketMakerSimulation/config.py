""" This file contains the configuration for the market maker simulation """

class Config:

    # Market parameters
    TICK_SIZE = 0.01 # Minimum price change
    LOT_SIZE = 100 # Number of shares per lot
    INITIAL_PRICE = 100 # Starting price

    # Order book parameters
    MAX_ORDER_BOOK_DEPTH = 10 # Maximum number of price levels to track

    # Market maker parameters
    INITIAL_CAPITAL = 1000000 # Initial capital
    BASE_SPREAD = 0.20 # Base spread 
    QUOTE_SIZE = 1000 # Default size for quotes (10 lots)
    MAX_INVENTORY = 10000 # Maximum position size (long or short)
    INVENTORY_SKEW_FACTOR = 0.0001 # How much to adjust spread based on inventory

    # Risk management parameters
    MAX_POSITION_VALUE = 500000 # Maximum position value in dollars
    STOP_LOSS_THRESHOLD = -10000 # Stop trading if PnL drops below this amount
    MAX_ORDERS_PER_SIDE = 5 # Maximum number of order on each side

    # Market dynamics parameters
    VOLATILITY = 0.0002 # Price volatility per tick
    MEAN_REVERSION_SPEED = 0.01 # How quickly price reverts to mean
    EVENT_PROBABILITY = 0.02 # Probability of a new event (jump)
    INFORMED_TRADER_PROB = 0.3 # Probability of informed trader arrival
    NOISE_TRADER_PROB = 0.5 # Probability of noise trader arrival

    # Simulation parameters
    SIMULATION_TICKS = 10000 # Number of ticks (time steps) to run the simulation
    RANDOM_SEED = 37 # Seed for reproducibility

    # Transaction costs
    EXCHANGE_FEE = 0.00001 # Fee per traded unit
    LATENCY_FEE = 0.0001 # Cost of latency

    # Backtesting parameters
    TICK_FREQUENCY_MS = 100 # Milliseconds per tick
    WARMUP_PERIOD = 100 # Ticks before market starts quoting

    # Visualization parameters
    PLOT_FREQUENCY = 100 # Update plots every 100 ticks
    SHOW_BOOK_DEPTH = 5 # Number of price levels to show in book depth plot

    # Participant parameters
    NUMB_INFORMED_TRADERS = 2 # Number of informed traders
    NUMB_NOISE_TRADERS = 20 # Number of noise traders
    NUMB_ARBITRAGEURS = 1 # Number of arbitrageurs
    NUMB_MOMENTUM_TRADERS = 1 # Number of momentum traders