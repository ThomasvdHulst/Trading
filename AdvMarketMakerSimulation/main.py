""" Main entry point for the simulation. """

# Import used libraries
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import load_config
from simulation.simulator import MarketSimulator


def main():
    """ Run the market maker simulation. """

    print("Welcome to the Market Maker Simulation!")
    
    # Load configuration
    config_path = 'configs/default.yaml'
    print(f"Loading configuration from {config_path}...")

    try:
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        config = load_config()

    print()
    print("Configuration summary:")
    print("Market maker:")
    print(f"Initial capital: ${config.market_maker.initial_capital:,.2f}")
    print(f"Max inventory: {config.market_maker.max_inventory:,.2f}")
    print(f"Base spread: {config.market_maker.base_spread:,.2f}")
    print(f"Quote levels: {config.market_maker.num_quote_levels}")
    print()
    print("Models:")
    print(f"Volatility: {'EGARCH' if config.models.garch.use_egarch else 'GARCH'} (p={config.models.garch.p}, q={config.models.garch.q})")
    print(f"Microprice: {'Multi-level' if config.models.microprice.use_multi_level else 'Single-level'}")
    print(f"Adverse selection: Threshold={config.models.adverse_selection.toxicity_threshold}")
    print()
    print("Simulation:")
    print(f"Duration: {config.simulation.ticks} ticks")
    print(f"Participants:")
    print(f"{config.participants.informed_traders.count} informed traders (accuracy={config.participants.informed_traders.accuracy})")
    print(f"{config.participants.noise_traders.count} noise traders (price_sensitivity={config.participants.noise_traders.price_sensitivity})")
    print(f"{config.participants.arbitrageurs.count} arbitrageurs (threshold={config.participants.arbitrageurs.threshold})")
    print(f"{config.participants.momentum_traders.count} momentum traders (lookback={config.participants.momentum_traders.lookback_period}, threshold={config.participants.momentum_traders.momentum_threshold})")
    print()

    # Run simulation
    print("Initializing simulator...")
    simulator = MarketSimulator(config)
    results = simulator.run()


if __name__ == "__main__":
    main()