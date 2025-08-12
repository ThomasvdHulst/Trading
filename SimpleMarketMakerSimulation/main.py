""" This file contains the main function for running the market simulation """

# Import used libraries
import warnings
warnings.filterwarnings("ignore")

from config import Config
from market_simulator import MarketSimulator
from backtester import MarketMakerBacktester
from risk_manager import RiskManager


def main():
    """ Run the market maker simulation """

    print("MARKET MAKER SIMULATION")
    print("\n")

    # Initialize configuration
    config = Config()

    print("Configuration:")
    print(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Max inventory: {config.MAX_INVENTORY}")
    print(f"Base spread: {config.BASE_SPREAD:.3f}")
    print(f"Simulation ticks: {config.SIMULATION_TICKS}")
    print(f"Quote size: {config.QUOTE_SIZE}")
    print("\n")

    # Initialize components
    market_sim = MarketSimulator(config)
    risk_manager = RiskManager(config)
    backtester = MarketMakerBacktester(config)

    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(market_sim)
    print("\n")

    # Analyze results
    print("Analyzing results...")
    analysis = backtester.analyze_performance(results)
    print("\n")

    # Generate report
    print("Generating report...")
    backtester.generate_report(analysis)
    print("\n")

    # Risk analysis
    print("Performing risk analysis...")
    risk_assessment = risk_manager.assess_risk(
        market_sim.market_maker, 
        market_sim.order_book
    )
    risk_metrics = risk_manager.get_risk_metrics()

    print("Risk Assessment:")
    print(f"Overall Risk Level: {risk_assessment['overall_risk'].upper()}")
    print(f"Inventory Risk: {risk_assessment['inventory_risk']}")
    print(f"PnL Risk: {risk_assessment['pnl_risk']}")
    print(f"Adverse Selection Risk: {risk_assessment['adverse_selection_risk']}")
    print(f"Market Risk: {risk_assessment['market_risk']}")

    print("\n")
    print("Risk Metrics:")
    print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"Risk State: {risk_metrics['risk_state']}")
    print(f"Inventory Imbalance Time: {risk_metrics['inventory_imbalance_time']} ticks")
    print(f"VAR 95: {risk_metrics['var_95']:.2f}")

    

if __name__ == "__main__":
    main()