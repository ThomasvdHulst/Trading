""" Configuration system using Pydantic for type validation """


from typing import Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class MarketRegime(str, Enum):
    """ Enum for market regimes """
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    QUIET = "quiet"


class MarketConfig(BaseModel):
    """ Market structure configuration """

    tick_size : float = Field(0.01, gt=0, description="Minimum price increment")
    lot_size: int = Field(100, gt=0, description="Minimum order size")
    initial_price: float = Field(100.0, description="Starting price for simulation")
    max_order_book_depth: int = Field(10, gt=0, description="Maximum tracked price levels")
    
    model_config = {"validate_assignment": True}


class MarketMakerConfig(BaseModel):
    """ Market maker strategy configuration"""

    trader_id: str = Field("MM_01", description="Unique identifier for the market maker")
    initial_capital: float = Field(1000000.0, gt=0, description="Starting capital")
    base_spread: float = Field(0.20, gt=0, description="Base bid-ask spread")
    quote_size: int = Field(1000, gt=0, description="Default order size")
    max_inventory: int = Field(10000, gt=0, description="Maximum position size")
    inventory_skew_factor: float = Field(0.0001, ge=0, description="Price adjustment per unit inventory")
    max_orders_per_side: int = Field(3, gt=0, le=5, description="Maximum orders per side")
    num_quote_levels: int = Field(3, gt=0, le=5, description="Number of quote levels")
    level_spacing_ticks: int = Field(2, gt=0, description="Tick spacing between levels")
    
    model_config = {"validate_assignment": True}


class RiskConfig(BaseModel):
    """ Risk management configuration """

    max_position_value: float = Field(500000.0, gt=0, description="Maximum position value in currency")
    stop_loss_threshold: float = Field(-10000.0, description="Stop trading if PnL below this")
    inventory_pct_limit: float = Field(0.8, gt=0, le=1, description="Start reducing when at x% of max inventory")

    model_config = {"validate_assignment": True}


class GARCHConfig(BaseModel):
    """ GARCH model configuration """

    p: int = Field(1, ge=1, description="GARCH lag order")
    q: int = Field(1, ge=1, description="ARCH lag order")
    vol_target: float = Field(0.15, gt=0, description="Annual volatility target")
    update_frequency: int = Field(100, gt=0, description="Ticks between model refits")
    min_observations: int = Field(50, gt=0, description="Minimum observations for fitting")
    use_egarch: bool = Field(True, description="Use EGARCH model")

    model_config = {"validate_assignment": True}

    # Professional practice: validate interdependent fields
    @field_validator('min_observations')
    @classmethod
    def validate_min_obs(cls, v, info):
        """ Ensure we have enough observations for the specified lag orders"""
        
        if info.data and 'p' in info.data and 'q' in info.data:
            p_val = info.data['p']
            q_val = info.data['q']
            min_required = max(p_val, q_val) * 10
            if v < min_required:
                raise ValueError(f"Minimum observations must be at least {min_required} for GARCH({p_val}, {q_val})")
        return v
    

class MicropriceConfig(BaseModel):
    """ Microprice model configuration """

    use_multi_level: bool = Field(True, description="Use multiple levels for microprice")
    depth_levels: int = Field(5, ge=1, le=10, description="Number of levels to track")
    depth_decay: float = Field(0.5, gt=0, le=1, description="Exponential decay factor for deeper levels")

    model_config = {"validate_assignment": True}


class AdverseSelectionConfig(BaseModel):
    """ Adverse selection model configuration """

    lookback_trades: int = Field(20, gt=0, description="Number of recent trades to consider")
    toxicity_threshold: float = Field(0.6, ge=0, le=1, description="Threshold for toxic trading")
    decay_factor: float = Field(0.95, gt=0, le=1, description="Decay factor for toxicity")
    min_trades_for_scoring: int = Field(5, gt=0, description="Minimum trades required for scoring")

    model_config = {"validate_assignment": True}


class ModelsConfig(BaseModel):
    """ All model configurations """
    garch: GARCHConfig = Field(default_factory=GARCHConfig)
    microprice: MicropriceConfig = Field(default_factory=MicropriceConfig)
    adverse_selection: AdverseSelectionConfig = Field(default_factory=AdverseSelectionConfig)

    model_config = {"validate_assignment": True}


class SpreadComponentsConfig(BaseModel):
    """ Spread calculation component weights """
    base_weight: float = Field(0.2, ge=0, le=1)
    volatility_weight: float = Field(0.3, ge=0, le=1)
    inventory_weight: float = Field(0.2, ge=0, le=1)
    adverse_weight: float = Field(0.2, ge=0, le=1)
    imbalance_weight: float = Field(0.1, ge=0, le=1)
    
    @field_validator('imbalance_weight')
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure weights sum to 1.0"""
        if info.data:
            total = sum([info.data.get(k, 0) for k in ['base_weight', 'volatility_weight', 
                                                       'inventory_weight', 'adverse_weight']]) + v
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Spread component weights must sum to 1.0, got {total}")
        return v
    
    model_config = {"validate_assignment": True}


class QuoteSizingConfig(BaseModel):
    """Quote sizing strategy configuration"""
    defensive_mode_threshold: int = Field(5, gt=0, description="Sweeps before defensive mode")
    sweep_detection_levels: int = Field(2, gt=0, description="Levels hit to detect sweep")
    size_reduction_per_level: float = Field(0.8, gt=0, le=1, description="Size multiplier per level")

    model_config = {"validate_assignment": True}


class StrategyConfig(BaseModel):
    """Strategy configuration"""
    spread_components: SpreadComponentsConfig = Field(default_factory=SpreadComponentsConfig)
    quote_sizing: QuoteSizingConfig = Field(default_factory=QuoteSizingConfig)

    model_config = {"validate_assignment": True}


class MarketDynamicsConfig(BaseModel):
    """Market dynamics simulation configuration"""
    volatility: float = Field(0.0002, gt=0, description="Price volatility per tick")
    mean_reversion_speed: float = Field(0.01, ge=0, description="Mean reversion strength")
    event_probability: float = Field(0.02, ge=0, le=1, description="Probability of price jump")

    model_config = {"validate_assignment": True}


class TraderGroupConfig(BaseModel):
    """Configuration for a group of traders"""
    count: int = Field(1, ge=0, description="Number of traders")
    probability: float = Field(0.5, ge=0, le=1, description="Order probability per tick")
    
    
class InformedTraderConfig(TraderGroupConfig):
    """Informed trader specific configuration"""
    accuracy: float = Field(0.7, ge=0, le=1, description="Prediction accuracy")

    model_config = {"validate_assignment": True}


class NoiseTraderConfig(TraderGroupConfig):
    """Noise trader specific configuration"""
    price_sensitivity: float = Field(0.002, gt=0, description="Price randomness factor")

    model_config = {"validate_assignment": True}


class ArbitrageurConfig(BaseModel):
    """Arbitrageur configuration"""
    count: int = Field(1, ge=0)
    threshold: float = Field(0.001, gt=0, description="Minimum profit to trade")

    model_config = {"validate_assignment": True}


class MomentumTraderConfig(BaseModel):
    """Momentum trader configuration"""
    count: int = Field(1, ge=0)
    lookback_period: int = Field(10, gt=0, description="Periods for momentum calculation")
    momentum_threshold: float = Field(0.0005, gt=0, description="Minimum momentum to trade")

    model_config = {"validate_assignment": True}


class ParticipantsConfig(BaseModel):
    """All market participants configuration"""
    informed_traders: InformedTraderConfig = Field(default_factory=lambda: InformedTraderConfig(count=2, probability=0.3))
    noise_traders: NoiseTraderConfig = Field(default_factory=lambda: NoiseTraderConfig(count=10, probability=0.5))
    arbitrageurs: ArbitrageurConfig = Field(default_factory=ArbitrageurConfig)
    momentum_traders: MomentumTraderConfig = Field(default_factory=MomentumTraderConfig)

    model_config = {"validate_assignment": True}


class SimulationConfig(BaseModel):
    """Simulation parameters"""
    ticks: int = Field(10000, gt=0, description="Number of simulation ticks")
    random_seed: int = Field(37, description="Random seed for reproducibility")
    tick_frequency_ms: int = Field(100, gt=0, description="Milliseconds per tick")
    warmup_period: int = Field(100, ge=0, description="Ticks before MM starts")

    model_config = {"validate_assignment": True}


class FeesConfig(BaseModel):
    """Transaction fees configuration"""
    exchange_fee: float = Field(0.00001, ge=0, description="Fee per traded unit")
    latency_cost: float = Field(0.0001, ge=0, description="Implicit latency cost")

    model_config = {"validate_assignment": True}

    
class Config(BaseModel):
    """Main configuration class - combines all sub-configurations"""
    market: MarketConfig = Field(default_factory=MarketConfig)
    market_maker: MarketMakerConfig = Field(default_factory=MarketMakerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    market_dynamics: MarketDynamicsConfig = Field(default_factory=MarketDynamicsConfig)
    participants: ParticipantsConfig = Field(default_factory=ParticipantsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)

    class Config:
        """ Pydantic configuration """
        # This allows us to use field names as attributes
        validate_assignment = True
        use_enum_values = True


def load_config(config_path: Optional[str] = None) -> Config:
    """ Load configuration from YAML file or use defaults 
    
    Args:
        config_path: Path to the configuration file

    Returns:
        Validated configuration object
    """

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Pydantic will validate the configuration and raise clear errors if any fields are invalid
        return Config(**config_dict)
    
    # Return defaults if no path provided
    return Config()


def save_config(config: Config, output_path: str) -> None:
    """ Save configuration to YAML file 
    
    Args:
        config: Configuration object
        output_path: Output file path
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Convert to dict and save as YAML
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


_default_config: Optional[Config] = None

def get_config() -> Config:
    """ Get the default configuration instance """

    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config