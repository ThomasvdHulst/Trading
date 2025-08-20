""" This file contains the GARCH model """

# Import used libraries
import numpy as np
from typing import Optional, Deque
from collections import deque
from arch import arch_model
import warnings

# Suppress arch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress data scale warnings for high-frequency tick data
from arch.univariate.base import DataScaleWarning
warnings.filterwarnings("ignore", category=DataScaleWarning)

from ..core.base import IVolatilityModel # Import the IVolatilityModel interface

# Handle imports more flexibly to work with different execution contexts
try:
    from ...config.config import GARCHConfig # Import the GARCHConfig class
except ImportError:
    # Fallback for when running as script or in different context
    import sys
    from pathlib import Path
    
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from config.config import GARCHConfig


class GARCHVolatilityModel(IVolatilityModel):
    """
    GARCH/EGARCH volatility model implementation.
    
    This model maintains a rolling window of price data and periodically
    refits the GARCH model to capture changing market dynamics.
    """
    
    def __init__(self, config: GARCHConfig):
        """
        Initialize GARCH model.
        
        Args:
            config: GARCH configuration parameters
        """
        self.config = config
        
        # Store price history for model fitting
        # Using deque for efficient append/pop operations
        self.price_history: Deque[float] = deque(maxlen=500)
        self.timestamps: Deque[float] = deque(maxlen=500)
        
        # Model state
        self.fitted_model = None
        self.last_fit_tick = 0
        self.current_tick = 0
        self.current_volatility = 0.0002  # Default volatility
        
        # Volatility forecasts cache
        self.volatility_forecast = None
        self.forecast_horizon = 10
        
        # Performance tracking
        self.fit_count = 0
        self.last_fit_success = True
        
        print(f"Initialized {'EGARCH' if config.use_egarch else 'GARCH'}({config.p},{config.q}) volatility model")
        

    def update(self, price: float, timestamp: float) -> None:
        """
        Update model with new price data.
        
        Args:
            price: Current market price
            timestamp: Current timestamp (tick number)
        """
        self.price_history.append(price) # Append the new price to the price history
        self.timestamps.append(timestamp) # Append the new timestamp to the timestamps
        self.current_tick = timestamp # Update the current tick
        
        # Check if we should refit the model
        if self.should_refit():
            self.fit()
            

    def should_refit(self) -> bool:
        """
        Determine if model should be refitted.
        """

        # Need minimum observations
        if len(self.price_history) < self.config.min_observations:
            return False
            
        # Check if enough ticks have passed since last fit
        ticks_since_fit = self.current_tick - self.last_fit_tick
        return ticks_since_fit >= self.config.update_frequency
        

    def fit(self) -> None:
        """
        Fit GARCH model to recent price data.
        """

        if len(self.price_history) < self.config.min_observations:
            print(f"Insufficient data for GARCH fitting: {len(self.price_history)} < {self.config.min_observations}")
            return
            
        try:
            # Calculate returns
            prices = np.array(self.price_history)
            returns = np.diff(np.log(prices)) * 100  # Log returns in percentage
            
            # Skip if returns are too uniform (happens during initialization)
            if np.std(returns) < 1e-8:
                return
            
            # Create GARCH model
            if self.config.use_egarch:
                # EGARCH captures asymmetric effects (leverage effect)
                # Negative shocks have larger impact than positive shocks
                model = arch_model(
                    returns,
                    vol='EGARCH',  # Exponential GARCH
                    p=self.config.p,  # GARCH lag order
                    q=self.config.q,  # ARCH lag order
                    dist='t' # Student's t-distribution for fat tails
                )
            else:
                # Standard GARCH model
                model = arch_model(
                    returns,
                    vol='GARCH',
                    p=self.config.p,
                    q=self.config.q,
                    dist='t'
                )
            
            # Fit model with error handling
            self.fitted_model = model.fit(disp='off', show_warning=False)
            
            # Extract current volatility (annualized)
            # Convert from percentage to decimal
            conditional_vol = self.fitted_model.conditional_volatility[-1] / 100
            
            # Scale to tick frequency (from annual)
            # Assuming 252 trading days, 6.5 hours per day, 3600 seconds per hour
            ticks_per_year = 252 * 6.5 * 3600 * 1000 / 100  # 100ms per tick
            self.current_volatility = conditional_vol / np.sqrt(ticks_per_year)
            
            # Generate volatility forecast
            try:

                forecast = self.fitted_model.forecast(horizon=self.forecast_horizon)
                self.volatility_forecast = forecast.variance.values[-1, :] / 10000  # Convert to decimal

            except Exception as e:

                # Fallback to simulation-based forecasting
                #print(f"Standard forecasting failed, using simulation")
                try:

                    # Use simulation forecasting as fallback
                    sim_forecast = self.fitted_model.forecast(horizon=self.forecast_horizon, method='simulation', simulations=1000)
                    self.volatility_forecast = sim_forecast.variance.values[-1, :] / 10000  # Convert to decimal

                except Exception as sim_e:
                    
                    # If both methods fail, use current volatility for all horizons
                    print(f"Simulation forecasting also failed: {str(sim_e)[:50]}")
                    self.volatility_forecast = np.full(self.forecast_horizon, self.current_volatility ** 2)
            
            # Update state
            self.last_fit_tick = self.current_tick
            self.fit_count += 1
            self.last_fit_success = True
            
            #if self.fit_count % 10 == 1:  # Print every 10th fit
            #    print(f"GARCH model fitted successfully (fit #{self.fit_count})")
            #    print(f"Current volatility: {self.current_volatility:.6f}")
                
        except Exception as e:
            # Don't crash the system if model fitting fails
            self.last_fit_success = False
            print(f"GARCH fitting failed: {str(e)[:100]}")
            
            # Fall back to simple volatility calculation
            if len(self.price_history) > 20:
                prices = np.array(list(self.price_history)[-20:])
                returns = np.diff(prices) / prices[:-1]
                self.current_volatility = np.std(returns)
                

    def get_forecast(self, horizon: int = 1) -> float:
        """
        Get volatility forecast for given horizon.
        
        Args:
            horizon: Number of periods ahead to forecast
            
        Returns:
            Forecasted volatility
        """

        if self.volatility_forecast is not None and horizon <= len(self.volatility_forecast):
            return float(self.volatility_forecast[horizon - 1])
        
        # Return current volatility if no forecast available
        return self.current_volatility
        

    def get_current_volatility(self) -> float:
        """
        Get current (instantaneous) volatility estimate.
        
        Returns:
            Current volatility estimate
        """
        return self.current_volatility
        

    def get_volatility_regime(self) -> str:
        """
        Classify current volatility regime.
        
        Returns:
            Volatility regime classification
        """

        # Define regime thresholds relative to target volatility
        vol_ratio = self.current_volatility / (self.config.vol_target / np.sqrt(252 * 6.5 * 36000))
        
        if vol_ratio < 0.5:
            return "very_low"
        elif vol_ratio < 0.8:
            return "low"
        elif vol_ratio < 1.2:
            return "normal"
        elif vol_ratio < 2.0:
            return "high"
        else:
            return "extreme"
            

    def get_model_parameters(self) -> Optional[dict]:
        """
        Get fitted model parameters for analysis.
        
        Returns:
            Dictionary of model parameters or None if not fitted
        """

        if self.fitted_model is None:
            return None
            
        params = {
            'omega': float(self.fitted_model.params.get('omega', 0)),  # Constant
            'alpha': float(self.fitted_model.params.get('alpha[1]', 0)),  # ARCH coefficient
            'beta': float(self.fitted_model.params.get('beta[1]', 0)),  # GARCH coefficient
        }
        
        # Add EGARCH specific parameters if applicable
        if self.config.use_egarch:
            params['gamma'] = float(self.fitted_model.params.get('gamma[1]', 0))  # Asymmetry parameter
            
        # Add persistence (alpha + beta for GARCH)
        params['persistence'] = params['alpha'] + params['beta']
        
        return params
        

    def get_metrics(self) -> dict:
        """
        Get model metrics for monitoring.
        
        Returns:
            Dictionary of model metrics
        """

        metrics = {
            'current_volatility': self.current_volatility,
            'volatility_regime': self.get_volatility_regime(),
            'fit_count': self.fit_count,
            'last_fit_success': self.last_fit_success,
            'observations': len(self.price_history),
            'ticks_since_fit': self.current_tick - self.last_fit_tick,
        }
        
        # Add model parameters if available
        params = self.get_model_parameters()
        if params:
            metrics['model_params'] = params
            
        return metrics