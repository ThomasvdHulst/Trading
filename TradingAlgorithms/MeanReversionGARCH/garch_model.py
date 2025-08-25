"""
This file contains the GARCH model for the mean reversion strategy.
"""

# Import used libraries
import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Define class for GARCH model
class GARCHModel:

    def __init__(self, config):
        self.config = config
        self.garch_p = config.GARCH_P
        self.garch_q = config.GARCH_Q
        self.garch_dist = config.GARCH_DIST
        self.lookback = config.GARCH_LOOKBACK
        self.model = None
        self.fitted_model = None


    def calculate_returns(self, prices):
        """ Calculate the log-returns from the prices """

        returns = np.log(prices / prices.shift(1)).dropna() # Compute log-returns
        returns *= 100 # Convert to percentage

        return returns
    

    def fit_garch_model(self, returns):
        """ Fit a GARCH model to the returns """

        try:
            # Create a GARCH model
            self.model = arch_model(
                returns, # Returns
                vol='Garch', # Type of volatility model
                p=self.garch_p, # Number of lagged squared returns
                q=self.garch_q, # Number of lagged volatility
                dist=self.garch_dist # Distribution of the returns
            )

            # Fit the model
            self.fitted_model = self.model.fit(disp='off')

            return True

        except Exception as e:
            print(f"Error fitting GARCH model: {e}")
            return False
        

    def forecast_volatility(self, steps=1):
        """ Forecast the volatility using the GARCH model """

        if self.fitted_model is None:
            return None
        

        try:
            # Get volatility forecast
            forecast = self.fitted_model.forecast(horizon=steps)
            forecasted_variance = forecast.variance.iloc[-1, 0]
            forecasted_volatility = np.sqrt(forecasted_variance)

            return forecasted_volatility / 100 # Convert back from percentage
        
        except Exception as e:
            print(f"Error forecasting volatility: {e}")
            return None
        

    def rolling_garch_forecast(self, data):
        """ Apply rolling GARCH volatility forecast """

        prices = data['close'].copy()
        returns = self.calculate_returns(prices)

        # Initialize arrays to store results
        forecasted_volatilities = []
        conditional_volatilities = []

        print("Running rolling GARCH volatility forecast...")

        for i in range(len(returns)):
            if i < self.lookback:
                # Not enough data, so just use simple rolling std
                vol_estimate = returns.iloc[:i+1].std()
                forecasted_volatilities.append(vol_estimate)
                conditional_volatilities.append(vol_estimate)
            else:
                # Use GARCH model
                train_returns = returns.iloc[i-self.lookback:i+1]

                if self.fit_garch_model(train_returns):
                    # Get one-step ahead forecast
                    forecast_vol = self.forecast_volatility()

                    # Get current conditional volatility
                    current_vol = np.sqrt(self.fitted_model.conditional_volatility.iloc[-1]) / 100

                    if forecast_vol is not None:
                        forecasted_volatilities.append(forecast_vol)
                        conditional_volatilities.append(current_vol)
                    else:
                        # Fallback to rolling std
                        vol_estimate = train_returns.std()
                        forecasted_volatilities.append(vol_estimate)
                        conditional_volatilities.append(vol_estimate)

            if i % 100 == 0:
                print(f"Done with {i / len(returns) * 100:.2f}%")

        results_data = data.iloc[1:].copy() # Start from second row, due to missing observation in returns
        results_data['forecasted_volatility'] = forecasted_volatilities
        results_data['conditional_volatility'] = conditional_volatilities

        return results_data
    

    def print_garch_summary(self, data):
        """ Print summary statistics of the GARCH model """

        if "forecasted_volatility" not in data.columns:
            print("No GARCH volatility data available")
            return
        
        print("---------------------")
        print("GARCH model summary:")
        print("Forecasted volatility statistics:")
        print(f"Mean: {data['forecasted_volatility'].mean():.4f}")
        print(f"Std: {data['forecasted_volatility'].std():.4f}")
        print(f"Min: {data['forecasted_volatility'].min():.4f}")
        print(f"Max: {data['forecasted_volatility'].max():.4f}")
        
        print("Conditional volatility statistics:")
        print(f"Mean: {data['conditional_volatility'].mean():.4f}")
        print(f"Std: {data['conditional_volatility'].std():.4f}")
        print(f"Min: {data['conditional_volatility'].min():.4f}")
        print(f"Max: {data['conditional_volatility'].max():.4f}")
        print("---------------------")
        