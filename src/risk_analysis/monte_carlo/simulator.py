from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from src.models import Risk, Scenario
from src.risk_analysis.cascade.business_impact import BusinessMetrics, BusinessImpact

@dataclass
class SimulationParameters:
    num_simulations: int = 10000
    confidence_level: float = 0.95
    time_horizon: int = 365  # days

class MonteCarloSimulator:
    def __init__(self, params: SimulationParameters):
        self.params = params
        
    def simulate_business_impacts(
        self,
        base_impact: BusinessImpact,
        risk: Risk,
        scenario: Scenario
    ) -> Dict[str, np.ndarray]:
        """Run Monte Carlo simulation for business impacts"""
        
        # Initialize arrays for each metric
        revenue_impacts = np.zeros(self.params.num_simulations)
        cost_impacts = np.zeros(self.params.num_simulations)
        market_share_impacts = np.zeros(self.params.num_simulations)
        operational_impacts = np.zeros(self.params.num_simulations)
        supply_chain_impacts = np.zeros(self.params.num_simulations)
        
        # Run simulations
        for i in range(self.params.num_simulations):
            # Generate random variations
            revenue_var = np.random.normal(1, 0.2)  # 20% standard deviation
            cost_var = np.random.normal(1, 0.15)
            market_var = np.random.normal(1, 0.25)
            op_var = np.random.normal(1, 0.1)
            supply_var = np.random.normal(1, 0.3)
            
            # Apply variations to base impacts
            revenue_impacts[i] = base_impact.direct_impact.revenue * revenue_var
            cost_impacts[i] = base_impact.direct_impact.costs * cost_var
            market_share_impacts[i] = base_impact.direct_impact.market_share * market_var
            operational_impacts[i] = base_impact.direct_impact.operational_efficiency * op_var
            supply_chain_impacts[i] = base_impact.direct_impact.supply_chain_disruption * supply_var
            
        return {
            "revenue": revenue_impacts,
            "costs": cost_impacts,
            "market_share": market_share_impacts,
            "operational": operational_impacts,
            "supply_chain": supply_chain_impacts
        }
        
    def calculate_risk_metrics(self, simulation_results: Dict[str, np.ndarray]) -> Dict:
        """Calculate VaR and other risk metrics"""
        metrics = {}
        
        for metric, values in simulation_results.items():
            var = np.percentile(values, (1 - self.params.confidence_level) * 100)
            cvar = np.mean(values[values <= var])
            
            metrics[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "var": var,
                "cvar": cvar,
                "max_loss": np.min(values),
                "confidence_interval": [
                    np.percentile(values, 2.5),
                    np.percentile(values, 97.5)
                ]
            }
            
        return metrics
        
    def stress_test(
        self,
        simulation_results: Dict[str, np.ndarray],
        stress_factor: float = 2.0
    ) -> Dict:
        """Perform stress testing on simulation results"""
        stress_metrics = {}
        
        for metric, values in simulation_results.items():
            # Apply stress factor to standard deviation
            stressed_values = np.random.normal(
                np.mean(values),
                np.std(values) * stress_factor,
                self.params.num_simulations
            )
            
            stress_metrics[metric] = {
                "stressed_var": np.percentile(
                    stressed_values,
                    (1 - self.params.confidence_level) * 100
                ),
                "stressed_mean": np.mean(stressed_values),
                "stress_impact": (
                    np.mean(stressed_values) - np.mean(values)
                ) / np.mean(values)
            }
            
        return stress_metrics