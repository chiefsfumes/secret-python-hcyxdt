from typing import Dict, List, Tuple
import numpy as np
from src.models import Risk, Scenario
from src.risk_analysis.cascade.business_impact import BusinessMetrics, BusinessImpact, CascadeNode

class PropagationEngine:
    def __init__(self, risk_network: Dict, scenario: Scenario):
        self.risk_network = risk_network
        self.scenario = scenario
        self.propagation_delays = self._init_propagation_delays()
        self.impact_multipliers = self._init_impact_multipliers()
        
    def _init_propagation_delays(self) -> Dict[str, Dict[str, int]]:
        """Initialize base propagation delays between risk categories"""
        return {
            "Physical Risks": {
                "Supply Chain Disruptions": 7,    # 1 week
                "Market Risks": 30,               # 1 month
                "Financial System Risks": 14      # 2 weeks
            },
            "Transition Risks": {
                "Market Risks": 90,              # 3 months
                "Financial System Risks": 60      # 2 months
            },
            "Nature-related risks": {
                "Supply Chain Disruptions": 180,  # 6 months
                "Market Risks": 365               # 1 year
            }
        }
        
    def _init_impact_multipliers(self) -> Dict[str, float]:
        """Initialize scenario-specific impact multipliers"""
        if self.scenario.name == "Delayed Transition":
            return {
                "Physical Risks": 1.5,
                "Transition Risks": 2.0,
                "Nature-related risks": 1.8,
                "Supply Chain Disruptions": 2.0,
                "Market Risks": 1.5,
                "Financial System Risks": 1.7
            }
        else:  # Net Zero 2050
            return {
                "Physical Risks": 1.0,
                "Transition Risks": 1.2,
                "Nature-related risks": 1.1,
                "Supply Chain Disruptions": 1.0,
                "Market Risks": 1.0,
                "Financial System Risks": 1.0
            }
            
    def calculate_propagation_strength(
        self,
        source_risk: Risk,
        target_risk: Risk,
        base_strength: float
    ) -> float:
        """Calculate scenario-adjusted propagation strength"""
        # Apply category-specific multipliers
        source_multiplier = self.impact_multipliers.get(source_risk.category, 1.0)
        target_multiplier = self.impact_multipliers.get(target_risk.category, 1.0)
        
        # Calculate propagation delay
        delay = self.get_propagation_delay(source_risk, target_risk)
        
        # Apply time decay to strength
        time_decay = np.exp(-0.1 * delay / 365.0)  # Yearly decay rate
        
        return base_strength * source_multiplier * target_multiplier * time_decay
        
    def get_propagation_delay(self, source_risk: Risk, target_risk: Risk) -> int:
        """Get scenario-adjusted propagation delay between risks"""
        base_delay = self.propagation_delays.get(
            source_risk.category, {}
        ).get(target_risk.category, 30)  # Default 30 days
        
        # Adjust delay based on scenario
        if self.scenario.name == "Delayed Transition":
            return int(base_delay * 0.7)  # 30% faster propagation
        return base_delay