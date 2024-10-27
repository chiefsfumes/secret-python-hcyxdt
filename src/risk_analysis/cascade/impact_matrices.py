from typing import Dict, List, Tuple
import numpy as np

class BusinessImpactMatrices:
    def __init__(self):
        # Revenue impact matrix by risk category (% impact on revenue)
        self.revenue_matrix = {
            "Physical Risks": {
                "Acute Physical Risks": {
                    "direct": -0.15,      # 15% revenue drop
                    "duration": 90,        # 90 days
                    "recovery_rate": 0.02  # 2% recovery per day
                },
                "Chronic Physical Risks": {
                    "direct": -0.08,
                    "duration": 365,
                    "recovery_rate": 0.005
                }
            },
            "Transition Risks": {
                "Policy and Legal Risks": {
                    "direct": -0.05,
                    "duration": 180,
                    "recovery_rate": 0.01
                },
                "Market Risks": {
                    "direct": -0.12,
                    "duration": 120,
                    "recovery_rate": 0.015
                }
            },
            "Nature-related risks": {
                "Biodiversity Loss": {
                    "direct": -0.07,
                    "duration": 730,
                    "recovery_rate": 0.003
                }
            },
            "Systemic Risks": {
                "Supply Chain Disruptions": {
                    "direct": -0.20,
                    "duration": 60,
                    "recovery_rate": 0.025
                }
            }
        }

        # Operating margin impact matrix (percentage points)
        self.margin_matrix = {
            "Physical Risks": {
                "Acute Physical Risks": -3.5,
                "Chronic Physical Risks": -2.0
            },
            "Transition Risks": {
                "Policy and Legal Risks": -2.5,
                "Market Risks": -1.5
            },
            "Nature-related risks": {
                "Biodiversity Loss": -1.0
            },
            "Systemic Risks": {
                "Supply Chain Disruptions": -4.0
            }
        }

        # Market share impact matrix (percentage points)
        self.market_share_matrix = {
            "Physical Risks": {
                "Acute Physical Risks": -2.0,
                "Chronic Physical Risks": -1.0
            },
            "Transition Risks": {
                "Policy and Legal Risks": -0.5,
                "Market Risks": -2.5
            },
            "Nature-related risks": {
                "Biodiversity Loss": -0.3
            },
            "Systemic Risks": {
                "Supply Chain Disruptions": -1.5
            }
        }

        # Interaction multipliers for cascading effects
        self.interaction_multipliers = {
            ("Physical Risks", "Supply Chain Disruptions"): 1.5,
            ("Supply Chain Disruptions", "Market Risks"): 1.3,
            ("Market Risks", "Financial System Risks"): 1.2
        }

        # Scenario adjustment factors
        self.scenario_adjustments = {
            "Net Zero 2050": {
                "Physical Risks": 0.8,      # 20% less severe
                "Transition Risks": 1.2,    # 20% more severe initially
                "Nature-related risks": 0.9
            },
            "Delayed Transition": {
                "Physical Risks": 1.4,      # 40% more severe
                "Transition Risks": 1.6,    # 60% more severe
                "Nature-related risks": 1.3
            }
        }

    def get_revenue_impact(self, risk_category: str, risk_subcategory: str, 
                          scenario: str, impact_score: float) -> Dict:
        """Calculate revenue impact based on risk category and scenario"""
        base_impact = self.revenue_matrix[risk_category][risk_subcategory]
        scenario_mult = self.scenario_adjustments[scenario][risk_category]
        
        return {
            "impact_pct": base_impact["direct"] * impact_score * scenario_mult,
            "duration": base_impact["duration"],
            "recovery_rate": base_impact["recovery_rate"]
        }

    def get_margin_impact(self, risk_category: str, risk_subcategory: str,
                         scenario: str, impact_score: float) -> float:
        """Calculate margin impact in percentage points"""
        base_impact = self.margin_matrix[risk_category][risk_subcategory]
        scenario_mult = self.scenario_adjustments[scenario][risk_category]
        return base_impact * impact_score * scenario_mult

    def get_market_share_impact(self, risk_category: str, risk_subcategory: str,
                              scenario: str, impact_score: float) -> float:
        """Calculate market share impact in percentage points"""
        base_impact = self.market_share_matrix[risk_category][risk_subcategory]
        scenario_mult = self.scenario_adjustments[scenario][risk_category]
        return base_impact * impact_score * scenario_mult

    def get_interaction_multiplier(self, source_category: str, 
                                 target_category: str) -> float:
        """Get multiplier for cascading effects between risk categories"""
        return self.interaction_multipliers.get(
            (source_category, target_category), 
            1.0  # Default to no amplification
        )