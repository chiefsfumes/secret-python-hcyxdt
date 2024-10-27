from typing import List, Dict, Tuple
from src.models import Risk
import numpy as np

def suggest_mitigation_strategies(risks: List[Risk], scenario_impacts: Dict[str, List[Tuple[Risk, float]]],
                                  simulation_results: Dict[str, Dict[int, List[float]]]) -> Dict[int, List[str]]:
    mitigation_strategies = {}
    
    for risk in risks:
        strategies = []
        
        # Analyze scenario impacts
        max_impact_scenario = max(scenario_impacts.items(), key=lambda x: next((impact for r, impact in x[1] if r.id == risk.id), 0))
        
        if max_impact_scenario[0] == "Net Zero 2050":
            strategies.append("Accelerate transition to low-carbon technologies")
        elif max_impact_scenario[0] == "Delayed Transition":
            strategies.append("Prepare for abrupt policy changes and market shifts")
        elif max_impact_scenario[0] == "Current Policies":
            strategies.append("Enhance resilience to physical climate risks")
        
        # Analyze Monte Carlo simulation results
        risk_simulation = {scenario: results[risk.id] for scenario, results in simulation_results.items()}
        high_variability_scenarios = [scenario for scenario, results in risk_simulation.items() if np.std(results) > 0.5]
        
        if high_variability_scenarios:
            strategies.append(f"Develop flexible strategies to address high uncertainty in {', '.join(high_variability_scenarios)} scenarios")
        
        # Category-specific strategies
        if risk.category == "Physical Risk":
            strategies.append("Invest in climate-resilient infrastructure and operations")
        elif risk.category == "Transition Risk":
            strategies.append("Diversify product/service portfolio to align with low-carbon economy")
        elif risk.category == "Market Risk":
            strategies.append("Monitor and adapt to changing consumer preferences and market dynamics")
        elif risk.category == "Policy Risk":
            strategies.append("Engage in policy discussions and prepare for various regulatory scenarios")
        elif risk.category == "Reputation Risk":
            strategies.append("Enhance sustainability reporting and stakeholder communication")
        
        mitigation_strategies[risk.id] = strategies
    
    return mitigation_strategies