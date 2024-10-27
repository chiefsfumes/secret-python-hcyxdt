from typing import Dict, List
from src.models import Risk, Scenario
from src.risk_analysis.cascade.analyzer import CascadeAnalyzer
from src.risk_analysis.cascade.business_impact import BusinessMetrics

def example_cascade_analysis():
    # Example risks
    extreme_weather = Risk(
        id=1,
        description="Extreme weather events",
        category="Physical Risks",
        subcategory="Acute Physical Risks",
        likelihood=0.8,
        impact=0.7
    )
    
    supply_chain = Risk(
        id=2,
        description="Supply chain disruption",
        category="Systemic Risks",
        subcategory="Supply Chain Disruptions",
        likelihood=0.6,
        impact=0.8
    )
    
    market_share = Risk(
        id=3,
        description="Market share loss",
        category="Transition Risks",
        subcategory="Market Risks",
        likelihood=0.4,
        impact=0.6
    )
    
    # Example correlation matrix
    correlations = {
        (1, 2): 0.8,  # Strong correlation between extreme weather and supply chain
        (2, 3): 0.6,  # Moderate correlation between supply chain and market share
        (1, 3): 0.4   # Weak correlation between extreme weather and market share
    }
    
    # Example scenarios
    net_zero = Scenario(
        name="Net Zero 2050",
        description="Net zero emissions achieved by 2050",
        time_horizon="2050",
        temp_increase=1.5,
        carbon_price=250,
        renewable_energy=0.75,
        policy_stringency=0.9,
        biodiversity_loss=0.1,
        ecosystem_degradation=0.2,
        financial_stability=0.8,
        supply_chain_disruption=0.3
    )
    
    delayed = Scenario(
        name="Delayed Transition",
        description="Delayed climate action",
        time_horizon="2050",
        temp_increase=2.5,
        carbon_price=125,
        renewable_energy=0.55,
        policy_stringency=0.6,
        biodiversity_loss=0.3,
        ecosystem_degradation=0.4,
        financial_stability=0.6,
        supply_chain_disruption=0.5
    )
    
    # Analyze cascades under different scenarios
    risks = [extreme_weather, supply_chain, market_share]
    risk_network = {1: {2: 0.8, 3: 0.4}, 2: {3: 0.6}}
    
    # Net Zero 2050 analysis
    nz_analyzer = CascadeAnalyzer(risks, risk_network, net_zero, correlations)
    nz_cascade = nz_analyzer.analyze_cascade(extreme_weather)
    
    print("\nNet Zero 2050 Cascade:")
    print_cascade_results(nz_cascade)
    
    # Delayed Transition analysis
    dt_analyzer = CascadeAnalyzer(risks, risk_network, delayed, correlations)
    dt_cascade = dt_analyzer.analyze_cascade(extreme_weather)
    
    print("\nDelayed Transition Cascade:")
    print_cascade_results(dt_cascade)

def print_cascade_results(cascade_node, level=0):
    """Helper function to print cascade results"""
    indent = "  " * level
    total_impact = cascade_node.get_total_impact()
    
    print(f"{indent}Risk {cascade_node.risk_id}:")
    print(f"{indent}  Revenue Impact: {total_impact.revenue:.2%}")
    print(f"{indent}  Cost Impact: {total_impact.costs:.2%}")
    print(f"{indent}  Market Share Impact: {total_impact.market_share:.2%}")
    print(f"{indent}  Time to Effect: {cascade_node.business_impact.time_to_effect} days")
    print(f"{indent}  Duration: {cascade_node.business_impact.duration} days")
    
    for child in cascade_node.children:
        print_cascade_results(child, level + 1)

if __name__ == "__main__":
    example_cascade_analysis()