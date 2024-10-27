import numpy as np
from typing import List, Dict, Optional
from src.models import Risk, Scenario, SimulationResult, ExternalData, Entity
from src.risk_analysis.time_series_analysis import (
    calculate_physical_acute_impact,
    calculate_physical_chronic_impact,
    calculate_policy_legal_impact,
    calculate_technology_impact,
    calculate_market_impact,
    calculate_reputation_impact,
    project_nature_risk,
    project_systemic_risk,
    calculate_default_impact
)
import logging

logger = logging.getLogger(__name__)

def perform_monte_carlo_simulations(
    risks: List[Risk], 
    scenarios: Dict[str, Scenario], 
    external_data: Dict[str, ExternalData],
    entities: List[Entity],
    num_simulations: int = 10000
) -> Dict[str, Dict[str, Dict[int, SimulationResult]]]:
    """
    Perform Monte Carlo simulations incorporating time series analysis and entity-specific impacts.
    
    Returns:
        Dict[scenario_name][entity_name][risk_id] = SimulationResult
    """
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        logger.info(f"Running Monte Carlo simulation for scenario: {scenario_name}")
        scenario_results = {entity.name: {} for entity in entities}
        
        for entity in entities:
            logger.debug(f"Processing entity: {entity.name}")
            
            for risk in risks:
                try:
                    impact_distribution = []
                    likelihood_distribution = []
                    time_series_distributions = []
                    
                    for _ in range(num_simulations):
                        # Perturb both scenario and external data
                        perturbed_scenario = perturb_scenario(scenario)
                        perturbed_external_data = perturb_external_data(external_data)
                        
                        # Get time series forecast based on risk category
                        time_series = get_category_specific_forecast(
                            risk, 
                            perturbed_external_data, 
                            perturbed_scenario,
                            entity
                        )
                        
                        # Calculate impact and likelihood using time series
                        impact = calculate_risk_impact(
                            risk, 
                            perturbed_scenario, 
                            time_series,
                            entity
                        )
                        likelihood = calculate_risk_likelihood(
                            risk, 
                            perturbed_scenario, 
                            time_series,
                            entity
                        )
                        
                        impact_distribution.append(impact)
                        likelihood_distribution.append(likelihood)
                        time_series_distributions.append(time_series)
                    
                    scenario_results[entity.name][risk.id] = SimulationResult(
                        risk_id=risk.id,
                        scenario=scenario_name,
                        impact_distribution=impact_distribution,
                        likelihood_distribution=likelihood_distribution,
                        time_series_distributions=time_series_distributions  # Add this to SimulationResult model
                    )
                    
                except Exception as e:
                    logger.error(f"Error in Monte Carlo simulation for risk {risk.id}: {str(e)}")
                    
        results[scenario_name] = scenario_results
    
    return results

def get_category_specific_forecast(
    risk: Risk, 
    external_data: Dict[str, ExternalData],
    scenario: Scenario,
    entity: Entity
) -> List[float]:
    """Get appropriate time series forecast based on risk category."""
    try:
        if risk.category == "Physical Risks":
            if risk.subcategory == "Acute Physical Risks":
                forecast = calculate_physical_acute_impact(risk, external_data)
            else:  # Chronic Physical Risks
                forecast = calculate_physical_chronic_impact(risk, external_data)
        elif risk.category == "Transition Risks":
            if risk.subcategory == "Policy and Legal Risks":
                forecast = calculate_policy_legal_impact(risk, external_data)
            elif risk.subcategory == "Technology Risks":
                forecast = calculate_technology_impact(risk, external_data)
            elif risk.subcategory == "Market Risks":
                forecast = calculate_market_impact(risk, external_data)
            else:  # Reputation Risks
                forecast = calculate_reputation_impact(risk, external_data)
        elif risk.category == "Nature-related risks":
            forecast = project_nature_risk(risk, external_data)
        elif risk.category == "Systemic Risks":
            forecast = project_systemic_risk(risk, external_data)
        else:
            forecast = calculate_default_impact(risk, external_data)
        
        return forecast
    except Exception as e:
        logger.error(f"Error in category-specific forecast for risk {risk.id}: {str(e)}")
        return [risk.impact] * len(next(iter(external_data.values())).values())

def perturb_external_data(external_data: Dict[str, ExternalData]) -> Dict[str, ExternalData]:
    """Create perturbed version of external data for Monte Carlo simulation."""
    perturbed_data = {}
    for year, data in external_data.items():
        perturbed_values = {}
        for field, value in data.model_dump().items():
            if isinstance(value, (int, float)):
                # Add noise while maintaining trends
                perturbed_values[field] = max(0, value * np.random.normal(1, 0.05))
        perturbed_data[year] = ExternalData(**perturbed_values)
    return perturbed_data

def calculate_risk_impact(
    risk: Risk, 
    scenario: Scenario, 
    time_series: List[float],
    entity: Entity
) -> float:
    """Calculate risk impact using time series data and entity context."""
    # Use the mean of the time series as the base impact
    base_impact = np.mean(time_series)
    
    # Apply scenario factors
    temp_factor = 1 + (scenario.temp_increase - 1.5) * 0.1
    carbon_price_factor = 1 + (scenario.carbon_price / 100) * 0.05
    renewable_factor = 1 - scenario.renewable_energy * 0.2
    
    # Apply entity-specific factors
    entity_factor = 1.0
    if "Manufacturing" in entity.key_products:
        entity_factor *= 1.2  # Manufacturing might be more impacted
    elif "Services" in entity.key_products:
        entity_factor *= 0.9  # Services might be less impacted
    
    # Consider geographical factors
    if "Global" in entity.region:
        entity_factor *= 1.1  # Global entities might be more exposed
    elif len(entity.region) == 1:
        entity_factor *= 0.9  # Local entities might be less exposed
    
    impact = base_impact * temp_factor * carbon_price_factor * renewable_factor * entity_factor
    return min(1.0, max(0.0, impact))

def calculate_risk_likelihood(
    risk: Risk, 
    scenario: Scenario, 
    time_series: List[float],
    entity: Entity
) -> float:
    """Calculate risk likelihood using time series data and entity context."""
    # Use the trend in the time series to influence likelihood
    trend_factor = 1.0
    if len(time_series) > 1:
        trend = np.polyfit(range(len(time_series)), time_series, 1)[0]
        trend_factor = 1 + trend
    
    base_likelihood = risk.likelihood * trend_factor
    
    # Apply scenario factors
    policy_factor = 1 - scenario.policy_stringency * 0.3
    ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.4
    financial_factor = 1 + (1 - scenario.financial_stability) * 0.2
    
    # Apply entity-specific factors
    entity_factor = 1.0
    if entity.industry:
        if entity.industry.lower() in ["energy", "manufacturing"]:
            entity_factor *= 1.15
        elif entity.industry.lower() in ["technology", "services"]:
            entity_factor *= 0.85
    
    likelihood = base_likelihood * policy_factor * ecosystem_factor * financial_factor * entity_factor
    return min(1.0, max(0.0, likelihood))

def perturb_scenario(scenario: Scenario, perturbation_factor: float = 0.05) -> Scenario:
    """
    Create a perturbed version of the scenario for Monte Carlo simulation.
    
    Args:
        scenario: Original scenario
        perturbation_factor: Standard deviation for the normal distribution used in perturbation
    
    Returns:
        Perturbed scenario
    """
    try:
        # Create a copy of the scenario's data
        perturbed_data = scenario.model_dump()
        
        # Perturb numerical values while maintaining valid ranges
        for field, value in perturbed_data.items():
            if isinstance(value, (int, float)) and field not in ['name', 'description', 'time_horizon']:
                # Add random noise using normal distribution
                noise = np.random.normal(1, perturbation_factor)
                perturbed_value = value * noise
                
                # Ensure values stay within valid ranges (0 to 1 for most fields)
                if field in ['renewable_energy', 'policy_stringency', 'biodiversity_loss', 
                           'ecosystem_degradation', 'financial_stability', 'supply_chain_disruption']:
                    perturbed_data[field] = min(1.0, max(0.0, perturbed_value))
                elif field == 'temp_increase':
                    # Temperature increase should stay positive
                    perturbed_data[field] = max(0.0, perturbed_value)
                elif field == 'carbon_price':
                    # Carbon price should stay positive
                    perturbed_data[field] = max(0.0, perturbed_value)
                else:
                    perturbed_data[field] = perturbed_value
        
        # Create new scenario with perturbed values
        return Scenario(**perturbed_data)
        
    except Exception as e:
        logger.error(f"Error in scenario perturbation: {str(e)}")
        return scenario  # Return original scenario if perturbation fails
