from typing import List, Dict, Tuple
from src.config import NUM_SIMULATIONS, LLM_MODEL, LLM_API_KEY, COMPANY_INFO, SCENARIOS
from src.models import Risk, ExternalData, Scenario, SimulationResult, Entity
from src.prompts import RISK_ASSESSMENT_PROMPT
from src.risk_analysis.time_series_analysis import calculate_physical_acute_impact, calculate_physical_chronic_impact, calculate_policy_legal_impact, calculate_technology_impact, calculate_market_impact, calculate_reputation_impact, project_nature_risk, project_systemic_risk, calculate_default_impact
import openai
import numpy as np
from scipy.stats import norm
import logging

openai.api_key = LLM_API_KEY
logger = logging.getLogger(__name__)

def simulate_scenario_impact(risks: List[Risk], external_data: Dict[str, ExternalData], scenarios: Dict[str, Scenario], entities: List[Entity]) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Enhanced scenario impact simulation using sophisticated time series analysis."""
    impacts = {scenario_name: {entity.name: {} for entity in entities} for scenario_name in scenarios}
    
    for scenario_name, scenario in scenarios.items():
        for entity in entities:
            for risk in risks:
                try:
                    # Use category-specific time series analysis based on risk type
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
                    
                    # Apply scenario-specific adjustments
                    adjusted_forecast = adjust_forecast_for_scenario(forecast, scenario, risk)
                    impacts[scenario_name][entity.name][risk.id] = adjusted_forecast
                    
                except Exception as e:
                    logger.error(f"Error in scenario impact simulation for risk {risk.id}: {str(e)}")
                    impacts[scenario_name][entity.name][risk.id] = risk.impact
    
    return impacts

def adjust_forecast_for_scenario(forecast: List[float], scenario: Scenario, risk: Risk) -> List[float]:
    """Apply scenario-specific adjustments to the forecast."""
    adjusted_forecast = forecast.copy()
    
    # Apply temperature impact
    temp_factor = 1 + (scenario.temp_increase - 1.5) * 0.1
    
    # Apply carbon price impact
    carbon_price_factor = 1 + (scenario.carbon_price / 100) * 0.05
    
    # Apply renewable energy transition impact
    renewable_factor = 1 - scenario.renewable_energy * 0.2
    
    # Apply category-specific scenario adjustments
    if risk.category == "Physical Risks":
        adjusted_forecast = [f * temp_factor * 1.2 for f in adjusted_forecast]  # More sensitive to temperature
    elif risk.category == "Transition Risks":
        adjusted_forecast = [f * carbon_price_factor * renewable_factor for f in adjusted_forecast]
    elif risk.category == "Nature-related risks":
        biodiversity_factor = 1 + scenario.biodiversity_loss * 0.3
        ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.25
        adjusted_forecast = [f * biodiversity_factor * ecosystem_factor for f in adjusted_forecast]
    elif risk.category == "Systemic Risks":
        financial_factor = 1 + (1 - scenario.financial_stability) * 0.2
        supply_chain_factor = 1 + scenario.supply_chain_disruption * 0.15
        adjusted_forecast = [f * financial_factor * supply_chain_factor for f in adjusted_forecast]
    
    return adjusted_forecast

def monte_carlo_simulation(
    risks: List[Risk],
    external_data: Dict[str, ExternalData],
    scenarios: Dict[str, Scenario],
    entities: List[Entity],
    num_simulations: int = 10000
) -> Dict[str, Dict[str, Dict[int, SimulationResult]]]:
    results = {}
    for scenario_name, scenario in scenarios.items():
        scenario_results = {entity.name: {} for entity in entities}
        for risk in risks:
            entity = next((e for e in entities if e.name == risk.entity), None)
            if entity:
                impact_distribution = []
                likelihood_distribution = []
                for _ in range(num_simulations):
                    perturbed_scenario = perturb_scenario(scenario)
                    impact = calculate_risk_impact(risk, external_data, perturbed_scenario, entity)
                    likelihood = calculate_risk_likelihood(risk, external_data, perturbed_scenario, entity)
                    impact_distribution.append(impact)
                    likelihood_distribution.append(likelihood)
                scenario_results[entity.name][risk.id] = SimulationResult(
                    risk_id=risk.id,
                    scenario=scenario_name,
                    impact_distribution=impact_distribution,
                    likelihood_distribution=likelihood_distribution
                )
        results[scenario_name] = scenario_results
    return results

def calculate_risk_impact(risk: Risk, external_data: Dict[str, ExternalData], scenario: Scenario, entity: Entity) -> float:
    # Implement a more sophisticated impact calculation considering entity context
    base_impact = risk.impact
    temp_factor = 1 + (scenario.temp_increase - 1.5) * 0.1
    carbon_price_factor = 1 + (scenario.carbon_price / 100) * 0.05
    renewable_factor = 1 - scenario.renewable_energy * 0.2
    
    # Consider entity-specific factors
    entity_factor = 1.0
    if "Manufacturing" in entity.key_products:
        entity_factor *= 1.2  # Manufacturing might be more impacted
    elif "Services" in entity.key_products:
        entity_factor *= 0.9  # Services might be less impacted
    
    impact = base_impact * temp_factor * carbon_price_factor * renewable_factor * entity_factor
    return min(1.0, max(0.0, impact))

def calculate_risk_likelihood(risk: Risk, external_data: Dict[str, ExternalData], scenario: Scenario, entity: Entity) -> float:
    # Implement a more sophisticated likelihood calculation considering entity context
    base_likelihood = risk.likelihood
    policy_factor = 1 - scenario.policy_stringency * 0.3
    ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.4
    
    # Consider entity-specific factors
    entity_factor = 1.0
    if "Global" in entity.region:
        entity_factor *= 1.1  # Global entities might be more exposed to risks
    elif len(entity.region) == 1:
        entity_factor *= 0.9  # Local entities might be less exposed to global risks
    
    likelihood = base_likelihood * policy_factor * ecosystem_factor * entity_factor
    return min(1.0, max(0.0, likelihood))

def perturb_scenario(scenario: Scenario, perturbations: Dict[str, float] = None) -> Scenario:
    """
    Perturbs the given scenario based on default random noise or specific perturbations.

    Args:
        scenario (Scenario): The original scenario to perturb.
        perturbations (Dict[str, float], optional): A dictionary where keys are scenario attributes
            to perturb and values are the percentage changes to apply. Defaults to random perturbations.

    Returns:
        Scenario: A new Scenario object with perturbed values.
    """
    perturbed_values = {}
    try:
        scenario_dict = scenario.dict()
    except AttributeError:
        logger.warning("'dict()' method not available, using custom_dict() method")
        scenario_dict = scenario.custom_dict()

    if perturbations:
        for field, pct_change in perturbations.items():
            if field in scenario_dict and isinstance(scenario_dict[field], (int, float)):
                new_value = scenario_dict[field] * (1 + pct_change)
                perturbed_values[field] = max(0, new_value)
            else:
                logger.warning(f"Field '{field}' not found or not numeric in scenario. Skipping perturbation for this field.")
    else:
        for field, value in scenario_dict.items():
            if field not in ['name', 'description', 'time_horizon'] and isinstance(value, (int, float)):
                perturbed_values[field] = max(0, value * np.random.normal(1, 0.1))  # 10% standard deviation

    return Scenario(
        name=scenario.name,
        description=scenario.description,
        time_horizon=scenario.time_horizon,
        **perturbed_values
    )

def analyze_sensitivity(
    monte_carlo_results: Dict[str, Dict[int, SimulationResult]] = None,
    base_scenario: Scenario = None,
    variable: str = None,
    range_pct: float = None,
    entities: List[Entity] = None,
    risks: List[Risk] = None,
    external_data: Dict[str, ExternalData] = None
) -> Dict[str, Dict[str, float]]:
    logger.info("Starting sensitivity analysis")
    
    # Check which analysis type we're performing
    if monte_carlo_results is not None:
        logger.info("Performing Monte Carlo sensitivity analysis")
        return analyze_monte_carlo_sensitivity(monte_carlo_results)
    
    # Check if we have all required parameters for scenario variable sensitivity
    required_params = {
        'base_scenario': base_scenario,
        'variable': variable,
        'range_pct': range_pct,
        'entities': entities,
        'risks': risks,
        'external_data': external_data
    }
    
    missing_params = [param for param, value in required_params.items() if value is None]
    
    if missing_params:
        logger.error(f"Missing required parameters for scenario variable sensitivity: {', '.join(missing_params)}")
        raise ValueError(
            f"Missing required parameters for scenario variable sensitivity: {', '.join(missing_params)}. "
            "Provide either monte_carlo_results or all parameters for scenario variable sensitivity."
        )
    
    logger.info("Performing scenario variable sensitivity analysis")
    return analyze_scenario_variable_sensitivity(
        base_scenario=base_scenario,
        variable=variable,
        range_pct=range_pct,
        entities=entities,
        risks=risks,
        external_data=external_data
    )

def analyze_monte_carlo_sensitivity(monte_carlo_results: Dict[str, Dict[int, SimulationResult]]) -> Dict[str, Dict[str, float]]:
    logger.info("Starting Monte Carlo sensitivity analysis")
    
    if not monte_carlo_results:
        logger.warning("Empty monte_carlo_results provided")
        return {}
        
    sensitivity_results = {}
    
    for scenario, risk_results in monte_carlo_results.items():
        logger.debug(f"Analyzing scenario: {scenario}")
        scenario_sensitivity = {}
        
        for risk_id, simulation_result in risk_results.items():
            try:
                impact_mean = np.mean(simulation_result.impact_distribution)
                impact_std = np.std(simulation_result.impact_distribution)
                likelihood_mean = np.mean(simulation_result.likelihood_distribution)
                likelihood_std = np.std(simulation_result.likelihood_distribution)
                
                # Avoid division by zero
                impact_sensitivity = impact_std / impact_mean if impact_mean != 0 else 0
                likelihood_sensitivity = likelihood_std / likelihood_mean if likelihood_mean != 0 else 0
                
                scenario_sensitivity[f"risk_{risk_id}_impact"] = impact_sensitivity
                scenario_sensitivity[f"risk_{risk_id}_likelihood"] = likelihood_sensitivity
                
                logger.debug(f"Risk {risk_id} sensitivities - Impact: {impact_sensitivity:.3f}, Likelihood: {likelihood_sensitivity:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating sensitivity for risk {risk_id}: {str(e)}")
                scenario_sensitivity[f"risk_{risk_id}_impact"] = 0
                scenario_sensitivity[f"risk_{risk_id}_likelihood"] = 0
        
        sensitivity_results[scenario] = scenario_sensitivity
    
    return sensitivity_results

def analyze_scenario_variable_sensitivity(
    base_scenario: Scenario,
    variable: str,
    range_pct: float,
    entities: List[Entity],
    risks: List[Risk],
    external_data: Dict[str, ExternalData]
) -> Dict[str, Dict[str, float]]:
    logger.info(f"Starting scenario variable sensitivity analysis for variable: {variable}")
    
    if not hasattr(base_scenario, variable):
        logger.error(f"Variable {variable} not found in scenario attributes")
        raise ValueError(f"Variable {variable} not found in scenario attributes")
        
    sensitivity_results = {}
    base_value = getattr(base_scenario, variable)
    low_value = base_value * (1 - range_pct)
    high_value = base_value * (1 + range_pct)
    
    logger.debug(f"Testing sensitivity range: {low_value} to {high_value}")
    
    for entity in entities:
        logger.debug(f"Analyzing entity: {entity.name}")
        entity_sensitivity = {}
        
        # Create low and high scenarios
        low_scenario = perturb_scenario(base_scenario, {variable: -range_pct})
        high_scenario = perturb_scenario(base_scenario, {variable: range_pct})
        
        # Remove entity filtering - analyze all risks for each entity
        logger.debug(f"Analyzing {len(risks)} risks for entity {entity.name}")
        
        for risk in risks:
            try:
                low_impact = calculate_risk_impact(risk, external_data, low_scenario, entity)
                high_impact = calculate_risk_impact(risk, external_data, high_scenario, entity)
                
                # Calculate sensitivity
                sensitivity = (high_impact - low_impact) / (high_value - low_value) if high_value != low_value else 0
                entity_sensitivity[f"risk_{risk.id}_{variable}"] = sensitivity
                
                logger.debug(f"Risk {risk.id} sensitivity to {variable}: {sensitivity:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating sensitivity for risk {risk.id}: {str(e)}")
                entity_sensitivity[f"risk_{risk.id}_{variable}"] = 0
        
        sensitivity_results[entity.name] = entity_sensitivity
    
    return sensitivity_results


def calculate_var_cvar(simulation_results: Dict[str, Dict[str, Dict[int, SimulationResult]]], confidence_level: float = 0.95) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    var_cvar_results = {}
    for scenario, entities in simulation_results.items():
        var_cvar_results[scenario] = {}
        for entity_name, risks in entities.items():
            var_cvar_results[scenario][entity_name] = {}
            for risk_id, result in risks.items():
                impacts = np.array(result.impact_distribution)
                var = np.percentile(impacts, (1 - confidence_level) * 100)
                cvar = np.mean(impacts[impacts > var])
                var_cvar_results[scenario][entity_name][risk_id] = {
                    "VaR": var,
                    "CVaR": cvar
                }
    return var_cvar_results

def perform_stress_testing(risks: List[Risk], scenarios: Dict[str, Scenario], external_data: Dict[str, ExternalData], entities: List[Entity]) -> Dict[str, Dict[str, List[Tuple[Risk, float]]]]:
    stress_test_results = {}
    for scenario_name, scenario in scenarios.items():
        # Create a stressed scenario
        stressed_scenario = Scenario(
            name=f"Stressed_{scenario_name}",
            temp_increase=scenario.temp_increase * 1.5,
            carbon_price=scenario.carbon_price * 2,
            renewable_energy=scenario.renewable_energy * 0.5,
            policy_stringency=scenario.policy_stringency * 0.5,
            biodiversity_loss=min(1.0, scenario.biodiversity_loss * 1.5),
            ecosystem_degradation=min(1.0, scenario.ecosystem_degradation * 1.5),
            financial_stability=scenario.financial_stability * 0.5,
            supply_chain_disruption=min(1.0, scenario.supply_chain_disruption * 2)
        )
        
        # Simulate impacts under stressed conditions
        stress_test_results[scenario_name] = simulate_scenario_impact(risks, external_data, stressed_scenario, entities)
    
    return stress_test_results

def generate_scenario_narratives(scenarios: Dict[str, Scenario], entities: List[Entity]) -> Dict[str, Dict[str, str]]:
    narratives = {}
    for scenario_name, scenario in scenarios.items():
        narratives[scenario_name] = {}
        for entity in entities:
            prompt = f"""
            Generate a detailed narrative for the following climate scenario, considering the context of {COMPANY_INFO.name}, a {COMPANY_INFO.industry} company, specifically for the {entity.name} entity:

            Company: {COMPANY_INFO.name}
            Industry: {COMPANY_INFO.industry}
            Region: {COMPANY_INFO.region}
            Key Products: {', '.join(COMPANY_INFO.key_products)}

            Entity: {entity.name}
            Description: {entity.description}
            Key Products: {', '.join(entity.key_products)}
            Region: {', '.join(entity.region)}

            Scenario: {scenario_name}
            - Temperature increase: {scenario.temp_increase}°C
            - Carbon price: ${scenario.carbon_price}/ton
            - Renewable energy share: {scenario.renewable_energy * 100}%
            - Policy stringency: {scenario.policy_stringency * 100}%
            - Biodiversity loss: {scenario.biodiversity_loss * 100}%
            - Ecosystem degradation: {scenario.ecosystem_degradation * 100}%
            - Financial stability: {scenario.financial_stability * 100}%
            - Supply chain disruption: {scenario.supply_chain_disruption * 100}%

            Provide a compelling narrative that describes the overall state of the world in this scenario, including key challenges and opportunities for the {entity.name} entity of {COMPANY_INFO.name}, major societal and environmental changes, potential technological advancements or setbacks, and the general economic landscape. Consider how this scenario might specifically impact the {entity.name} entity's operations, supply chain, and market position.
            """

            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in climate scenario analysis and futurism."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            narratives[scenario_name][entity.name] = response.choices[0].message['content']
    
    return narratives

