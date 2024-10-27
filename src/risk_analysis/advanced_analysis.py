from typing import List, Dict, Any
from src.models import Risk, Scenario, Company, PESTELAnalysis, SystemicRisk, SimulationResult, Entity, ExternalData  # Add ExternalData
from src.config import (
    LLM_MODEL, 
    LLM_API_KEY, 
    SCENARIOS, 
    COMPANY_INFO,
    TIME_SERIES_HORIZON,  # Add this to config.py
    NUM_SIMULATIONS  # Add this to config.py
)
from src.prompts import (
    RISK_NARRATIVE_PROMPT, 
    EXECUTIVE_INSIGHTS_PROMPT, 
    SYSTEMIC_RISK_PROMPT, 
    MITIGATION_STRATEGY_PROMPT, 
    PESTEL_ANALYSIS_PROMPT, 
    RISK_ASSESSMENT_PROMPT
)
from src.risk_analysis.time_series_analysis import (
    calculate_category_specific_impacts,
    project_nature_risk,
    project_systemic_risk,
    analyze_impact_trends,  # Add this import
    identify_critical_periods,  # Add this import
    forecast_cumulative_impact  # Add this import
)
from src.risk_analysis.scenario_analysis import simulate_scenario_impact, monte_carlo_simulation, calculate_risk_impact, calculate_risk_likelihood, analyze_sensitivity
from src.sensitivity_analysis.monte_carlo import perform_monte_carlo_simulations
import numpy as np
import re
from src.risk_analysis.pestel_analysis import perform_pestel_analysis
from src.risk_analysis.systemic_risk_analysis import analyze_systemic_risks, identify_trigger_points, assess_resilience
from src.risk_analysis.interaction_analysis import analyze_risk_interactions, build_risk_network, create_risk_interaction_matrix, simulate_risk_interactions
from src.utils.llm_util import get_llm_response  # Added
import logging
import json
import os
from src.risk_analysis.scenario_analysis import (
    simulate_scenario_impact,
    monte_carlo_simulation,
    calculate_risk_impact,
    calculate_risk_likelihood,
    analyze_sensitivity,
    calculate_var_cvar
)

logger = logging.getLogger(__name__)

def conduct_advanced_risk_analysis(risks: List[Risk], scenarios: Dict[str, Scenario], company_info: Company, external_data: Dict) -> Dict:
    """Enhanced advanced risk analysis incorporating sophisticated time series analysis."""
    logger.info("Starting advanced risk analysis with enhanced time series capabilities")
    
    entities = list(company_info.entities.values())
    
    # Use category-specific time series projections
    time_series_results = {}
    for entity in entities:
        entity_risks = {}
        for risk in risks:
            try:
                if risk.category == "Nature-related risks":
                    projection = project_nature_risk(risk, external_data)
                elif risk.category == "Systemic Risks":
                    projection = project_systemic_risk(risk, external_data)
                else:
                    # Use appropriate category-specific calculation
                    projection = calculate_category_specific_impacts(risk, external_data)
                entity_risks[risk.id] = projection
            except Exception as e:
                logger.error(f"Error in time series projection for risk {risk.id}: {str(e)}")
                entity_risks[risk.id] = [risk.impact] * TIME_SERIES_HORIZON
        time_series_results[entity.name] = entity_risks
    
    # Enhanced scenario impact simulation
    scenario_impacts = simulate_scenario_impact(risks, external_data, scenarios, entities)
    
    # Monte Carlo simulation with enhanced time series
    monte_carlo_results = perform_monte_carlo_simulations(
        risks=risks,
        scenarios=scenarios,
        external_data=external_data,
        entities=entities
    )
    
    # Analyze trends and critical periods using enhanced time series
    impact_trends = analyze_impact_trends(time_series_results)
    critical_periods = identify_critical_periods(time_series_results, threshold=0.7)
    cumulative_impact = forecast_cumulative_impact(time_series_results)
    
    # Calculate VaR and CVaR using the Monte Carlo results
    risk_metrics = calculate_var_cvar(monte_carlo_results)
    
    # Perform sensitivity analysis
    sensitivity_results = analyze_sensitivity(
        monte_carlo_results=monte_carlo_results
    )
    
    comprehensive_analysis = {}
    for scenario_name, scenario in scenarios.items():
        logger.info(f"Analyzing scenario: {scenario_name}")
        scenario_analysis = {}
        
        for entity_name, entity in company_info.entities.items():
            logger.info(f"Analyzing entity: {entity_name} for scenario: {scenario_name}")
            
            entity_analysis = {}
            risk_scores = []
            
            # Remove this filtering - we want to assess all risks for each entity
            # entity_specific_risks = [r for r in risks if r.entity == entity_name]
            # logger.info(f"Found {len(entity_specific_risks)} risks specific to entity {entity_name}")
            # logger.debug(f"Entity-specific risks: {[{'id': r.id, 'description': r.description} for r in entity_specific_risks]}")
            
            # Analyze all risks for this entity
            logger.info(f"Assessing all {len(risks)} risks for entity {entity_name}")
            logger.debug(f"Risks being assessed: {[{'id': r.id, 'description': r.description} for r in risks]}")
            
            # Analyze how each risk affects this entity
            for risk in risks:
                logger.debug(f"Assessing impact of risk {risk.id} (from entity {risk.entity}) on entity: {entity_name}")
                
                try:
                    # Assess how this risk affects this specific entity in this scenario
                    assessment = llm_risk_assessment(risk, scenario, company_info, entity)
                    entity_analysis[risk.id] = assessment
                    
                    # Calculate risk score with more detailed logging
                    likelihood_impact_change = assessment.get('likelihood_impact_change', {})
                    likelihood_change = get_change_value(likelihood_impact_change.get('likelihood_change', 'No Change'))
                    impact_change = get_change_value(likelihood_impact_change.get('impact_change', 'No Change'))
                    risk_score = likelihood_change * impact_change
                    
                    logger.debug(f"Calculated score for risk {risk.id}: likelihood_change={likelihood_change}, impact_change={impact_change}, final_score={risk_score}")
                    
                    # Store the score with more context
                    risk_scores.append({
                        'risk_id': risk.id,
                        'score': risk_score,
                        'original_entity': risk.entity,
                        'assessed_entity': entity_name,
                        'likelihood_change': likelihood_change,
                        'impact_change': impact_change
                    })
                except Exception as e:
                    logger.error(f"Error assessing risk {risk.id} for entity {entity_name}: {str(e)}")
                    logger.exception("Detailed error information:")
            
            logger.info(f"Completed {len(risk_scores)} risk assessments for entity {entity_name}")
            
            # Calculate summary statistics with more detailed logging
            if risk_scores:
                scores_only = [r['score'] for r in risk_scores]
                logger.debug(f"Risk scores for {entity_name}: {scores_only}")
                
                entity_analysis['summary'] = {
                    "average_risk_score": np.mean(scores_only),
                    "max_risk_score": max(scores_only),
                    "risk_score_volatility": np.std(scores_only),
                    "num_risks_assessed": len(risk_scores),
                    "risk_details": risk_scores  # Store detailed risk information
                }
                
                logger.info(f"Summary statistics for {entity_name}: avg={entity_analysis['summary']['average_risk_score']:.2f}, max={entity_analysis['summary']['max_risk_score']:.2f}, volatility={entity_analysis['summary']['risk_score_volatility']:.2f}")
            else:
                logger.error(f"No risk scores calculated for entity {entity_name} in scenario {scenario_name}. This might indicate a problem with risk assessment or filtering.")
                entity_analysis['summary'] = {
                    "average_risk_score": None,
                    "max_risk_score": None,
                    "risk_score_volatility": None,
                    "num_risks_assessed": 0,
                    "risk_details": []
                }
            
            scenario_analysis[entity_name] = entity_analysis
        
        comprehensive_analysis[scenario_name] = scenario_analysis

    # Generate executive insights
    logger.info("Generating executive insights")
    executive_insights = generate_executive_insights(comprehensive_analysis, risks, company_info)
    comprehensive_analysis['executive_insights'] = executive_insights

    return comprehensive_analysis

def llm_risk_assessment(risk: Risk, scenario: Scenario, company_info: Company, entity: Entity) -> Dict[str, Any]:
    # Create checkpoint directory with company name
    checkpoint_dir = f"checkpoints/{company_info.name.lower().replace(' ', '_')}"
    checkpoint_key = f"{risk.id}_{scenario.name}_{entity.name}"
    checkpoint_file = f"{checkpoint_dir}/risk_assessment_{checkpoint_key}.json"
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint for {checkpoint_key}")
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint for {checkpoint_key}: {str(e)}")
            # Continue with normal processing if checkpoint loading fails
    
    # Update the prompt to clarify we're assessing cross-entity impact
    prompt = RISK_ASSESSMENT_PROMPT.format(
        risk_id=risk.id,
        industry=company_info.industry,
        company_name=company_info.name,
        company_region=company_info.region,
        key_products=', '.join(company_info.key_products),
        risk_description=risk.description,
        risk_category=risk.category,
        risk_subcategory=risk.subcategory,
        risk_likelihood=risk.likelihood,
        risk_impact=risk.impact,
        risk_time_horizon=risk.time_horizon,
        risk_original_entity=risk.entity,  # Add this to provide context about the risk's original entity
        scenario_name=scenario.name,
        temp_increase=scenario.temp_increase,
        carbon_price=scenario.carbon_price,
        renewable_energy=scenario.renewable_energy,
        policy_stringency=scenario.policy_stringency,
        biodiversity_loss=scenario.biodiversity_loss,
        ecosystem_degradation=scenario.ecosystem_degradation,
        financial_stability=scenario.financial_stability,
        supply_chain_disruption=scenario.supply_chain_disruption,
        entity_name=entity.name,
        entity_description=entity.description,
        entity_key_products=', '.join(entity.key_products),
        entity_region=', '.join(entity.region)
    )
    
    logger.info(f"Sending prompt to LLM for risk assessment:\n{json.dumps({'prompt': prompt}, indent=2)}")

    system_message = "You are an expert in climate risk assessment."

    try:
        parsed_response = get_llm_response(prompt, system_message)
        logger.info(f"Parsed LLM response:\n{json.dumps(parsed_response, indent=2)}")
        
        # Ensure the expected structure is present
        if 'likelihood_impact_change' not in parsed_response:
            parsed_response['likelihood_impact_change'] = {
                'likelihood_change': 'No Change',
                'impact_change': 'No Change',
                'explanation': 'No explanation provided by LLM'
            }
        
        # Save checkpoint in company-specific directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(parsed_response, f)
            logger.info(f"Saved checkpoint for {checkpoint_key}")
        except Exception as e:
            logger.error(f"Error saving checkpoint for {checkpoint_key}: {str(e)}")
        
        return parsed_response
    except Exception as e:
        logger.error(f"Error in LLM risk assessment for risk {risk.id}: {str(e)}")
        return {"error": str(e), "likelihood_impact_change": {"likelihood_change": "No Change", "impact_change": "No Change"}}

def parse_llm_response(content: str) -> Dict[str, Any]:
    """
    Parses the JSON response from the LLM.

    Args:
        content (str): The JSON string returned by the LLM.

    Returns:
        Dict[str, Any]: The parsed JSON as a Python dictionary.

    Raises:
        ValueError: If the content is not valid JSON.
    """
    try:
        # Remove any leading/trailing backticks and whitespace
        cleaned_content = content.strip('`').strip()
        # Remove "json" if it appears at the start of the content
        if cleaned_content.startswith('json'):
            cleaned_content = cleaned_content[4:].lstrip()
        parsed_response = json.loads(cleaned_content)
        return parsed_response
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
        logger.debug(f"LLM response content: {content}")
        raise ValueError("Invalid JSON format received from LLM.") from e

def assess_aggregate_impact(risks: List[Risk], interaction_matrix: np.ndarray, external_data: Dict[str, ExternalData] = None) -> Dict[str, float]:
    """Enhanced aggregate impact assessment using sophisticated time series analysis."""
    try:
        n = len(risks)
        base_impacts = np.array([risk.impact for risk in risks])
        
        # Calculate category-specific impacts if external data is available
        if external_data:
            category_impacts = []
            for risk in risks:
                impacts = calculate_category_specific_impacts(risk, external_data)
                category_impacts.append(np.mean(impacts))  # Use mean impact from time series
            base_impacts = np.array(category_impacts)
        
        aggregate_impacts = []
        for _ in range(NUM_SIMULATIONS):
            risk_levels = np.random.beta(2, 2, n) * base_impacts
            for _ in range(10):  # Simulate interactions for 10 time steps
                influence = interaction_matrix @ risk_levels
                risk_levels = np.clip(risk_levels + 0.1 * influence, 0, 1)
            aggregate_impacts.append(np.sum(risk_levels))
        
        return {
            "mean": np.mean(aggregate_impacts),
            "median": np.median(aggregate_impacts),
            "95th_percentile": np.percentile(aggregate_impacts, 95),
            "max": np.max(aggregate_impacts)
        }
    except Exception as e:
        logger.error(f"Error in aggregate impact assessment: {str(e)}")
        return {"mean": np.mean(base_impacts), "error": str(e)}

def identify_tipping_points(risks: List[Risk], interaction_matrix: np.ndarray) -> List[Dict[str, Any]]:
    n = len(risks)
    base_impacts = np.array([risk.impact for risk in risks])
    tipping_points = []

    for i in range(n):
        impact_levels = np.linspace(0, 1, 100)
        aggregate_impacts = []
        for level in impact_levels:
            risk_levels = base_impacts.copy()
            risk_levels[i] = level
            for _ in range(10):  # Simulate interactions for 10 time steps
                influence = interaction_matrix @ risk_levels
                risk_levels = np.clip(risk_levels + 0.1 * influence, 0, 1)
            aggregate_impacts.append(np.sum(risk_levels))
        
        # Detect sudden changes in the rate of change
        rate_of_change = np.diff(aggregate_impacts)
        threshold = np.mean(rate_of_change) + 2 * np.std(rate_of_change)
        tipping_point_indices = np.where(rate_of_change > threshold)[0]
        
        if len(tipping_point_indices) > 0:
            tipping_points.append({
                "risk_id": risks[i].id,
                "risk_description": risks[i].description,
                "tipping_point_level": impact_levels[tipping_point_indices[0]],
                "aggregate_impact": aggregate_impacts[tipping_point_indices[0]]
            })
    
    return tipping_points

def generate_risk_narratives(risks: List[Risk], comprehensive_analysis: Dict[str, Dict[str, Dict[int, str]]], company_info: Company) -> Dict[int, str]:
    risk_narratives = {}
    for risk in risks:
        entity = next((e for e in company_info.entities.values() if e.name == risk.entity), None)
        scenario_analyses = "\n\n".join([
            f"Scenario: {scenario}\n" + 
            "\n".join([
                f"Entity: {entity_name}\nAnalysis: {analysis[risk.id]}" 
                for entity_name, analysis in scenario_analysis.items() if risk.id in analysis
            ]) 
            for scenario, scenario_analysis in comprehensive_analysis.items()
        ])
        
        prompt = RISK_NARRATIVE_PROMPT.format(
            risk_id=risk.id,
            risk_description=risk.description,
            company_name=company_info.name,
            industry=company_info.industry,
            company_region=company_info.region,
            key_products=', '.join(company_info.key_products),
            entity_name=entity.name if entity else "",
            entity_description=entity.description if entity else "",
            entity_key_products=', '.join(entity.key_products) if entity else "",
            entity_region=', '.join(entity.region) if entity else "",
            scenario_analyses=scenario_analyses
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Sending prompt to LLM for risk narrative:\n{prompt}")

        system_message = "You are an expert in climate risk assessment and scenario analysis."
        response_content = get_llm_response(prompt, system_message)
        
        logger.info(f"Received response from LLM:\n{response_content}")
        
        risk_narratives[risk.id] = response_content

    return risk_narratives

def generate_executive_insights(comprehensive_analysis: Dict[str, Dict[str, Dict[int, str]]], risks: List[Risk], company_info: Company) -> Dict[str, str]:
    entity_insights = {}
    for entity_name, entity in company_info.entities.items():
        entity_risks = [risk for risk in risks if risk.entity == entity_name]
        entity_analyses = "\n\n".join([
            f"Risk: {risk.description}\n" + 
            "\n".join([
                f"{scenario}: {scenario_analysis[entity_name][risk.id]}" 
                for scenario, scenario_analysis in comprehensive_analysis.items() 
                if entity_name in scenario_analysis and risk.id in scenario_analysis[entity_name]
            ]) 
            for risk in entity_risks
        ])

        prompt = EXECUTIVE_INSIGHTS_PROMPT.format(
            company_name=company_info.name,
            industry=company_info.industry,
            company_region=company_info.region,
            key_products=', '.join(company_info.key_products),
            entity_name=entity.name,
            entity_description=entity.description,
            entity_key_products=', '.join(entity.key_products),
            entity_region=', '.join(entity.region),
            all_analyses=entity_analyses
        )

        system_message = "You are a senior climate risk analyst providing insights to top executives."
        response_content = get_llm_response(prompt, system_message)
        entity_insights[entity_name] = response_content

    # Generate overall company insights
    all_analyses = "\n\n".join([
        f"Entity: {entity_name}\n{insights}" for entity_name, insights in entity_insights.items()
    ])
    
    overall_prompt = EXECUTIVE_INSIGHTS_PROMPT.format(
        company_name=company_info.name,
        industry=company_info.industry,
        company_region=company_info.region,
        key_products=', '.join(company_info.key_products),
        entity_name="Overall Company",
        entity_description="Entire company analysis",
        entity_key_products=', '.join(company_info.key_products),
        entity_region=', '.join(company_info.region),
        all_analyses=all_analyses
    )

    overall_insights = get_llm_response(overall_prompt, system_message)
    
    entity_insights['overall'] = overall_insights
    return entity_insights

def perform_cross_scenario_analysis(comprehensive_analysis: Dict[str, Dict[str, Dict[int, str]]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    cross_scenario_results = {}
    for risk_id in comprehensive_analysis[next(iter(comprehensive_analysis))].keys():
        risk_results = {}
        for scenario, scenarios in comprehensive_analysis.items():
            analysis = scenarios[risk_id]
            impact_score = analysis.get('impact', 0.5)  # Default value if not present
            likelihood_score = analysis.get('likelihood', 0.5)
            adaptability_score = analysis.get('adaptability', 0.5)
            
            risk_results[scenario] = {
                "impact": impact_score,
                "likelihood": likelihood_score,
                "adaptability": adaptability_score
            }
        cross_scenario_results[risk_id] = risk_results
    return cross_scenario_results

def identify_key_uncertainties(cross_scenario_results: Dict[int, Dict[str, Dict[str, float]]]) -> List[int]:
    uncertainties = []
    for risk_id, scenarios in cross_scenario_results.items():
        impact_variance = np.var([s['impact'] for s in scenarios.values()])
        likelihood_variance = np.var([s['likelihood'] for s in scenarios.values()])
        if impact_variance > 0.1 or likelihood_variance > 0.1:  # Threshold for high uncertainty
            uncertainties.append(risk_id)
    return uncertainties

def generate_mitigation_strategies(risks: List[Risk], comprehensive_analysis: Dict[str, Dict[str, Dict[int, str]]], company_info: Company) -> Dict[int, List[str]]:
    mitigation_strategies = {}
    for risk in risks:
        entity = next((e for e in company_info.entities.values() if e.name == risk.entity), None)
        scenario_analyses = "\n".join([
            f"Scenario: {scenario}\nEntity: {risk.entity}\nAnalysis: {scenario_analysis[entity_name][risk.id]}" 
            for scenario, scenario_analysis in comprehensive_analysis.items() 
            for entity_name in scenario_analysis 
            if risk.id in scenario_analysis[entity_name]
        ])
        
        prompt = MITIGATION_STRATEGY_PROMPT.format(
            company_name=company_info.name,
            industry=company_info.industry,
            company_region=company_info.region,
            key_products=', '.join(company_info.key_products),
            risk_id=risk.id,
            risk_description=risk.description,
            risk_category=risk.category,
            risk_subcategory=risk.subcategory,
            entity_name=entity.name if entity else "",
            entity_description=entity.description if entity else "",
            entity_key_products=', '.join(entity.key_products) if entity else "",
            entity_region=', '.join(entity.region) if entity else "",
            scenario_analyses=scenario_analyses
        )
        
        system_message = "You are an expert in climate risk mitigation and adaptation strategies."
        response_content = get_llm_response(prompt, system_message)
        
        strategies = response_content.get('mitigation_strategies', [])
        mitigation_strategies[risk.id] = strategies
    
    return mitigation_strategies

def get_change_value(change: str) -> float:
    change_values = {
        'Significant Decrease': 0.5,
        'Moderate Decrease': 0.75,
        'Slight Decrease': 0.9,
        'No Change': 1.0,
        'Slight Increase': 1.1,
        'Moderate Increase': 1.25,
        'Significant Increase': 1.5
    }
    return change_values.get(change, 1.0)


