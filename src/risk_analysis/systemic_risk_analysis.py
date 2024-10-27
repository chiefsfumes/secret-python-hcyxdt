from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
from src.models import Risk, ExternalData, SimulationResult, Company
import logging

# Keep existing functions
logger = logging.getLogger(__name__)

def identify_trigger_points(risks: List[Risk], risk_network: Dict, external_data: Dict[str, ExternalData]) -> Dict[int, Dict]:
    trigger_points = {}
    
    # Convert dictionary to NetworkX graph if needed
    if isinstance(risk_network, dict):
        G = nx.Graph()
        for node, neighbors in risk_network.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)
        risk_network = G
    
    try:
        centrality = nx.betweenness_centrality(risk_network, weight='weight')
        
        for risk in risks:
            if centrality[risk.id] > np.mean(list(centrality.values())):
                # Get neighbors and their weights using NetworkX methods
                neighbors = list(risk_network.neighbors(risk.id))
                total_weight = sum(risk_network[risk.id][neighbor]['weight'] 
                                 for neighbor in neighbors)
                
                if total_weight > 0.5 * len(neighbors):  # More than half of max possible weight
                    trigger_points[risk.id] = {
                        "description": risk.description,
                        "centrality": centrality[risk.id],
                        "connected_risks": neighbors,
                        "total_interaction_weight": total_weight,
                        "external_factors": identify_relevant_external_factors(risk, external_data)
                    }
    except Exception as e:
        logger.error(f"Error in identify_trigger_points: {str(e)}")
        # Return empty dict if analysis fails
        return {}
    
    return trigger_points

def assess_system_resilience(risks: List[Risk], risk_network: nx.Graph, scenario_impacts: Dict[str, List[Tuple[Risk, float]]]) -> Dict[str, float]:
    logger.info("Starting system resilience assessment")
    logger.debug(f"Number of risks provided: {len(risks)}")
    logger.debug(f"Risk network nodes: {risk_network.number_of_nodes()}")
    logger.debug(f"Scenarios received: {list(scenario_impacts.keys())}")
    
    resilience_metrics = {}
    
    # Validate inputs
    if not risks:
        logger.warning("No risks provided for resilience assessment")
        return {"error": "No risks provided"}
        
    if not risk_network or risk_network.number_of_nodes() == 0:
        logger.warning("Empty risk network provided")
        return {"error": "Empty risk network"}
        
    # Network-based resilience metrics with error handling
    try:
        # Density is always calculable
        resilience_metrics["network_density"] = nx.density(risk_network)
        logger.debug(f"Network density: {resilience_metrics['network_density']}")
        
        # Average clustering requires at least one node with neighbors
        if risk_network.number_of_edges() > 0:
            resilience_metrics["average_clustering"] = nx.average_clustering(risk_network, weight='weight')
            logger.debug(f"Average clustering: {resilience_metrics['average_clustering']}")
        else:
            resilience_metrics["average_clustering"] = 0
            logger.warning("No edges in network, setting average_clustering to 0")
        
        # Assortativity requires at least one edge
        if risk_network.number_of_edges() > 0:
            resilience_metrics["assortativity"] = nx.degree_assortativity_coefficient(risk_network, weight='weight')
            logger.debug(f"Assortativity: {resilience_metrics['assortativity']}")
        else:
            resilience_metrics["assortativity"] = 0
            logger.warning("No edges in network, setting assortativity to 0")
            
    except Exception as e:
        logger.error(f"Error calculating network metrics: {str(e)}")
        resilience_metrics.update({
            "network_density": 0,
            "average_clustering": 0,
            "assortativity": 0
        })
    
    # Impact-based resilience metrics
    try:
        for scenario, impacts in scenario_impacts.items():
            logger.debug(f"Processing scenario: {scenario}")
            logger.debug(f"Raw impacts data for {scenario}: {impacts}")
            impact_values = []
            
            # Handle nested dictionary structure
            for entity, risk_impacts in impacts.items():
                logger.debug(f"Processing entity: {entity} with impacts: {risk_impacts}")
                # Extract values from the inner dictionary
                for risk_id, impact_value in risk_impacts.items():
                    logger.debug(f"Processing risk_id: {risk_id} with impact: {impact_value}")
                    if isinstance(impact_value, (int, float)):
                        impact_values.append(impact_value)
            
            logger.debug(f"Collected impact values for {scenario}: {impact_values}")
            
            if impact_values:
                mean_impact = np.mean(impact_values)
                logger.info(f"{scenario} - Mean impact: {mean_impact}")
                
                if mean_impact != 0:
                    impact_dispersion = np.std(impact_values) / mean_impact
                    logger.info(f"{scenario} - Impact dispersion (std/mean): {impact_dispersion}")
                    resilience_metrics[f"{scenario}_impact_dispersion"] = impact_dispersion
                else:
                    logger.warning(f"{scenario} - Mean impact is 0, setting impact_dispersion to 0")
                    resilience_metrics[f"{scenario}_impact_dispersion"] = 0
                
                max_impact = max(impact_values)
                logger.info(f"{scenario} - Maximum impact: {max_impact}")
                resilience_metrics[f"{scenario}_max_impact"] = max_impact
                
                logger.info(f"{scenario} - Metrics calculated successfully")
            else:
                logger.warning(f"No valid impact values for scenario {scenario}")
                resilience_metrics[f"{scenario}_impact_dispersion"] = 0
                resilience_metrics[f"{scenario}_max_impact"] = 0
                
    except Exception as e:
        logger.error(f"Error calculating impact-based metrics: {str(e)}", exc_info=True)
        for scenario in scenario_impacts.keys():
            resilience_metrics.update({
                f"{scenario}_impact_dispersion": 0,
                f"{scenario}_max_impact": 0
            })
    
    # Adaptive capacity metric
    try:
        valid_impacts = [risk.impact for risk in risks if risk.impact is not None]
        logger.debug(f"Valid impacts for adaptive capacity: {valid_impacts}")
        if valid_impacts:
            resilience_metrics["adaptive_capacity"] = 1 - np.mean(valid_impacts)
            logger.debug(f"Adaptive capacity: {resilience_metrics['adaptive_capacity']}")
        else:
            resilience_metrics["adaptive_capacity"] = 0
            logger.warning("No valid impact values for adaptive capacity calculation")
            
    except Exception as e:
        logger.error(f"Error calculating adaptive capacity: {str(e)}")
        resilience_metrics["adaptive_capacity"] = 0
    
    logger.info("System resilience assessment completed")
    logger.debug(f"Final resilience metrics: {resilience_metrics}")
    return resilience_metrics


def analyze_systemic_risks(risks_by_entity: Dict[str, List[Risk]], company: Company) -> Dict[str, Dict[str, Dict]]:
    systemic_risks_by_entity = {}
    for entity, risks in risks_by_entity.items():
        systemic_risks = {}
        for risk in risks:
            if is_systemic_risk(risk, company.industry, company.key_dependencies):
                systemic_risks[risk.id] = {
                    "description": risk.description,
                    "impact": risk.impact,
                    "systemic_factor": identify_systemic_factor(risk)
                }
        systemic_risks_by_entity[entity] = systemic_risks
    return systemic_risks_by_entity


def assess_resilience(risks: List[Risk], scenario_impacts: Dict[str, Dict[int, float]], simulation_results: Dict[str, Dict[int, SimulationResult]]) -> Dict[str, float]:
    resilience_scores = {}
    for scenario, impacts in scenario_impacts.items():
        scenario_resilience = calculate_scenario_resilience(impacts, simulation_results[scenario])
        resilience_scores[scenario] = scenario_resilience
    return resilience_scores

def is_systemic_risk(risk: Risk, company_industry: str, key_dependencies: List[str]) -> bool:
    systemic_keywords = ["market-wide", "industry-wide", "global", "systemic", "interconnected"]
    return any(keyword in risk.description.lower() for keyword in systemic_keywords) or \
           any(dep.lower() in risk.description.lower() for dep in key_dependencies)

def identify_systemic_factor(risk: Risk) -> str:
    if "financial" in risk.description.lower():
        return "Financial System"
    elif "supply chain" in risk.description.lower():
        return "Supply Chain"
    elif "geopolitical" in risk.description.lower():
        return "Geopolitical"
    else:
        return "Other"

def identify_relevant_external_factors(risk: Risk, external_data: Dict[str, ExternalData]) -> List[str]:
    relevant_factors = []
    latest_year = max(external_data.keys())
    if "economic" in risk.description.lower():
        relevant_factors.append(f"GDP Growth: {external_data[latest_year].gdp_growth}%")
    if "population" in risk.description.lower():
        relevant_factors.append(f"Population: {external_data[latest_year].population}")
    return relevant_factors

def calculate_scenario_resilience(impacts: Dict[int, float], simulation_results: Dict[int, SimulationResult]) -> float:
    total_impact = sum(impacts.values())
    variance = sum(np.var(result.impact_distribution) for result in simulation_results.values())
    return 1 / (total_impact * (1 + variance))  # Higher resilience for lower impact and lower variance

def analyze_risk_cascades(risk_network: nx.Graph, initial_risks: List[int], threshold: float = 0.5, max_steps: int = 10) -> Dict[int, List[float]]:
    cascade_progression = {risk: [1.0] for risk in initial_risks}
    for _ in range(max_steps):
        new_activations = {}
        for node in risk_network.nodes():
            if node not in cascade_progression:
                neighbor_influence = sum(cascade_progression.get(neighbor, [0])[-1] * risk_network[node][neighbor]['weight']
                                         for neighbor in risk_network.neighbors(node))
                if neighbor_influence > threshold:
                    new_activations[node] = neighbor_influence
        
        if not new_activations:
            break
        
        for node, activation in new_activations.items():
            cascade_progression[node] = [0.0] * (len(next(iter(cascade_progression.values()))) - 1) + [activation]
        
        for progression in cascade_progression.values():
            progression.append(progression[-1])
    
    return cascade_progression

def identify_risk_feedback_loops(risk_network: nx.Graph) -> List[List[int]]:
    feedback_loops = list(nx.simple_cycles(risk_network))
    return [loop for loop in feedback_loops if len(loop) > 2]

def assess_network_resilience(risk_network: nx.Graph) -> Dict[str, float]:
    resilience_metrics = {
        "average_clustering": nx.average_clustering(risk_network),
        "average_shortest_path_length": nx.average_shortest_path_length(risk_network, weight='weight'),
        "graph_density": nx.density(risk_network),
        "assortativity": nx.degree_assortativity_coefficient(risk_network)
    }
    return resilience_metrics




