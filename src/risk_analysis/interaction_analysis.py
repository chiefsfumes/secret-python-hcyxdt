from typing import List, Dict, Tuple
from src.models import Risk, RiskInteraction, Entity, Company
from src.config import LLM_MODEL, COMPANY_INFO
from src.prompts import INTERACTION_ANALYSIS_PROMPT, RISK_INTERACTION_SUMMARY_PROMPT
from src.utils.llm_util import get_llm_response
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import logging
import json
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def analyze_risk_interactions(risks: List[Risk], company: Company) -> Dict[str, List[RiskInteraction]]:
    """Analyze risk interactions once at company level, then map to entities."""
    logger.info(f"Starting risk interaction analysis for {len(risks)} risks")
    
    # First, analyze all risk interactions once at company level
    checkpoint_dir = Path('checkpoints') / sanitize_company_name(company.name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "risk_interactions.json"
    
    # Get or compute base interactions
    base_interactions = _get_or_compute_base_interactions(risks, company, checkpoint_file)
    
    # Then map these interactions to entities
    entity_interactions = {}
    for entity_name, entity in company.entities.items():
        entity_risks = [r for r in risks if r.entity == entity_name]
        # Only include interactions relevant to this entity's risks
        entity_interactions[entity_name] = _map_interactions_to_entity(
            base_interactions, 
            entity_risks
        )
    
    return entity_interactions

def _get_or_compute_base_interactions(risks: List[Risk], company: Company, checkpoint_file: Path) -> List[RiskInteraction]:
    """Compute or load cached risk interactions at company level."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                existing_interactions = json.load(f)
            logger.info(f"Loaded {len(existing_interactions)} existing interactions from checkpoint")
            return [_create_risk_interaction(
                next(r for r in risks if r.id == int(pair.split('_')[0])),
                next(r for r in risks if r.id == int(pair.split('_')[1])),
                data
            ) for pair, data in existing_interactions.items()]
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {str(e)}")
    
    logger.info(f"Computing interactions for {len(risks)} risks")
    interactions = {}
    for i, risk1 in enumerate(risks):
        for j, risk2 in enumerate(risks[i+1:], start=i+1):
            logger.debug(f"Analyzing interaction between Risk {risk1.id} and Risk {risk2.id}")
            interaction = analyze_single_interaction(risk1, risk2)
            
            # Add validation and logging for interaction result
            if interaction and interaction.full_analysis:
                interactions[f"{risk1.id}_{risk2.id}"] = interaction.full_analysis
                logger.debug(f"Successfully analyzed interaction {risk1.id}_{risk2.id}")
            else:
                logger.warning(f"No valid analysis generated for interaction {risk1.id}_{risk2.id}")
    
    # Log interaction data before saving
    logger.debug(f"Interaction data to be saved: {json.dumps(interactions, indent=2)}")
    
    # Save checkpoint
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(interactions, f, indent=2)
        logger.info(f"Saved {len(interactions)} interactions to checkpoint")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
    
    return [_create_risk_interaction(
        next(r for r in risks if r.id == int(pair.split('_')[0])),
        next(r for r in risks if r.id == int(pair.split('_')[1])),
        data
    ) for pair, data in interactions.items()]

def _map_interactions_to_entity(base_interactions: List[RiskInteraction], entity_risks: List[Risk]) -> List[RiskInteraction]:
    """Map relevant interactions to entity context."""
    entity_risk_ids = {r.id for r in entity_risks}
    return [
        interaction for interaction in base_interactions
        if interaction.risk1_id in entity_risk_ids or interaction.risk2_id in entity_risk_ids
    ]

def sanitize_company_name(name: str) -> str:
    """Convert company name to lowercase, replace spaces/special chars with underscores."""
    # Remove special characters and convert to lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', name).lower()
    # Replace spaces with underscores
    return re.sub(r'\s+', '_', sanitized)

def _analyze_entity_interactions(risks: List[Risk], company: Company, entity_name: str) -> List[RiskInteraction]:
    """Analyze interactions between risks within a specific entity."""
    logger.debug(f"Analyzing {len(risks)} risks for entity: {entity_name}")
    interactions = []
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints') / sanitize_company_name(company.name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint file path
    checkpoint_file = checkpoint_dir / f"{sanitize_company_name(entity_name)}_interactions.json"
    
    # Load existing checkpoints if they exist
    existing_interactions = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                existing_interactions = json.load(f)
            logger.info(f"Loaded {len(existing_interactions)} existing interactions from checkpoint")
        except json.JSONDecodeError:
            logger.warning(f"Failed to load checkpoint file {checkpoint_file}, starting fresh")
    
    for i, risk1 in enumerate(risks):
        for j, risk2 in enumerate(risks[i+1:], start=i+1):
            interaction_key = f"{risk1.id}_{risk2.id}"
            logger.debug(f"Analyzing interaction between Risk {risk1.id} and Risk {risk2.id}")
            
            # Check if we already have this interaction
            if interaction_key in existing_interactions:
                logger.debug(f"Using cached interaction for {interaction_key}")
                interaction_data = existing_interactions[interaction_key]
            else:
                prompt = INTERACTION_ANALYSIS_PROMPT.format(
                    company_name=company.name,
                    company_industry=company.industry,
                    company_region=", ".join(company.region),
                    key_products=", ".join(company.key_products),
                    risk1_id=risk1.id,
                    risk2_id=risk2.id,
                    risk1_description=risk1.description,
                    risk1_category=risk1.category or "Unknown",
                    risk1_subcategory=risk1.subcategory or "Unknown",
                    risk2_description=risk2.description,
                    risk2_category=risk2.category or "Unknown",
                    risk2_subcategory=risk2.subcategory or "Unknown",
                    entity_name=entity_name,
                )
                
                response = get_llm_response(prompt, "You are an expert in climate risk assessment and risk interactions.")
                
                if not response:
                    logger.warning(f"Failed to get valid LLM response for interaction between Risk {risk1.id} and Risk {risk2.id}")
                    interaction_data = {
                        "interaction_score": {"score": 0.5},
                        "interaction_explanation": "Unable to parse LLM response",
                        "compounding_effects": [],
                        "mitigating_factors": []
                    }
                else:
                    interaction_data = response
                    # Store new interaction in checkpoint
                    existing_interactions[interaction_key] = interaction_data
                    
                    # Save checkpoint after each new interaction
                    try:
                        with open(checkpoint_file, 'w') as f:
                            json.dump(existing_interactions, f, indent=2)
                        logger.debug(f"Saved interaction checkpoint for {interaction_key}")
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint: {str(e)}")
            
            interaction = _create_risk_interaction(risk1, risk2, interaction_data)
            interactions.append(interaction)
    
    return interactions

def _aggregate_child_interactions(
    parent_entity: str,
    entity_interactions: Dict[str, List[RiskInteraction]],
    company: Company,
    parent_weight: float
) -> List[RiskInteraction]:
    """Aggregate interactions from child entities to parent level."""
    logger.debug(f"Aggregating interactions for parent entity: {parent_entity}")
    children = company.get_entity(parent_entity).sub_entities
    
    aggregated_interactions = []
    for child in children:
        child_entity = company.get_entity(child)
        child_weight = child_entity.weight
        weighted_interactions = [
            RiskInteraction(
                risk1_id=interaction.risk1_id,
                risk2_id=interaction.risk2_id,
                interaction_score=interaction.interaction_score * child_weight * parent_weight,
                interaction_type=interaction.interaction_type
            )
            for interaction in entity_interactions.get(child, [])
        ]
        aggregated_interactions.extend(weighted_interactions)
    
    return aggregated_interactions

def build_risk_network(
    risks: List[Risk],
    entity_interactions: Dict[str, List[RiskInteraction]],
    company: Company
) -> nx.Graph:
    """Build a single company-wide risk network."""
    logger.info("Building company-wide risk network")
    G = nx.Graph()
    
    # Add nodes (risks)
    for risk in risks:
        G.add_node(risk.id, **{
            'id': risk.id,
            'description': risk.description,
            'category': risk.category,
            'subcategory': risk.subcategory,
            'likelihood': risk.likelihood,
            'impact': risk.impact,
            'time_horizon': risk.time_horizon,
            'entity': risk.entity
        })
    
    # Add edges (interactions) from all entities
    for interactions in entity_interactions.values():
        for interaction in interactions:
            G.add_edge(
                interaction.risk1_id,
                interaction.risk2_id,
                weight=interaction.interaction_score,
                type=interaction.interaction_type,
                explanation=interaction.interaction_explanation,
                compounding_effects=interaction.compounding_effects,
                mitigating_factors=interaction.mitigating_factors
            )
    
    logger.debug(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def identify_central_risks(G: nx.Graph) -> Dict[int, float]:
    """Identify central risks in the company-wide network."""
    logger.info("Identifying central risks in company network")
    
    if len(G) == 0:
        logger.warning("Empty network provided")
        return {}
        
    try:
        centrality_measures = {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G, weight='weight'),
            "eigenvector": nx.eigenvector_centrality(G, weight='weight'),
            "pagerank": nx.pagerank(G, weight='weight')
        }
        
        combined_centrality = {}
        for node in G.nodes():
            measures = [measure[node] for measure in centrality_measures.values()]
            combined_centrality[node] = np.mean(measures)
        
        return combined_centrality
        
    except Exception as e:
        logger.error(f"Error computing centrality measures: {str(e)}")
        return {node: 0.0 for node in G.nodes()}

def detect_risk_clusters(G: nx.Graph, num_clusters: int = 3) -> Dict:
    """Detect risk clusters in the company-wide network."""
    logger.info("Detecting risk clusters in company network")
    
    if len(G) < num_clusters:
        logger.warning(f"Not enough nodes for clustering (nodes: {len(G)}, clusters: {num_clusters})")
        return {'clusters': [], 'cluster_sizes': {}, 'error': 'Not enough nodes'}
        
    try:
        adj_matrix = nx.to_numpy_array(G)
        kmeans = KMeans(n_clusters=min(num_clusters, len(G)), random_state=42)
        cluster_labels = kmeans.fit_predict(adj_matrix)
        
        clusters = {}
        for node, label in zip(G.nodes(), cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        
        return {
            'clusters': clusters,
            'cluster_sizes': {i: len(nodes) for i, nodes in clusters.items()},
            'cluster_assignments': {node: label for node, label in zip(G.nodes(), cluster_labels)}
        }
        
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        return {'clusters': [], 'cluster_sizes': {}, 'error': str(e)}

# Helper function to create risk interaction
def _create_risk_interaction(risk1: Risk, risk2: Risk, interaction_data: dict) -> RiskInteraction:
    # Add null check and provide default data
    if interaction_data is None:
        interaction_data = {
            'interaction_score': {'score': 0.5},
            'interaction_explanation': 'No interaction data available',
            'compounding_effects': [],
            'mitigating_factors': []
        }
        logger.warning(f"No interaction data provided for risks {risk1.id} and {risk2.id}. Using default values.")
    
    interaction_score = interaction_data.get('interaction_score', {}).get('score', 0.5)
    interaction_type = determine_interaction_type(interaction_score)
    
    interaction = RiskInteraction(risk1.id, risk2.id, interaction_score, interaction_type)
    interaction.full_analysis = interaction_data
    interaction.interaction_explanation = interaction_data.get('interaction_explanation')
    interaction.compounding_effects = interaction_data.get('compounding_effects', [])
    interaction.mitigating_factors = interaction_data.get('mitigating_factors', [])
    
    return interaction

# Keep existing helper functions
def determine_interaction_type(score: float) -> str:
    if score < 0.3:
        return "Weak"
    elif score < 0.7:
        return "Moderate"
    else:
        return "Strong"

def extract_interaction_score(analysis: str) -> float:
    import re
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", analysis)
    return float(numbers[-1]) if numbers else 0.5

def analyze_risk_cascades(G: nx.Graph, initial_risks: List[int], threshold: float = 0.5, max_steps: int = 10) -> Dict[int, List[float]]:
    cascade_progression = {risk: [1.0] for risk in initial_risks}
    for _ in range(max_steps):
        new_activations = {}
        for node in G.nodes():
            if node not in cascade_progression:
                neighbor_influence = sum(cascade_progression.get(neighbor, [0])[-1] * G[node][neighbor]['weight']
                                             for neighbor in G.neighbors(node))
                if neighbor_influence > threshold:
                    new_activations[node] = neighbor_influence
        
        if not new_activations:
            break
        
        for node, activation in new_activations.items():
            cascade_progression[node] = [0.0] * (len(next(iter(cascade_progression.values()))) - 1) + [activation]
        
        for progression in cascade_progression.values():
            progression.append(progression[-1])
    
    return cascade_progression

def calculate_risk_correlations(risks: List[Risk], simulation_results: Dict[str, Dict[int, List[float]]]) -> Dict[Tuple[int, int], float]:
    correlations = {}
    risk_ids = [risk.id for risk in risks]
    for i, risk1_id in enumerate(risk_ids):
        for risk2_id in risk_ids[i+1:]:
            correlation, _ = pearsonr(simulation_results[risk1_id], simulation_results[risk2_id])
            correlations[(risk1_id, risk2_id)] = correlation
    return correlations

def identify_risk_feedback_loops(G: nx.Graph) -> List[List[int]]:
    feedback_loops = list(nx.simple_cycles(G))
    return [loop for loop in feedback_loops if len(loop) > 2]

def analyze_network_resilience(G: nx.Graph) -> Dict[str, float]:
    resilience_metrics = {
        "average_clustering": nx.average_clustering(G),
        "average_shortest_path_length": nx.average_shortest_path_length(G, weight='weight') if nx.is_connected(G) else float('inf'),
        "graph_density": nx.density(G),
        "assortativity": nx.degree_assortativity_coefficient(G)
    }
    return resilience_metrics

def generate_risk_interaction_summary(interactions: List[RiskInteraction], central_risks: Dict[int, float], clusters: Dict[int, int]) -> str:
    top_interactions = '\n'.join([f"- Risk {i.risk1_id} - Risk {i.risk2_id} (Score: {i.interaction_score:.2f})" for i in sorted(interactions, key=lambda x: x.interaction_score, reverse=True)[:5]])
    central_risks_str = '\n'.join([f"- Risk {risk_id} (Centrality: {centrality:.2f})" for risk_id, centrality in sorted(central_risks.items(), key=lambda x: x[1], reverse=True)[:3]])
    risk_clusters = '\n'.join([f"- Cluster {cluster}: {', '.join([str(risk_id) for risk_id, c in clusters.items() if c == cluster])}" for cluster in set(clusters.values())])

    prompt = RISK_INTERACTION_SUMMARY_PROMPT.format(
        company_name=COMPANY_INFO.name,
        top_interactions=top_interactions,
        central_risks=central_risks_str,
        risk_clusters=risk_clusters
    )

    system_message = "You are an expert in climate risk assessment and network analysis. Always respond with valid JSON."
    summary_response = get_llm_response(prompt, system_message)

    if not summary_response:
        return "Unable to generate risk interaction summary due to LLM response issues."

    return summary_response.get('summary', 'No summary provided.')

def create_risk_interaction_matrix(risks: List[Risk]) -> np.ndarray:
    n = len(risks)
    matrix = np.zeros((n, n))
    for i, risk1 in enumerate(risks):
        for j, risk2 in enumerate(risks[i+1:], start=i+1):
            interaction = analyze_single_interaction(risk1, risk2)
            matrix[i, j] = matrix[j, i] = interaction.interaction_score
    return matrix

def analyze_single_interaction(risk1: Risk, risk2: Risk) -> RiskInteraction:
    """Analyze interaction between two risks with enhanced error handling and logging."""
    logger.debug(f"Analyzing interaction between risks {risk1.id} and {risk2.id}")
    
    try:
        prompt = INTERACTION_ANALYSIS_PROMPT.format(
            company_name=COMPANY_INFO.name,
            company_industry=COMPANY_INFO.industry,
            company_region=", ".join(COMPANY_INFO.region),
            key_products=", ".join(COMPANY_INFO.key_products),
            risk1_id=risk1.id,
            risk2_id=risk2.id,
            risk1_description=risk1.description,
            risk1_category=risk1.category or "Unknown",
            risk1_subcategory=risk1.subcategory or "Unknown",
            risk2_description=risk2.description,
            risk2_category=risk2.category or "Unknown",
            risk2_subcategory=risk2.subcategory or "Unknown"
        )
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        return _create_default_interaction(risk1, risk2)

    system_message = "You are an expert in climate risk assessment and risk interactions. Always respond with valid JSON."
    
    try:
        response = get_llm_response(prompt, system_message)
        logger.debug(f"LLM Response for interaction {risk1.id}_{risk2.id}: {response}")
        
        if not response:
            logger.warning(f"Empty LLM response for risks {risk1.id} and {risk2.id}")
            return _create_default_interaction(risk1, risk2)
            
        # Validate response structure
        required_fields = ['interaction_score', 'interaction_explanation', 'compounding_effects', 'mitigating_factors']
        if not all(field in response for field in required_fields):
            logger.warning(f"Missing required fields in LLM response for risks {risk1.id} and {risk2.id}")
            return _create_default_interaction(risk1, risk2)
            
        interaction_score = response['interaction_score'].get('score', 0.5)
        interaction_type = determine_interaction_type(interaction_score)
        
        interaction = RiskInteraction(risk1.id, risk2.id, interaction_score, interaction_type)
        interaction.full_analysis = response
        interaction.interaction_explanation = response.get('interaction_explanation')
        interaction.compounding_effects = response.get('compounding_effects', [])
        interaction.mitigating_factors = response.get('mitigating_factors', [])
        
        return interaction
        
    except Exception as e:
        logger.error(f"Error analyzing interaction between risks {risk1.id} and {risk2.id}: {str(e)}")
        return _create_default_interaction(risk1, risk2)

def _create_default_interaction(risk1: Risk, risk2: Risk) -> RiskInteraction:
    """Create a default interaction when analysis fails."""
    default_data = {
        'interaction_score': {'score': 0.5},
        'interaction_explanation': 'Analysis failed - using default values',
        'compounding_effects': [],
        'mitigating_factors': []
    }
    interaction = RiskInteraction(risk1.id, risk2.id, 0.5, 'Moderate')
    interaction.full_analysis = default_data
    interaction.interaction_explanation = default_data['interaction_explanation']
    interaction.compounding_effects = default_data['compounding_effects']
    interaction.mitigating_factors = default_data['mitigating_factors']
    return interaction

def simulate_risk_interactions(risks: List[Risk], interaction_matrix: np.ndarray, num_steps: int = 10) -> Dict[int, List[float]]:
    n = len(risks)
    risk_levels = np.array([risk.impact for risk in risks])
    risk_progression = {risk.id: [risk.impact] for risk in risks}

    for _ in range(num_steps):
        influence = interaction_matrix @ risk_levels
        risk_levels = np.clip(risk_levels + 0.1 * influence, 0, 1)
        for i, risk in enumerate(risks):
            risk_progression[risk.id].append(risk_levels[i])

    return risk_progression

def aggregate_company_network(entity_networks: Dict[str, nx.Graph], entities: Dict[str, Entity]) -> nx.Graph:
    """
    Aggregates entity-level risk networks into a company-wide network using entity weights.
    
    Parameters:
        entity_networks (Dict[str, nx.Graph]): Dictionary of entity names to their risk networks
        entities (Dict[str, Entity]): Dictionary of entity names to their Entity objects
    
    Returns:
        nx.Graph: Aggregated company-wide network
    """
    logger.info("Aggregating entity-level networks into company-wide network with weights")
    
    company_network = nx.Graph()
    edge_weights = {}
    
    # Calculate total weight for normalization
    total_weight = sum(entity.weight for entity in entities.values())
    
    # Combine all entity networks with weighted edges
    for entity_name, network in entity_networks.items():
        if entity_name not in entities:
            logger.warning(f"Entity {entity_name} not found in configuration. Skipping.")
            continue
            
        entity_weight = entities[entity_name].weight / total_weight
        logger.debug(f"Processing network for entity: {entity_name} (weight: {entity_weight:.3f})")
        
        # Add nodes with their attributes
        for node, attrs in network.nodes(data=True):
            if not company_network.has_node(node):
                company_network.add_node(node, **attrs)
        
        # Add weighted edges
        for u, v, data in network.edges(data=True):
            edge = tuple(sorted([u, v]))
            
            if edge not in edge_weights:
                edge_weights[edge] = {'weighted_sum': 0, 'weight_sum': 0}
            
            edge_weight = data.get('weight', 0)
            edge_weights[edge]['weighted_sum'] += edge_weight * entity_weight
            edge_weights[edge]['weight_sum'] += entity_weight
    
    # Add weighted edges to company network
    for (u, v), weights in edge_weights.items():
        weighted_avg = weights['weighted_sum'] / weights['weight_sum']
        company_network.add_edge(u, v, weight=weighted_avg)
    
    logger.info(f"Created weighted company-wide network with {company_network.number_of_nodes()} nodes and {company_network.number_of_edges()} edges")
    return company_network

