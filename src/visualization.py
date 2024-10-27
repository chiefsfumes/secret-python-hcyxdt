import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import numpy as np
import networkx as nx
import os
from src.models import Risk, RiskInteraction, SimulationResult
from src.config import OUTPUT_DIR, VIZ_DPI, HEATMAP_CMAP, TIME_SERIES_HORIZON
import logging

# Add this line near the top of the file, after the imports
logger = logging.getLogger(__name__)

def generate_visualizations(aggregated_results: Dict, output_dir: str):
    """
    Generate all visualizations based on the aggregated results.

    Parameters:
    - aggregated_results (Dict): The aggregated results from the risk assessment.
    - output_dir (str): The directory where the visualizations will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract data from aggregated_results
    risks = aggregated_results.get("risks", [])
    risk_interactions = aggregated_results.get("risk_interactions", [])
    monte_carlo_results = aggregated_results.get("monte_carlo_results", {})
    scenario_sensitivity = aggregated_results.get("scenario_sensitivity", {})
    time_series_results = aggregated_results.get("time_series_results", {})
    risk_networks = aggregated_results.get("risk_networks", {})
    risk_clusters = aggregated_results.get("risk_clusters", {})
    cumulative_impact = aggregated_results.get("cumulative_impact", {})
    interaction_matrix = {}  # Populate if available
    risk_progression = {}    # Populate if available
    aggregate_impact = aggregated_results.get("aggregate_impact", {})

    # If risk_networks, risk_clusters, interaction_matrix, risk_progression are available in aggregated_results,
    # extract them here. Otherwise, handle accordingly.

    # Generate individual visualizations
    risk_matrix(risks, output_dir)
    interaction_heatmap(risks, risk_interactions, output_dir)
    
    # Update the network visualization check
    if risk_networks:
        for entity_name, network in risk_networks.items():
            clusters = risk_clusters.get(entity_name, {})
            interaction_network(risks, risk_interactions, network, clusters, output_dir, entity_name)
            logger.info(f"Generated network visualization for entity: {entity_name}")
    else:
        logger.warning("Risk networks data not available. Skipping interaction network visualization.")

    if monte_carlo_results:
        monte_carlo_results_plot(monte_carlo_results, output_dir)
    else:
        logger.warning("Monte Carlo results data not available. Skipping Monte Carlo visualization.")

    if scenario_sensitivity:
        sensitivity_analysis_heatmap(scenario_sensitivity, output_dir)
    else:
        logger.warning("Sensitivity results data not available. Skipping sensitivity analysis heatmap.")

    if time_series_results:
        time_series_projection(risks, time_series_results, output_dir)
    else:
        logger.warning("Time series results data not available. Skipping time series projection.")

    if cumulative_impact:
        cumulative_impact_plot(cumulative_impact, output_dir)
    else:
        logger.warning("Cumulative impact data not available. Skipping cumulative impact plot.")

    if interaction_matrix:
        interaction_matrix_heatmap(risks, interaction_matrix, output_dir)
    else:
        logger.warning("Interaction matrix data not available. Skipping interaction matrix heatmap.")

    if risk_progression:
        risk_progression_plot(risks, risk_progression, output_dir)
    else:
        logger.warning("Risk progression data not available. Skipping risk progression plot.")

    if aggregate_impact:
        aggregate_impact_distribution(aggregate_impact, output_dir)
    else:
        logger.warning("Aggregate impact data not available. Skipping aggregate impact distribution.")

    # Add comparison between entities if applicable
    compare_entities(risks, aggregate_impact, output_dir)

def risk_matrix(risks: List[Risk], output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    categories = set(risk.category for risk in risks)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    for risk in risks:
        ax.scatter(risk.likelihood, risk.impact, s=100, c=[color_map[risk.category]], alpha=0.7)
        ax.annotate(risk.id, (risk.likelihood, risk.impact), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Likelihood')
    ax.set_ylabel('Impact')
    ax.set_title('Risk Matrix')
    
    # Create a ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap='tab10')
    sm.set_array([])
    
    # Add the colorbar to the figure, not to the axes
    cbar = fig.colorbar(sm, ax=ax, label='Risk Category', ticks=[])
    
    # Add category labels to the colorbar
    cbar.set_ticks(np.linspace(0, 1, len(categories)))
    cbar.set_ticklabels(categories)
    
    plt.savefig(os.path.join(output_dir, 'risk_matrix.png'), dpi=VIZ_DPI)
    plt.close(fig)

def interaction_matrix_heatmap(risks: List[Risk], interaction_matrix: np.ndarray, output_dir: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_matrix, annot=True, cmap=HEATMAP_CMAP, xticklabels=[r.id for r in risks], yticklabels=[r.id for r in risks])
    plt.title('Risk Interaction Matrix')
    plt.savefig(os.path.join(output_dir, 'interaction_matrix_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

def risk_progression_plot(risks: List[Risk], risk_progression: Dict[int, List[float]], output_dir: str):
    plt.figure(figsize=(12, 8))
    for risk_id, progression in risk_progression.items():
        plt.plot(progression, label=f'Risk {risk_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Risk Level')
    plt.title('Risk Progression Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_progression.png'), dpi=VIZ_DPI)
    plt.close()

def aggregate_impact_distribution(aggregate_impact: Dict[str, float], output_dir: str):
    plt.figure(figsize=(10, 6))
    impact_values = list(aggregate_impact.values())
    sns.histplot(impact_values, kde=True)
    if 'mean' in aggregate_impact:
        plt.axvline(aggregate_impact['mean'], color='r', linestyle='--', label='Mean')
    if '95th_percentile' in aggregate_impact:
        plt.axvline(aggregate_impact['95th_percentile'], color='g', linestyle='--', label='95th Percentile')
    plt.xlabel('Aggregate Impact')
    plt.ylabel('Frequency')
    plt.title('Distribution of Aggregate Impact')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'aggregate_impact_distribution.png'), dpi=VIZ_DPI)
    plt.close()

def interaction_heatmap(risks: List[Risk], risk_interactions: List[RiskInteraction], output_dir: str):
    n = len(risks)
    interaction_matrix = np.zeros((n, n))
    for interaction in risk_interactions:
        i = next(index for index, risk in enumerate(risks) if risk.id == interaction.risk1_id)
        j = next(index for index, risk in enumerate(risks) if risk.id == interaction.risk2_id)
        interaction_matrix[i, j] = interaction_matrix[j, i] = interaction.interaction_score
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(interaction_matrix, annot=True, cmap=HEATMAP_CMAP, xticklabels=[r.id for r in risks], yticklabels=[r.id for r in risks])
    plt.title('Risk Interaction Heatmap')
    plt.savefig(os.path.join(output_dir, 'interaction_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

def interaction_network(risks: List[Risk], risk_interactions: List[RiskInteraction], 
                      risk_network: nx.Graph, risk_clusters: Dict[int, int], 
                      output_dir: str, entity_name: str):
    """Updated to include entity name in output"""
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(risk_network)
    
    # Create a color map for risk categories
    categories = set(risk.category for risk in risks)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    nx.draw(risk_network, pos, with_labels=True, 
            node_color=[color_map[risk_network.nodes[node]['category']] for node in risk_network.nodes()],
            node_size=1000, font_size=8, font_weight='bold', edge_color='gray', 
            width=[risk_network[u][v]['weight'] * 2 for u, v in risk_network.edges()])
    
    nx.draw_networkx_edge_labels(risk_network, pos, 
                                edge_labels={(u, v): f"{risk_network[u][v]['weight']:.2f}" 
                                           for u, v in risk_network.edges()})
    
    plt.title(f'Risk Interaction Network - {entity_name}')
    plt.savefig(os.path.join(output_dir, f'interaction_network_{entity_name}.png'), dpi=VIZ_DPI)
    plt.close()

def monte_carlo_results_plot(simulation_results: Dict[str, Dict[int, SimulationResult]], output_dir: str):
    plt.figure(figsize=(16, 12))
    for scenario, results in simulation_results.items():
        for risk_id, sim_result in results.items():
            sns.kdeplot(sim_result.impact_distribution, label=f'Risk {risk_id} - {scenario}')
    plt.xlabel('Risk Impact')
    plt.ylabel('Density')
    plt.title('Monte Carlo Simulation Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monte_carlo_results.png'), dpi=VIZ_DPI)
    plt.close()

def sensitivity_analysis_heatmap(sensitivity_results: Dict[str, Dict[str, float]], output_dir: str):
    plt.figure(figsize=(14, 10))
    
    # Convert the sensitivity results to a DataFrame
    sensitivity_df = pd.DataFrame(sensitivity_results).T
    
    # Ensure all columns are numeric
    for col in sensitivity_df.columns:
        sensitivity_df[col] = pd.to_numeric(sensitivity_df[col], errors='coerce')
    
    # Drop any columns that couldn't be converted to numeric
    sensitivity_df = sensitivity_df.select_dtypes(include=[np.number])
    
    if not sensitivity_df.empty:
        # Create the heatmap using only numeric columns
        sns.heatmap(sensitivity_df, annot=True, cmap='coolwarm', center=0)
        plt.title('Sensitivity Analysis Heatmap')
        plt.savefig(os.path.join(output_dir, 'sensitivity_analysis_heatmap.png'), dpi=VIZ_DPI)
    else:
        logger.warning("No numeric data found for sensitivity analysis heatmap")
        plt.text(0.5, 0.5, "No numeric data available for heatmap", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Sensitivity Analysis Heatmap (No Data)')
        plt.savefig(os.path.join(output_dir, 'sensitivity_analysis_heatmap_no_data.png'), dpi=VIZ_DPI)
    
    plt.close()

    # Log any columns that were dropped
    try:
        dropped_columns = set(sensitivity_results[next(iter(sensitivity_results))].keys()) - set(sensitivity_df.columns)
        if dropped_columns:
            logger.warning("The following columns were dropped due to non-numeric values:")
            for col in dropped_columns:
                logger.warning(f"- {col}")
    except KeyError:
        logger.warning("No data available in sensitivity_results to determine dropped columns.")
    
    if sensitivity_df.empty:
        logger.warning("All columns in sensitivity results were non-numeric")

def time_series_projection(risks: List[Risk], time_series_results: Dict[int, List[float]], output_dir: str):
    # Set matplotlib logging level to WARNING to reduce noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    plt.figure(figsize=(16, 12))
    for risk_id, projections in time_series_results.items():
        try:
            risk = next(r for r in risks if r.id == risk_id)
            plt.plot(range(1, TIME_SERIES_HORIZON + 1), projections, label=f'Risk {risk_id} ({risk.category})')
        except StopIteration:
            logger.warning(f"Risk ID {risk_id} not found in risks list. Skipping.")
    plt.xlabel('Years into the future')
    plt.ylabel('Projected Impact')
    plt.title('Time Series Projection of Risk Impacts')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_projection.png'), dpi=VIZ_DPI)
    plt.close()

def cumulative_impact_plot(cumulative_impact: Dict[str, float], output_dir: str):
    plt.figure(figsize=(12, 8))
    if 'values' in cumulative_impact and isinstance(cumulative_impact['values'], list):
        plt.plot(range(len(cumulative_impact['values'])), cumulative_impact['values'])
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Impact')
        plt.title('Cumulative Impact Over Time')
        plt.savefig(os.path.join(output_dir, 'cumulative_impact_plot.png'), dpi=VIZ_DPI)
    else:
        logger.warning("Cumulative impact data format is incorrect.")
    plt.close()

def compare_entities(risks: List[Risk], aggregate_impact: Dict[str, float], output_dir: str):
    plt.figure(figsize=(12, 6))
    
    # Filter out any non-entity keys (like 'mean' or '95th_percentile')
    entities = [key for key in aggregate_impact.keys() if isinstance(aggregate_impact[key], (float, int))]
    
    if not entities:
        logger.warning("No valid entity data found in aggregate_impact")
        plt.close()
        return
        
    x = range(len(entities))
    impact_values = [aggregate_impact[entity] for entity in entities]
    
    # Create bar chart for impact values
    plt.bar(x, impact_values, align='center', alpha=0.8, label='Aggregate Impact')
    
    plt.xlabel('Entities')
    plt.ylabel('Impact')
    plt.title('Comparison of Entities: Aggregate Impact')
    plt.xticks(x, entities, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_comparison.png'), dpi=VIZ_DPI)
    plt.close()

def generate_risk_network_visualization(aggregated_results: Dict, output_dir: str) -> None:
    logger = logging.getLogger(__name__)
    
    if "risk_networks" not in aggregated_results or not aggregated_results["risk_networks"]:
        logger.warning("Risk networks data not available. Skipping interaction network visualization.")
        return

    # Create visualizations for each entity's network
    for entity, network in aggregated_results["risk_networks"].items():
        if not network:
            continue
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(network)
        
        # Draw the network
        nx.draw(network, pos, 
                with_labels=True,
                node_color='lightblue',
                node_size=1000,
                font_size=8,
                font_weight='bold')
        
        # Add edge labels for interaction scores
        edge_labels = nx.get_edge_attributes(network, 'weight')
        nx.draw_networkx_edge_labels(network, pos, edge_labels)
        
        plt.title(f'Risk Interaction Network - {entity}')
        plt.savefig(os.path.join(output_dir, f'risk_network_{entity}.png'))
        plt.close()

    # Generate company-wide network visualization if available
    if "company_network" in aggregated_results:
        company_network = aggregated_results["company_network"]
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(company_network)
        
        # Draw the company-wide network
        nx.draw(company_network, pos,
                with_labels=True,
                node_color='lightgreen',
                node_size=1200,
                font_size=10,
                font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(company_network, 'weight')
        nx.draw_networkx_edge_labels(company_network, pos, edge_labels)
        
        plt.title('Company-Wide Risk Interaction Network')
        plt.savefig(os.path.join(output_dir, 'company_wide_risk_network.png'))
        plt.close()

    logger.info(f"Generated risk network visualizations in {output_dir}")
