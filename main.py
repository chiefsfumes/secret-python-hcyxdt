from __future__ import annotations
import traceback
import os
import logging
import argparse
from typing import Dict, List
from src.config import setup_logging, SCENARIOS, OUTPUT_DIR, COMPANY_INFO
import time
import sys
import cProfile
import pstats
import io
from src.models import Risk, ExternalData, Scenario, Entity, Company

logger = logging.getLogger(__name__)

print("Script is starting...")

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check for __pycache__ directory
pycache_dir = '__pycache__'
pycache_exists = os.path.exists(pycache_dir)
print(f"__pycache__ directory exists: {pycache_exists}")

if pycache_exists:
    pyc_files = [f for f in os.listdir(pycache_dir) if f.endswith('.pyc')]
    print(f".pyc files in __pycache__: {pyc_files}")
else:
    print("No __pycache__ directory found. This is normal if this is the first run or if .pyc files are not being generated.")

def import_modules():
    pr = cProfile.Profile()
    pr.enable()
    
    global load_risk_data, load_external_data, categorize_risks, assess_risks, prioritize_risks, perform_pestel_analysis
    global analyze_risk_interactions, build_risk_network, create_risk_interaction_matrix, simulate_risk_interactions, identify_central_risks, detect_risk_clusters, analyze_risk_cascades
    global simulate_scenario_impact, analyze_sensitivity, time_series_analysis, analyze_impact_trends, identify_critical_periods, forecast_cumulative_impact
    global conduct_advanced_risk_analysis, assess_aggregate_impact, identify_tipping_points, generate_visualizations, generate_report, generate_mitigation_strategies
    global analyze_systemic_risks, identify_trigger_points, assess_system_resilience
    global perform_monte_carlo_simulations, generate_stakeholder_reports, Risk, ExternalData, Scenario, Entity, nx, get_llm_response

    import_times = {}

    def timed_import(module_name, import_statement):
        start_time = time.time()
        exec(import_statement, globals())  # Use globals() to ensure the imported names are available in the global scope
        end_time = time.time()
        import_times[module_name] = end_time - start_time
        print(f"Imported {module_name} in {import_times[module_name]:.4f} seconds")

    timed_import("data_loader", "from src.data_loader import load_risk_data, load_external_data")
    timed_import("risk_analysis.categorization", "from src.risk_analysis.categorization import categorize_risks, assess_risks, prioritize_risks, perform_pestel_analysis")
    timed_import("risk_analysis.interaction_analysis", "from src.risk_analysis.interaction_analysis import analyze_risk_interactions, build_risk_network, create_risk_interaction_matrix, simulate_risk_interactions, identify_central_risks, detect_risk_clusters, analyze_risk_cascades")
    timed_import("risk_analysis.scenario_analysis", "from src.risk_analysis.scenario_analysis import simulate_scenario_impact, analyze_sensitivity")
    timed_import("risk_analysis.time_series_analysis", "from src.risk_analysis.time_series_analysis import time_series_analysis, analyze_impact_trends, identify_critical_periods, forecast_cumulative_impact")
    timed_import("risk_analysis.advanced_analysis", "from src.risk_analysis.advanced_analysis import conduct_advanced_risk_analysis, assess_aggregate_impact, identify_tipping_points")
    timed_import("visualization", "from src.visualization import generate_visualizations")
    timed_import("reporting", "from src.reporting import generate_report, generate_mitigation_strategies")
    timed_import("risk_analysis.systemic_risk_analysis", "from src.risk_analysis.systemic_risk_analysis import analyze_systemic_risks, identify_trigger_points, assess_system_resilience")
    timed_import("sensitivity_analysis.monte_carlo", "from src.sensitivity_analysis.monte_carlo import perform_monte_carlo_simulations")
    timed_import("reporting_module.stakeholder_reports", "from src.reporting_module.stakeholder_reports import generate_stakeholder_reports")
    timed_import("models", "from src.models import Risk, ExternalData, Scenario, Entity")
    timed_import("networkx", "import networkx as nx")
    timed_import("utils.llm_util", "from src.utils.llm_util import get_llm_response")

    print("All modules imported successfully.")
    
    # Sort and print import times
    sorted_import_times = sorted(import_times.items(), key=lambda x: x[1], reverse=True)
    print("\nImport times (sorted from slowest to fastest):")
    for module, import_time in sorted_import_times:
        print(f"{module}: {import_time:.4f} seconds")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

    return import_times

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Climate Risk Assessment Tool")
    parser.add_argument("--risk_data", type=str, default="data/risk_data.csv", help="Path to risk data CSV file")
    parser.add_argument("--external_data", type=str, default="data/external_data.csv", help="Path to external data CSV file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")  # Add this line
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()

def cleanup_old_checkpoints(checkpoint_dir: str, max_age_days: int = 7) -> None:
    """Remove checkpoint files older than max_age_days."""
    current_time = time.time()
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > (max_age_days * 86400):  # Convert days to seconds
                    try:
                        os.remove(file_path)
                        logger.debug(f"Removed old checkpoint: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old checkpoint {file_path}: {str(e)}")

def main(args: argparse.Namespace) -> None:
    print("Main function called")
    print("Starting the script...")
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Advanced Climate Risk Assessment Tool")
    logger.debug(f"Arguments: {args}")

    # Create output and checkpoint directories
    print("Ensuring output and checkpoint directories exist...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)  # Add this line
    
    # Create company-specific checkpoint directory
    company_checkpoint_dir = os.path.join("checkpoints", COMPANY_INFO.name.lower().replace(' ', '_').rstrip('.'))
    os.makedirs(company_checkpoint_dir, exist_ok=True)  # Add this line

    print("Arguments parsed successfully.")
    print(f"Risk data file: {args.risk_data}")
    print(f"External data file: {args.external_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log level: {args.log_level}")

    try:
        # Import modules only when needed and capture import times
        import_times = import_modules()
        
        # Log import times
        logger.info("Module import times:")
        for module, import_time in sorted(import_times.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{module}: {import_time:.4f} seconds")

        # Data Collection and Preprocessing
        print("Loading risk data...")
        risks: List[Risk] = load_risk_data(args.risk_data)
        logger.info(f"Loaded {len(risks)} risks from the data file")
        
        # Print out information about each risk
        for risk in risks:
            logger.info(f"Risk ID: {risk.id}, Description: {risk.description}, Entity: {risk.entity}")

        print("Loading external data...")
        external_data: Dict[str, ExternalData] = load_external_data(args.external_data)

        # Perform analysis for each entity
        entity_results = {}
        
        # First, analyze risks at the lowest level entities
        leaf_entities = {name: entity for name, entity in COMPANY_INFO.entities.items() 
                        if not entity.sub_entities}
        
        print("Analyzing leaf-level entities first...")
        for entity_name, entity in leaf_entities.items():
            print(f"Analyzing risks for leaf entity: {entity_name}")
            logger.info(f"Analyzing risks for leaf entity: {entity_name}")
            
            # Instead of filtering, assign all risks to each entity
            entity_risks = risks.copy()  # Analyze all risks for each entity
            logger.info(f"Analyzing {len(entity_risks)} risks for entity: {entity_name}")
            
            # For risks without an entity assigned, set it to the current entity
            for risk in entity_risks:
                if not risk.entity:
                    risk.entity = entity_name
            
            # Enhanced Risk Categorization and Assessment
            categorized_risks = categorize_risks(entity_risks, entity)
            assessed_risks = assess_risks(entity_risks, entity)
            prioritized_risks = prioritize_risks(assessed_risks)
            pestel_analysis = perform_pestel_analysis(assessed_risks, external_data)
            
            # Risk Interaction Analysis
            print(f"Performing hierarchical risk interaction analysis...")
            logger.info(f"Starting hierarchical risk interaction analysis")

            # Use the new hierarchical analysis
            risk_interactions = analyze_risk_interactions(assessed_risks, COMPANY_INFO)
            logger.info(f"Generated hierarchical risk interactions")

            risk_network = build_risk_network(assessed_risks, risk_interactions, COMPANY_INFO)
            logger.info(f"Built hierarchical risk networks")

            central_risks = identify_central_risks(risk_network)
            risk_clusters = detect_risk_clusters(risk_network)
            
            # Time Series Analysis
            time_series_results = time_series_analysis({entity_name: entity_risks}, external_data)
            impact_trends = analyze_impact_trends(time_series_results)
            critical_periods = identify_critical_periods(time_series_results, threshold=0.7)
            cumulative_impact = forecast_cumulative_impact(time_series_results)
            
            # Scenario Analysis
            print(f"Performing scenario analysis for {entity_name}...")
            scenario_impacts = simulate_scenario_impact(assessed_risks, external_data, SCENARIOS, [entity])
            monte_carlo_results = perform_monte_carlo_simulations(
                risks=assessed_risks,
                scenarios=SCENARIOS,
                external_data=external_data,
                entities=[entity]  # Pass the current entity being analyzed
            )
            scenario_sensitivity = analyze_sensitivity(monte_carlo_results=monte_carlo_results)

            # If you want to perform the original scenario variable sensitivity analysis:
            base_scenario = next(iter(SCENARIOS.values()))  # Get the first scenario as the base scenario
            scenario_sensitivity = analyze_sensitivity(
                base_scenario=base_scenario,
                variable='temp_increase',
                range_pct=0.1,
                entities=[entity],
                risks=assessed_risks,
                external_data=external_data
            )
            
            # Advanced Analysis
            systemic_risks = analyze_systemic_risks({entity_name: entity_risks}, COMPANY_INFO)
            trigger_points = identify_trigger_points(assessed_risks, risk_network, external_data)
            resilience_assessment = assess_system_resilience(assessed_risks, risk_network, scenario_impacts)
            aggregate_impact = assess_aggregate_impact(assessed_risks, risk_interactions)
            tipping_points = identify_tipping_points(assessed_risks, risk_interactions)
            
            comprehensive_analysis = conduct_advanced_risk_analysis(assessed_risks, SCENARIOS, COMPANY_INFO, external_data)
            mitigation_strategies = generate_mitigation_strategies(assessed_risks, comprehensive_analysis)
            
            # Store results for the entity
            entity_results[entity_name] = {
                "categorized_risks": categorized_risks,
                "assessed_risks": assessed_risks,
                "prioritized_risks": prioritized_risks,
                "pestel_analysis": pestel_analysis,
                "risk_interactions": risk_interactions,
                "risk_network": risk_network,
                "central_risks": central_risks,
                "risk_clusters": risk_clusters,
                "time_series_results": time_series_results,
                "impact_trends": impact_trends,
                "critical_periods": critical_periods,
                "cumulative_impact": cumulative_impact,
                "scenario_impacts": scenario_impacts,
                "monte_carlo_results": monte_carlo_results,
                "scenario_sensitivity": scenario_sensitivity,
                "systemic_risks": systemic_risks,
                "trigger_points": trigger_points,
                "resilience_assessment": resilience_assessment,
                "aggregate_impact": aggregate_impact,
                "tipping_points": tipping_points,
                "llm_risk_assessment": comprehensive_analysis,
                "mitigation_strategies": mitigation_strategies,
                "executive_insights": comprehensive_analysis.get('executive_insights', {}).get(entity_name, ""),
                "sensitivity_analysis": {
                    "monte_carlo": monte_carlo_results,
                    "scenario_variable": scenario_sensitivity,
                    "risk_sensitivities": {
                        risk.id: {
                            "impact": scenario_sensitivity.get(f"risk_{risk.id}_impact", 0),
                            "likelihood": scenario_sensitivity.get(f"risk_{risk.id}_likelihood", 0)
                        } for risk in assessed_risks
                    }
                }
            }
        
        # Then analyze parent entities
        parent_entities = {name: entity for name, entity in COMPANY_INFO.entities.items() 
                         if entity.sub_entities}
        
        print("Analyzing parent-level entities...")
        for entity_name, entity in parent_entities.items():
            print(f"Analyzing risks for parent entity: {entity_name}")
            logger.info(f"Analyzing risks for parent entity: {entity_name}")
            
            # Get child entity results
            child_results = {name: entity_results[name] 
                           for name in entity.sub_entities 
                           if name in entity_results}
            
            # Aggregate child results with weights
            entity_results[entity_name] = aggregate_child_results(
                child_results,
                entity,
                COMPANY_INFO
            )
        
        # Aggregate final results across all entities
        print("Aggregating results across all entities...")
        aggregated_results = aggregate_entity_results(entity_results)
        
        # Generate visualizations and reports
        print("Generating visualizations and reports...")
        generate_visualizations(aggregated_results, args.output_dir)
        main_report = generate_report(aggregated_results, SCENARIOS, COMPANY_INFO)
        stakeholder_reports = generate_stakeholder_reports(main_report, COMPANY_INFO.industry)
        
        logger.info("Risk Assessment Report and stakeholder reports generated successfully.")
        print("Reports generated successfully.")
        logger.info(f"Main report saved to: {os.path.join(args.output_dir, 'climate_risk_report.json')}")
        print(f"Main report saved to: {os.path.join(args.output_dir, 'climate_risk_report.json')}")
        logger.info(f"Stakeholder reports saved in: {args.output_dir}")
        print(f"Stakeholder reports saved in: {args.output_dir}")
        logger.info(f"Visualizations saved in: {args.output_dir}")
        print(f"Visualizations saved in: {args.output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()
        raise

def aggregate_child_results(child_results: Dict[str, Dict], 
                          parent_entity: Entity,
                          company: Company) -> Dict:
    """Aggregate results from child entities to parent level."""
    aggregated = {
        "risks": [],
        "risk_interactions": [],
        "risk_network": None,
        "central_risks": {},
        "risk_clusters": {},
        # ... (other result fields) ...
    }
    
    total_weight = sum(company.get_entity(child).weight 
                      for child in parent_entity.sub_entities)
    
    for child_name, results in child_results.items():
        child_weight = company.get_entity(child_name).weight / total_weight
        
        # Weight and aggregate results
        for risk in results["assessed_risks"]:
            risk.impact *= child_weight
            risk.likelihood *= child_weight
            aggregated["risks"].append(risk)
        
        # Aggregate weighted interactions
        for interaction in results["risk_interactions"]:
            interaction.interaction_score *= child_weight
            aggregated["risk_interactions"].append(interaction)
        
        # ... (aggregate other results) ...
    
    return aggregated

def aggregate_entity_results(entity_results: Dict[str, Dict]) -> Dict:
    aggregated_results = {
        "risks": [],
        "risk_interactions": [],
        "scenario_impacts": {},
        "monte_carlo_results": {},
        "systemic_risks": {},
        "aggregate_impact": {},
        "tipping_points": [],
        "executive_insights": {},
        "mitigation_strategies": {},
        # Add these new fields
        "risk_networks": {},  # Store networks by entity
        "risk_clusters": {},  # Store clusters by entity
    }

    # Add sensitivity analysis aggregation
    if "sensitivity_analysis" not in aggregated_results:
        aggregated_results["sensitivity_analysis"] = {
            "monte_carlo": {},
            "scenario_variable": {},
            "risk_sensitivities": {}
        }

    for entity, results in entity_results.items():
        aggregated_results["risks"].extend(results["assessed_risks"])
        aggregated_results["risk_interactions"].extend(results["risk_interactions"])
        
        for scenario, impacts in results["scenario_impacts"].items():
            if scenario not in aggregated_results["scenario_impacts"]:
                aggregated_results["scenario_impacts"][scenario] = []
            aggregated_results["scenario_impacts"][scenario].extend(impacts)
        
        for scenario, mc_results in results["monte_carlo_results"].items():
            if scenario not in aggregated_results["monte_carlo_results"]:
                aggregated_results["monte_carlo_results"][scenario] = {}
            aggregated_results["monte_carlo_results"][scenario].update(mc_results)
        
        aggregated_results["systemic_risks"].update(results["systemic_risks"])
        
        for impact_type, impact_value in results["aggregate_impact"].items():
            if impact_type not in aggregated_results["aggregate_impact"]:
                aggregated_results["aggregate_impact"][impact_type] = 0
            aggregated_results["aggregate_impact"][impact_type] += impact_value
        
        aggregated_results["tipping_points"].extend(results["tipping_points"])
        
        # Update this part
        aggregated_results["executive_insights"][entity] = results["executive_insights"]
        
        for risk_id, strategies in results["mitigation_strategies"].items():
            if risk_id not in aggregated_results["mitigation_strategies"]:
                aggregated_results["mitigation_strategies"][risk_id] = []
            aggregated_results["mitigation_strategies"][risk_id].extend(strategies)

        # Add network and cluster storage
        if "risk_network" in results:
            aggregated_results["risk_networks"][entity] = results["risk_network"]
        if "risk_clusters" in results:
            aggregated_results["risk_clusters"][entity] = results["risk_clusters"]

        if "sensitivity_analysis" in results:
            # Aggregate monte carlo results
            aggregated_results["sensitivity_analysis"]["monte_carlo"].update(
                results["sensitivity_analysis"]["monte_carlo"]
            )
            
            # Aggregate scenario variable sensitivity
            aggregated_results["sensitivity_analysis"]["scenario_variable"].update(
                results["sensitivity_analysis"]["scenario_variable"]
            )
            
            # Aggregate risk sensitivities
            aggregated_results["sensitivity_analysis"]["risk_sensitivities"].update(
                results["sensitivity_analysis"]["risk_sensitivities"]
            )

    return aggregated_results

if __name__ == "__main__":
    print("Entering main block")
    args = parse_arguments()
    main(args)

