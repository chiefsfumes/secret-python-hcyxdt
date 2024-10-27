import json
import numpy as np
from typing import List, Dict, Tuple, Union
import pandas as pd
import os
from src.models import Risk, RiskInteraction, SimulationResult, Scenario, Company
from src.config import OUTPUT_DIR
from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_report(aggregated_results: Dict, scenarios: Dict[str, Scenario], company_info: Company) -> str:
    """
    Generate a comprehensive report from aggregated results
    """
    report = {
        "executive_summary": generate_executive_summary(
            risks=aggregated_results.get("risks", []),
            scenario_impacts=aggregated_results.get("scenario_impacts", {}),
            simulation_results=aggregated_results.get("monte_carlo_results", {}),
            advanced_analysis=aggregated_results.get("llm_risk_assessment", {}),
            aggregate_impact=aggregated_results.get("aggregate_impact", {}),
            tipping_points=aggregated_results.get("tipping_points", [])
        ),
        "entity_reports": {}
    }
    
    # Process each entity's results
    for entity_name in company_info.entities:
        if entity_name in aggregated_results.get("executive_insights", {}):
            report["entity_reports"][entity_name] = {
                "risk_overview": {
                    "total_risks": len(aggregated_results["risks"]),
                    "high_impact_risks": [
                        {"id": risk.id, "description": risk.description, "impact": risk.impact}
                        for risk in aggregated_results["risks"] if risk.impact > 0.7
                    ]
                },
                "scenario_analysis": aggregated_results.get("scenario_impacts", {}),
                "monte_carlo_results": aggregated_results.get("monte_carlo_results", {}),
                "sensitivity_analysis": aggregated_results.get("scenario_sensitivity", {}),
                "systemic_risks": aggregated_results.get("systemic_risks", {}),
                "trigger_points": aggregated_results.get("trigger_points", {}),
                "resilience_assessment": aggregated_results.get("resilience_assessment", {}),
                "aggregate_impact": aggregated_results.get("aggregate_impact", {}),
                "tipping_points": aggregated_results.get("tipping_points", []),
                "executive_insights": aggregated_results["executive_insights"][entity_name]
            }
    
    report["cross_entity_analysis"] = perform_cross_entity_analysis(report["entity_reports"])
    
    # Generate output files
    report_json = json.dumps(report, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'climate_risk_report.json'), 'w') as f:
        f.write(report_json)
    
    generate_html_report(report)
    
    return report_json


def generate_executive_summary(risks: List[Risk], scenario_impacts: Dict[str, List[Tuple[Risk, float]]], 
                               simulation_results: Dict[str, Dict[int, SimulationResult]],
                               advanced_analysis: Dict, aggregate_impact: Dict, tipping_points: List[Dict]) -> str:
    num_risks = len(risks)
    high_impact_risks = sum(1 for risk in risks if risk.impact > 0.7)
    
    # Safer handling of scenario impacts
    try:
        # Calculate total impact for each scenario, handling different data structures
        scenario_total_impacts = {}
        for scenario_name, impacts in scenario_impacts.items():
            total = 0
            for impact in impacts:
                if isinstance(impact, tuple):
                    # Handle tuple format (Risk, float)
                    total += impact[1]
                elif isinstance(impact, dict):
                    # Handle dictionary format with 'impact' key
                    impact_value = impact.get('impact')
                    if isinstance(impact_value, (int, float)):
                        total += impact_value
            scenario_total_impacts[scenario_name] = total
        
        # Find worst scenario based on total impact
        worst_scenario = max(scenario_total_impacts.items(), key=lambda x: x[1])
        worst_scenario_name = worst_scenario[0]
        worst_scenario_impacts = scenario_impacts[worst_scenario_name]
        
    except (ValueError, AttributeError, KeyError) as e:
        # Fallback if scenario analysis fails
        worst_scenario_name = "Unknown"
        worst_scenario_impacts = []
        print(f"Warning: Error processing scenario impacts: {e}")
    
    summary = f"""
    Executive Summary:
    
    This climate risk assessment identified {num_risks} distinct risks, with {high_impact_risks} classified as high-impact.
    The '{worst_scenario_name}' scenario presents the most significant challenges, with potential for severe impacts across multiple risk categories.
    
    Key findings:
    1. {summarize_top_risks(worst_scenario_impacts)}
    2. {summarize_monte_carlo_results(simulation_results)}
    3. Aggregate Impact: The mean aggregate impact across all risks is {aggregate_impact.get('mean', 0):.2f}, with a 95th percentile impact of {aggregate_impact.get('95th_percentile', 0):.2f}.
    4. Tipping Points: {len(tipping_points)} potential tipping points were identified, with the most critical occurring at an impact level of {max((tp.get('tipping_point_level', 0) for tp in tipping_points), default=0):.2f}.
    
    Advanced Analysis Insights:
    {advanced_analysis.get('executive_insights', 'No advanced analysis insights available.')}
    
    Immediate attention is required to develop and implement comprehensive mitigation strategies, particularly focusing on the high-impact risks and potential tipping points identified in this assessment.
    """
    
    return summary

def summarize_risk_interactions(risk_interactions: List[RiskInteraction]) -> str:
    strong_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Strong")
    moderate_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Moderate")
    weak_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Weak")
    
    summary = f"""
    Risk Interaction Summary:
    - Strong interactions: {strong_interactions}
    - Moderate interactions: {moderate_interactions}
    - Weak interactions: {weak_interactions}
    
    The analysis reveals a complex web of risk interactions, with {strong_interactions} strong interactions 
    indicating potential compounding effects that require careful consideration in risk mitigation strategies.
    """
    
    return summary

def summarize_scenario_impact(impacts: List[Tuple[Risk, float]]) -> str:
    sorted_impacts = sorted(impacts, key=lambda x: x[1], reverse=True)
    top_3_risks = sorted_impacts[:3]
    
    summary = f"""
    Scenario Impact Summary:
    Top 3 impacted risks:
    1. Risk {top_3_risks[0][0].id}: Impact score {top_3_risks[0][1]:.2f}
    2. Risk {top_3_risks[1][0].id}: Impact score {top_3_risks[1][1]:.2f}
    3. Risk {top_3_risks[2][0].id}: Impact score {top_3_risks[2][1]:.2f}
    
    This scenario shows significant impacts on the above risks, requiring targeted mitigation strategies.
    """
    
    return summary

def summarize_top_risks(impacts: Union[List[Tuple[Risk, float]], List[Dict]]) -> str:
    try:
        # Try original structure (List[Tuple[Risk, float]])
        sorted_impacts = sorted(impacts, key=lambda x: x[1], reverse=True)
        top_3_risks = sorted_impacts[:3]
        
        summary = f"""
        The top 3 risks under this scenario are:
        1. {top_3_risks[0][0].description} (Impact: {top_3_risks[0][1]:.2f})
        2. {top_3_risks[1][0].description} (Impact: {top_3_risks[1][1]:.2f})
        3. {top_3_risks[2][0].description} (Impact: {top_3_risks[2][1]:.2f})
        """
    except (IndexError, AttributeError, TypeError):
        try:
            # Try dictionary structure
            if isinstance(impacts[0], dict):
                sorted_impacts = sorted(impacts, key=lambda x: x.get('impact', 0), reverse=True)
                top_3_risks = sorted_impacts[:3]
                
                summary = f"""
                The top 3 risks under this scenario are:
                1. Risk {top_3_risks[0].get('risk_id', 'N/A')} (Impact: {top_3_risks[0].get('impact', 0):.2f})
                2. Risk {top_3_risks[1].get('risk_id', 'N/A')} (Impact: {top_3_risks[1].get('impact', 0):.2f})
                3. Risk {top_3_risks[2].get('risk_id', 'N/A')} (Impact: {top_3_risks[2].get('impact', 0):.2f})
                """
            # Try string or other format
            else:
                summary = "Unable to determine top risks due to unexpected data format"
        except (IndexError, AttributeError):
            summary = "Insufficient data to determine top risks"
    
    return summary

def summarize_monte_carlo_results(simulation_results: Dict[str, Dict[int, SimulationResult]]) -> str:
    scenario_summaries = []
    
    for scenario, results in simulation_results.items():
        max_impact_risk = max(results.items(), key=lambda x: np.mean(x[1].impact_distribution))
        max_likelihood_risk = max(results.items(), key=lambda x: np.mean(x[1].likelihood_distribution))
        
        scenario_summary = f"""
        {scenario} Scenario:
        - Highest impact risk: Risk {max_impact_risk[0]} (Mean impact: {np.mean(max_impact_risk[1].impact_distribution):.2f})
        - Highest likelihood risk: Risk {max_likelihood_risk[0]} (Mean likelihood: {np.mean(max_likelihood_risk[1].likelihood_distribution):.2f})
        """
        scenario_summaries.append(scenario_summary)
    
    return "\n".join(scenario_summaries)
def generate_mitigation_strategies(risks: List[Risk], advanced_analysis: Dict) -> Dict[int, List[str]]:
    mitigation_strategies = {}
    
    for risk in risks:
        strategies = []
        
        # Use the advanced analysis results directly
        risk_analysis = advanced_analysis.get(risk.id, {})
        
        # Add strategies from the advanced analysis
        if 'mitigation_strategies' in risk_analysis:
            for strategy in risk_analysis['mitigation_strategies']:
                strategies.append(strategy['description'])
        
        # Add default strategies based on risk category
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
    

def generate_html_report(report: Dict) -> None:
    try:
        # Set up Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__), 'templates')),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Load the HTML template
        template = env.get_template('climate_risk_report.html')
        
        # Render the template with the report data
        html_content = template.render(report=report)
        
        # Define the output path
        output_path = os.path.join(OUTPUT_DIR, 'climate_risk_report.html')
        
        # Write the rendered HTML to the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTML report successfully generated at {output_path}")
    
    except Exception as e:
        print(f"An error occurred while generating the HTML report: {e}")

def generate_entity_report(entity: str, risks: List[Risk], categorized_risks: Dict[str, List[Risk]], 
                           risk_interactions: List[RiskInteraction], scenario_impacts: Dict[str, List[Tuple[Risk, float]]],
                           simulation_results: Dict[str, Dict[int, SimulationResult]], clustered_risks: Dict[int, List[int]],
                           sensitivity_results: Dict[str, Dict[str, float]], time_series_results: Dict[int, List[float]],
                           scenarios: Dict[str, Scenario], advanced_analysis: Dict, systemic_risks: Dict,
                           trigger_points: Dict, resilience_assessment: Dict, monte_carlo_results: Dict,
                           quantitative_summaries: Dict[str, Dict[str, float]], aggregate_impact: Dict,
                           tipping_points: List[Dict]) -> Dict:
    # Calculate risk categories distribution
    risk_categories = {}
    for risk in risks:
        if risk.category:
            risk_categories[risk.category] = risk_categories.get(risk.category, 0) + 1

    report = {
        "entity_name": entity,
        "risk_overview": {
            "total_risks": len(risks),
            "risk_categories": risk_categories,  # Add this line to include risk categories
            "high_impact_risks": [{"id": risk.id, "description": risk.description, "impact": risk.impact} 
                                  for risk in risks if risk.impact > 0.7]
        },
        "scenario_analysis": {
            scenario: {
                "summary": summarize_scenario_impact(impacts),
                "detailed_impacts": [{"risk_id": risk.id, "impact": impact} for risk, impact in impacts],
                "llm_analysis": advanced_analysis.get(scenario, {})
            } for scenario, impacts in scenario_impacts.items()
        },
        "monte_carlo_results": {
            scenario: {
                risk_id: {
                    "mean_impact": np.mean(result.impact_distribution),
                    "std_impact": np.std(result.impact_distribution),
                    "5th_percentile_impact": np.percentile(result.impact_distribution, 5),
                    "95th_percentile_impact": np.percentile(result.impact_distribution, 95)
                } for risk_id, result in scenario_results.items()
            } for scenario, scenario_results in monte_carlo_results.items()
        },
        "sensitivity_analysis": sensitivity_results,
        "risk_narratives": {risk.id: generate_risk_narrative(risk, scenario_impacts, monte_carlo_results) for risk in risks},
        "executive_insights": advanced_analysis.get("executive_insights", ""),
        "mitigation_strategies": generate_mitigation_strategies(risks, advanced_analysis),
        "systemic_risks": systemic_risks,
        "trigger_points": trigger_points,
        "resilience_assessment": resilience_assessment,
        "aggregate_impact": aggregate_impact,
        "tipping_points": tipping_points,
        "time_series_analysis": {
            risk_id: {
                "projections": projections,
                "trend": analyze_trend(projections)
            } for risk_id, projections in time_series_results.items()
        },
        "risk_clusters": {cluster_id: [risk_id for risk_id in cluster] for cluster_id, cluster in clustered_risks.items()},
        "risk_interactions": summarize_risk_interactions(risk_interactions)
    }
    
    return report

def perform_cross_entity_analysis(entity_reports: Dict[str, Dict]) -> Dict:
    cross_entity_analysis = {
        "risk_comparison": compare_risks_across_entities(entity_reports),
        "impact_comparison": compare_impacts_across_entities(entity_reports),
        "resilience_comparison": compare_resilience_across_entities(entity_reports),
        "common_risks": identify_common_risks(entity_reports),
        "entity_risk_profiles": generate_entity_risk_profiles(entity_reports),
        "aggregate_impact_comparison": compare_aggregate_impacts(entity_reports),
        "systemic_risk_overview": analyze_systemic_risks_across_entities(entity_reports),
        "scenario_sensitivity_comparison": compare_scenario_sensitivity(entity_reports),
        "tipping_point_analysis": analyze_tipping_points_across_entities(entity_reports)
    }
    
    return cross_entity_analysis

def compare_risks_across_entities(entity_reports: Dict[str, Dict]) -> Dict:
    risk_comparison = {}
    for entity, report in entity_reports.items():
        # Get risk categories from risk_overview, providing empty dict as default
        risk_overview = report.get("risk_overview", {})
        
        risk_comparison[entity] = {
            "total_risks": risk_overview.get("total_risks", 0),
            "high_impact_risks": len(risk_overview.get("high_impact_risks", [])),
            # Use risk_category_distribution if it exists, otherwise calculate from categorized_risks
            "risk_category_distribution": risk_overview.get("risk_categories", {})
        }
    return risk_comparison

def compare_impacts_across_entities(entity_reports: Dict[str, Dict]) -> Dict:
    impact_comparison = {}
    for entity, report in entity_reports.items():
        impact_comparison[entity] = {}
        scenario_analysis = report.get("scenario_analysis", {})
        
        for scenario, scenario_data in scenario_analysis.items():
            # Handle both possible data structures
            impacts = []
            if isinstance(scenario_data, dict) and "detailed_impacts" in scenario_data:
                impacts = scenario_data["detailed_impacts"]
            elif isinstance(scenario_data, list):
                impacts = scenario_data
                
            if impacts:
                try:
                    # Calculate average and max impact, handling different impact formats
                    impact_values = []
                    for impact in impacts:
                        if isinstance(impact, dict):
                            impact_values.append(impact.get("impact", 0))
                        elif isinstance(impact, tuple):
                            impact_values.append(impact[1])  # Assuming tuple format (Risk, float)
                            
                    impact_comparison[entity][scenario] = {
                        "average_impact": np.mean(impact_values) if impact_values else 0,
                        "max_impact": max(impact_values) if impact_values else 0
                    }
                except Exception as e:
                    print(f"Warning: Error processing impacts for {entity}/{scenario}: {e}")
                    impact_comparison[entity][scenario] = {
                        "average_impact": 0,
                        "max_impact": 0
                    }
    
    return impact_comparison

def compare_resilience_across_entities(entity_reports: Dict[str, Dict]) -> Dict:
    return {entity: report["resilience_assessment"] for entity, report in entity_reports.items()}

def identify_common_risks(entity_reports: Dict[str, Dict]) -> List[str]:
    all_risks = [set(report["risk_overview"]["high_impact_risks"]) for report in entity_reports.values()]
    return list(set.intersection(*all_risks))

def generate_entity_risk_profiles(entity_reports: Dict[str, Dict]) -> Dict:
    risk_profiles = {}
    for entity, report in entity_reports.items():
        risk_profiles[entity] = {
            "top_risks": sorted(report["risk_overview"]["high_impact_risks"], key=lambda x: x["impact"], reverse=True)[:5],
            "main_risk_categories": sorted(report["risk_overview"]["risk_categories"].items(), key=lambda x: x[1], reverse=True)[:3],
            "key_systemic_risks": list(report["systemic_risks"].keys())[:3]
        }
    return risk_profiles

def compare_aggregate_impacts(entity_reports: Dict[str, Dict]) -> Dict:
    return {entity: report["aggregate_impact"] for entity, report in entity_reports.items()}

def analyze_systemic_risks_across_entities(entity_reports: Dict[str, Dict]) -> Dict:
    all_systemic_risks = set()
    for report in entity_reports.values():
        all_systemic_risks.update(report["systemic_risks"].keys())
    
    systemic_risk_overview = {risk: [] for risk in all_systemic_risks}
    for entity, report in entity_reports.items():
        for risk in all_systemic_risks:
            if risk in report["systemic_risks"]:
                systemic_risk_overview[risk].append(entity)
    
    return systemic_risk_overview

def compare_scenario_sensitivity(entity_reports: Dict[str, Dict]) -> Dict:
    sensitivity_comparison = {}
    for entity, report in entity_reports.items():
        sensitivity_comparison[entity] = {
            scenario: max(sensitivities.values()) for scenario, sensitivities in report["sensitivity_analysis"].items()
        }
    return sensitivity_comparison

def analyze_tipping_points_across_entities(entity_reports: Dict[str, Dict]) -> Dict:
    tipping_point_analysis = {}
    for entity, report in entity_reports.items():
        tipping_point_analysis[entity] = {
            "num_tipping_points": len(report["tipping_points"]),
            "critical_tipping_point": max(report["tipping_points"], key=lambda tp: tp["aggregate_impact"]) if report["tipping_points"] else None
        }
    return tipping_point_analysis

# Helper functions

def generate_risk_narrative(risk: Risk, scenario_impacts: Dict[str, List[Tuple[Risk, float]]], monte_carlo_results: Dict[str, Dict[int, SimulationResult]]) -> str:
    narrative = f"Risk {risk.id}: {risk.description}\n\n"
    narrative += f"Category: {risk.category}\n"
    narrative += f"Base Impact: {risk.impact:.2f}\n\n"
    
    for scenario, impacts in scenario_impacts.items():
        impact = next((impact for r, impact in impacts if r.id == risk.id), None)
        if impact is not None:
            narrative += f"{scenario} Scenario Impact: {impact:.2f}\n"
    
    narrative += "\nMonte Carlo Simulation Results:\n"
    for scenario, results in monte_carlo_results.items():
        if risk.id in results:
            sim_result = results[risk.id]
            narrative += f"{scenario}:\n"
            narrative += f"  Mean Impact: {np.mean(sim_result.impact_distribution):.2f}\n"
            narrative += f"  95th Percentile Impact: {np.percentile(sim_result.impact_distribution, 95):.2f}\n"
    
    return narrative

def analyze_trend(projections: List[float]) -> Dict[str, float]:
    trend = np.polyfit(range(len(projections)), projections, 1)
    return {
        "slope": trend[0],
        "average": np.mean(projections),
        "volatility": np.std(projections)
    }









