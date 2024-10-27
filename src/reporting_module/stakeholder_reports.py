from typing import Dict, List
import json
import os
from src.config import OUTPUT_DIR

def generate_stakeholder_reports(main_report: Dict, company_industry: str) -> Dict[str, Dict]:
    """Generate different versions of the report for various stakeholders."""
    stakeholder_reports = {
        "board_executive": generate_board_executive_report(main_report, company_industry),
        "investors": generate_investor_report(main_report, company_industry),
        "regulators": generate_regulatory_report(main_report, company_industry),
        "public": generate_public_report(main_report, company_industry)
    }
    
    # Save reports to files
    for stakeholder, report in stakeholder_reports.items():
        file_path = os.path.join(OUTPUT_DIR, f"{stakeholder}_report.json")
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return stakeholder_reports

def generate_board_executive_report(main_report: Dict, company_industry: str) -> Dict:
    """Generate a focused report for board members and executives."""
    return {
        "executive_summary": main_report.get("executive_summary", "No summary available"),
        "entity_summaries": {
            entity_name: {
                "high_impact_risks": entity_report["risk_overview"]["high_impact_risks"][:5],
                "key_metrics": {
                    "total_risks": entity_report["risk_overview"]["total_risks"],
                    "aggregate_impact": entity_report.get("aggregate_impact", {}).get("mean", 0.0),
                    "resilience_score": entity_report.get("resilience_assessment", {}).get("overall_score", 0.0)
                },
                "executive_insights": entity_report.get("executive_insights", "No insights available")
            }
            for entity_name, entity_report in main_report.get("entity_reports", {}).items()
        },
        "cross_entity_analysis": main_report.get("cross_entity_analysis", {}),
        "key_recommendations": extract_key_recommendations(main_report)
    }

def generate_investor_report(main_report: Dict, company_industry: str) -> Dict:
    """Generate a report focused on financial and strategic implications."""
    return {
        "executive_summary": main_report.get("executive_summary", "No summary available"),
        "entity_analysis": {
            entity_name: {
                "risk_metrics": {
                    "total_risks": entity_report["risk_overview"]["total_risks"],
                    "high_impact_risks": len(entity_report["risk_overview"]["high_impact_risks"]),
                    "aggregate_impact": entity_report.get("aggregate_impact", {}).get("mean", 0.0)
                },
                "scenario_analysis": entity_report.get("scenario_analysis", {}),
                "monte_carlo_results": entity_report.get("monte_carlo_results", {})
            }
            for entity_name, entity_report in main_report.get("entity_reports", {}).items()
        },
        "cross_entity_insights": main_report.get("cross_entity_analysis", {}),
        "industry_comparison": extract_industry_comparison(main_report, company_industry)
    }

def generate_regulatory_report(main_report: Dict, company_industry: str) -> Dict:
    """Generate a detailed compliance-focused report."""
    return {
        "methodology_overview": {
            "risk_assessment_approach": "Comprehensive multi-factor analysis including Monte Carlo simulations",
            "data_sources": ["Historical climate data", "Industry benchmarks", "Regulatory guidelines"],
            "scenario_analysis": "IPCC-aligned climate scenarios"
        },
        "entity_compliance": {
            entity_name: {
                "risk_assessment": entity_report["risk_overview"],
                "mitigation_strategies": entity_report.get("mitigation_strategies", {}),
                "scenario_analysis": entity_report.get("scenario_analysis", {}),
                "resilience_metrics": entity_report.get("resilience_assessment", {})
            }
            for entity_name, entity_report in main_report.get("entity_reports", {}).items()
        },
        "cross_entity_compliance": main_report.get("cross_entity_analysis", {}),
        "industry_specific_considerations": extract_industry_considerations(company_industry)
    }

def generate_public_report(main_report: Dict, company_industry: str) -> Dict:
    """Generate a simplified report for public consumption."""
    return {
        "overview": simplify_executive_summary(main_report.get("executive_summary", "")),
        "key_findings": {
            "total_risks_assessed": sum(
                entity_report["risk_overview"]["total_risks"]
                for entity_report in main_report.get("entity_reports", {}).values()
            ),
            "main_risk_areas": extract_main_risk_areas(main_report),
            "mitigation_highlights": extract_mitigation_highlights(main_report)
        },
        "entity_highlights": {
            entity_name: {
                "key_risks": entity_report["risk_overview"]["high_impact_risks"][:3],
                "main_actions": extract_entity_actions(entity_report)
            }
            for entity_name, entity_report in main_report.get("entity_reports", {}).items()
        }
    }

# Helper functions
def extract_key_recommendations(main_report: Dict) -> List[str]:
    """Extract and prioritize key recommendations from the report."""
    recommendations = []
    for entity_report in main_report.get("entity_reports", {}).values():
        if "executive_insights" in entity_report:
            recommendations.extend([
                insight for insight in entity_report["executive_insights"].split("\n")
                if "recommend" in insight.lower()
            ])
    return list(set(recommendations))[:5]

def extract_industry_comparison(main_report: Dict, company_industry: str) -> Dict:
    """Extract industry-specific comparison metrics."""
    return {
        "industry_average_risks": 0.0,  # Placeholder - would need industry data
        "company_position": "Above average",  # Placeholder - would need industry data
        "key_differentiators": []  # Placeholder - would need industry data
    }

def extract_industry_considerations(company_industry: str) -> Dict:
    """Extract industry-specific regulatory considerations."""
    return {
        "industry_specific_regulations": [],  # Placeholder - would need regulatory data
        "compliance_requirements": [],  # Placeholder - would need regulatory data
        "reporting_obligations": []  # Placeholder - would need regulatory data
    }

def simplify_executive_summary(summary: str) -> str:
    """Simplify the executive summary for public consumption."""
    # Remove technical details and simplify language
    return summary.split("\n")[0] if summary else "No summary available"

def extract_main_risk_areas(main_report: Dict) -> List[str]:
    """Extract main risk areas from the report."""
    risk_areas = set()
    for entity_report in main_report.get("entity_reports", {}).values():
        for risk in entity_report["risk_overview"]["high_impact_risks"]:
            if "category" in risk:
                risk_areas.add(risk["category"])
    return list(risk_areas)

def extract_mitigation_highlights(main_report: Dict) -> List[str]:
    """Extract key mitigation strategies."""
    highlights = []
    for entity_report in main_report.get("entity_reports", {}).values():
        if "mitigation_strategies" in entity_report:
            highlights.extend(entity_report["mitigation_strategies"].values())
    return list(set(highlights))[:5]

def extract_entity_actions(entity_report: Dict) -> List[str]:
    """Extract key actions for an entity."""
    actions = []
    if "mitigation_strategies" in entity_report:
        actions.extend(list(entity_report["mitigation_strategies"].values())[:3])
    return actions
