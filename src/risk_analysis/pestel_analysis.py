# We have brought legacy lost functions up to here, not in this file

from typing import List, Dict, Union
from src.models import Risk, ExternalData

def perform_pestel_analysis(risks: List[Risk], external_data: Union[Dict[str, ExternalData], List[ExternalData]]) -> Dict[str, List[Dict[str, str]]]:
    pestel_categories = {
        "Political": [],
        "Economic": [],
        "Social": [],
        "Technological": [],
        "Environmental": [],
        "Legal": []
    }
    
    for risk in risks:
        category = categorize_risk_pestel(risk)
        pestel_categories[category].append({
            "risk_id": risk.id,
            "description": risk.description or "No description provided",
            "impact": risk.impact if risk.impact is not None else "Not specified"
        })
    
    # Enrich with external data
    enrich_pestel_with_external_data(pestel_categories, external_data)
    
    return pestel_categories

def categorize_risk_pestel(risk: Risk) -> str:
    description = risk.description.lower() if risk.description else ""
    if "regulation" in description or "policy" in description:
        return "Political"
    elif "economic" in description or "financial" in description:
        return "Economic"
    elif "social" in description or "demographic" in description:
        return "Social"
    elif "technology" in description or "innovation" in description:
        return "Technological"
    elif "environmental" in description or "climate" in description:
        return "Environmental"
    elif "legal" in description or "liability" in description:
        return "Legal"
    else:
        return "Environmental"  # Default category for climate risks

def enrich_pestel_with_external_data(pestel_categories: Dict[str, List[Dict[str, str]]], external_data: Union[Dict[str, ExternalData], List[ExternalData]]):
    # Add relevant external data to each PESTEL category
    if isinstance(external_data, dict):
        latest_data = list(external_data.values())[-1]  # Assuming the last entry is the most recent
    elif isinstance(external_data, list):
        latest_data = external_data[-1]  # Assuming the last entry is the most recent
    else:
        raise ValueError("external_data must be either a dictionary or a list")
    
    if hasattr(latest_data, 'gdp_growth'):
        pestel_categories["Economic"].append({
            "factor": "GDP Growth",
            "value": f"{latest_data.gdp_growth}%"
        })
    
    if hasattr(latest_data, 'population'):
        pestel_categories["Social"].append({
            "factor": "Population",
            "value": str(latest_data.population)
        })
    
    if hasattr(latest_data, 'energy_demand'):
        pestel_categories["Environmental"].append({
            "factor": "Energy Demand",
            "value": f"{latest_data.energy_demand} TWh"
        })
