from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np

class Entity(BaseModel):
    name: str
    description: str
    key_products: List[str]
    region: List[str]
    industry: Optional[str] = None
    parent_entity: Optional[str] = None
    sub_entities: List[str] = Field(default_factory=list)
    risk_profile: Dict[str, Union[float, str]] = Field(default_factory=dict)
    weight: float = Field(default=1.0)  # Add this line - default weight of 1.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "key_products": self.key_products,
            "region": self.region,
            "industry": self.industry,
            "parent_entity": self.parent_entity,
            "sub_entities": self.sub_entities,
            "weight": self.weight  # Add this line
        }
    
    def update_risk_profile(self, risk_data: dict):
        self.risk_profile.update(risk_data)

class Company(BaseModel):
    name: str
    industry: str
    region: List[str]
    key_products: List[str]
    key_dependencies: List[str] = Field(default_factory=list)
    entities: Dict[str, Entity] = Field(default_factory=dict)

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity
        if entity.parent_entity:
            parent = self.entities.get(entity.parent_entity)
            if parent:
                parent.sub_entities.append(entity.name)

    def get_entity(self, entity_name: str) -> Optional[Entity]:
        return self.entities.get(entity_name)

    def get_entity_hierarchy(self) -> Dict[str, List[str]]:
        hierarchy = {}
        for entity in self.entities.values():
            if not entity.parent_entity:
                hierarchy[entity.name] = self._get_sub_entities(entity.name)
        return hierarchy

    def _get_sub_entities(self, entity_name: str) -> List[Union[str, Dict[str, List]]]:
        entity = self.entities[entity_name]
        if not entity.sub_entities:
            return []
        return [{sub: self._get_sub_entities(sub)} for sub in entity.sub_entities]

    def aggregate_results(self, entity_results: Dict[str, Dict]) -> Dict[str, Dict]:
        aggregated = {}
        for entity_name, entity in self.entities.items():
            if not entity.parent_entity:
                aggregated[entity_name] = self._aggregate_entity(entity_name, entity_results)
        return aggregated

    def _aggregate_entity(self, entity_name: str, entity_results: Dict[str, Dict]) -> Dict:
        entity = self.entities[entity_name]
        result = entity_results.get(entity_name, {})
        
        for sub_entity in entity.sub_entities:
            sub_result = self._aggregate_entity(sub_entity, entity_results)
            for key, value in sub_result.items():
                if key in result:
                    if isinstance(value, (int, float)):
                        result[key] += value
                    elif isinstance(value, list):
                        result[key].extend(value)
                    elif isinstance(value, dict):
                        result[key].update(value)
                else:
                    result[key] = value
        
        return result

class Risk(BaseModel):
    id: int
    description: Optional[str] = "No description provided"
    category: Optional[str] = None
    subcategory: Optional[str] = None
    tertiary_category: Optional[str] = None
    likelihood: Optional[float] = None
    impact: Optional[float] = None
    time_horizon: Optional[str] = None
    industry_specific: Optional[bool] = False
    sasb_category: Optional[str] = None
    assessment_explanation: Optional[str] = None
    entity: Optional[str] = None

    @validator('likelihood', 'impact')
    def check_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v

    def to_dict(self) -> Dict:
        return self.dict()

class ExternalData(BaseModel):
    year: int
    gdp_growth: float
    population: int
    energy_demand: float
    carbon_price: float
    renewable_energy_share: float
    biodiversity_index: float
    deforestation_rate: float

@dataclass
class RiskInteraction:
    def __init__(self, risk1_id: int, risk2_id: int, interaction_score: float, interaction_type: str):
        self.risk1_id = risk1_id
        self.risk2_id = risk2_id
        self.interaction_score = interaction_score
        self.interaction_type = interaction_type
        self.full_analysis = None
        self.interaction_explanation = None
        self.compounding_effects = None
        self.mitigating_factors = None

class SimulationResult(BaseModel):
    risk_id: int
    scenario: str
    impact_distribution: List[float]
    likelihood_distribution: List[float]
    time_series_distributions: List[List[float]] = Field(default_factory=list)  # Add this line
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for the distributions."""
        return {
            "impact": {
                "mean": float(np.mean(self.impact_distribution)),
                "std": float(np.std(self.impact_distribution)),
                "min": float(np.min(self.impact_distribution)),
                "max": float(np.max(self.impact_distribution)),
                "median": float(np.median(self.impact_distribution)),
                "q25": float(np.percentile(self.impact_distribution, 25)),
                "q75": float(np.percentile(self.impact_distribution, 75))
            },
            "likelihood": {
                "mean": float(np.mean(self.likelihood_distribution)),
                "std": float(np.std(self.likelihood_distribution)),
                "min": float(np.min(self.likelihood_distribution)),
                "max": float(np.max(self.likelihood_distribution)),
                "median": float(np.median(self.likelihood_distribution)),
                "q25": float(np.percentile(self.likelihood_distribution, 25)),
                "q75": float(np.percentile(self.likelihood_distribution, 75))
            },
            "time_series": {
                "final_mean": float(np.mean([ts[-1] for ts in self.time_series_distributions])),
                "max_mean": float(np.mean([max(ts) for ts in self.time_series_distributions])),
                "volatility": float(np.mean([np.std(ts) for ts in self.time_series_distributions]))
            }
        }

class PESTELAnalysis(BaseModel):
    political: Dict[str, str]
    economic: Dict[str, str]
    social: Dict[str, str]
    technological: Dict[str, str]
    environmental: Dict[str, str]
    legal: Dict[str, str]
    overall_assessment: str

class SASBMaterialRisk(BaseModel):
    risk_id: int
    sasb_category: str
    description: str
    impact: float

    @validator('impact')
    def check_impact(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Impact must be between 0 and 1')
        return v

class SystemicRisk(BaseModel):
    risk_id: int
    description: str
    impact: float
    systemic_factor: str
    connected_risks: List[int]
    trigger_points: List[str]

    @validator('impact')
    def check_impact(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Impact must be between 0 and 1')
        return v

class Scenario(BaseModel):
    name: str
    description: str
    time_horizon: str
    temp_increase: float = Field(default=0.0)
    carbon_price: float = Field(default=0.0)
    renewable_energy: float = Field(default=0.0)
    policy_stringency: float = Field(default=0.0)
    biodiversity_loss: float = Field(default=0.0)
    ecosystem_degradation: float = Field(default=0.0)
    financial_stability: float = Field(default=0.0)
    supply_chain_disruption: float = Field(default=0.0)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return self.dict()

    def custom_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "temp_increase": self.temp_increase,
            "carbon_price": self.carbon_price,
            "renewable_energy": self.renewable_energy,
            "policy_stringency": self.policy_stringency,
            "biodiversity_loss": self.biodiversity_loss,
            "ecosystem_degradation": self.ecosystem_degradation,
            "financial_stability": self.financial_stability,
            "supply_chain_disruption": self.supply_chain_disruption,
        }
