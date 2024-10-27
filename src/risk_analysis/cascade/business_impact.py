from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BusinessMetrics:
    revenue: float = 0.0
    costs: float = 0.0
    market_share: float = 0.0
    operational_efficiency: float = 0.0
    supply_chain_disruption: float = 0.0

@dataclass
class BusinessImpact:
    direct_impact: BusinessMetrics
    propagated_impacts: List[BusinessMetrics]
    time_to_effect: int  # Days until impact manifests
    duration: int        # Duration of impact in days
    confidence: float    # Confidence level in the impact assessment

    def total_impact(self) -> BusinessMetrics:
        total = BusinessMetrics()
        # Add direct impact
        total.revenue = self.direct_impact.revenue
        total.costs = self.direct_impact.costs
        total.market_share = self.direct_impact.market_share
        total.operational_efficiency = self.direct_impact.operational_efficiency
        total.supply_chain_disruption = self.direct_impact.supply_chain_disruption
        
        # Add propagated impacts
        for impact in self.propagated_impacts:
            total.revenue += impact.revenue
            total.costs += impact.costs
            total.market_share += impact.market_share
            total.operational_efficiency += impact.operational_efficiency
            total.supply_chain_disruption += impact.supply_chain_disruption
            
        return total

@dataclass
class CascadeNode:
    risk_id: int
    business_impact: BusinessImpact
    children: List['CascadeNode']
    parent: Optional['CascadeNode'] = None
    
    def add_child(self, child: 'CascadeNode'):
        child.parent = self
        self.children.append(child)
        
    def get_total_impact(self) -> BusinessMetrics:
        total = self.business_impact.total_impact()
        for child in self.children:
            child_impact = child.get_total_impact()
            total.revenue += child_impact.revenue
            total.costs += child_impact.costs
            total.market_share += child_impact.market_share
            total.operational_efficiency += child_impact.operational_efficiency
            total.supply_chain_disruption += child_impact.supply_chain_disruption
        return total