from typing import Dict, List, Optional
import logging
from src.models import Risk, Scenario
from src.risk_analysis.cascade.business_impact import BusinessMetrics, BusinessImpact, CascadeNode
from src.risk_analysis.cascade.propagation import PropagationEngine

logger = logging.getLogger(__name__)

class CascadeAnalyzer:
    def __init__(
        self,
        risks: List[Risk],
        risk_network: Dict,
        scenario: Scenario,
        correlation_matrix: Optional[Dict[Tuple[int, int], float]] = None
    ):
        self.risks = risks
        self.risk_network = risk_network
        self.scenario = scenario
        self.correlation_matrix = correlation_matrix or {}
        self.propagation_engine = PropagationEngine(risk_network, scenario)
        
    def analyze_cascade(self, initial_risk: Risk) -> CascadeNode:
        """Analyze cascade effects starting from an initial risk"""
        logger.info(f"Starting cascade analysis for risk {initial_risk.id}")
        
        # Create root node with initial impact
        root_impact = self._calculate_initial_impact(initial_risk)
        root_node = CascadeNode(
            risk_id=initial_risk.id,
            business_impact=root_impact,
            children=[]
        )
        
        # Build cascade tree
        self._build_cascade_tree(root_node, visited=set([initial_risk.id]))
        
        return root_node
        
    def _calculate_initial_impact(self, risk: Risk) -> BusinessImpact:
        """Calculate initial business impact for a risk"""
        # Base impact calculations
        revenue_impact = -risk.impact * 0.1  # 10% of impact score
        cost_impact = risk.impact * 0.15     # 15% of impact score
        
        # Create business metrics
        direct_impact = BusinessMetrics(
            revenue=revenue_impact,
            costs=cost_impact,
            market_share=-risk.impact * 0.05,
            operational_efficiency=-risk.impact * 0.2,
            supply_chain_disruption=risk.impact * 0.3
        )
        
        # Determine time factors based on risk category
        if risk.category == "Physical Risks":
            time_to_effect = 0  # Immediate
            duration = 90       # 3 months
        elif risk.category == "Transition Risks":
            time_to_effect = 30  # 1 month
            duration = 365      # 1 year
        else:
            time_to_effect = 60  # 2 months
            duration = 730      # 2 years
            
        return BusinessImpact(
            direct_impact=direct_impact,
            propagated_impacts=[],
            time_to_effect=time_to_effect,
            duration=duration,
            confidence=0.8
        )
        
    def _build_cascade_tree(self, current_node: CascadeNode, visited: set):
        """Recursively build cascade tree"""
        current_risk = next(r for r in self.risks if r.id == current_node.risk_id)
        
        # Get connected risks
        for connected_risk in self.risks:
            if connected_risk.id in visited:
                continue
                
            # Check if risks are connected in network
            if not self._are_risks_connected(current_risk, connected_risk):
                continue
                
            # Calculate propagated impact
            propagated_impact = self._calculate_propagated_impact(
                current_risk,
                connected_risk,
                current_node.business_impact
            )
            
            # Create child node
            child_node = CascadeNode(
                risk_id=connected_risk.id,
                business_impact=propagated_impact,
                children=[]
            )
            current_node.add_child(child_node)
            
            # Recursively build tree (with depth limit)
            if len(visited) < 5:  # Limit cascade depth
                visited.add(connected_risk.id)
                self._build_cascade_tree(child_node, visited)
                visited.remove(connected_risk.id)
                
    def _are_risks_connected(self, risk1: Risk, risk2: Risk) -> bool:
        """Check if two risks are connected in the network"""
        return (risk1.id, risk2.id) in self.correlation_matrix
        
    def _calculate_propagated_impact(
        self,
        source_risk: Risk,
        target_risk: Risk,
        source_impact: BusinessImpact
    ) -> BusinessImpact:
        """Calculate propagated business impact"""
        # Get base correlation
        correlation = self.correlation_matrix.get((source_risk.id, target_risk.id), 0.0)
        
        # Calculate propagation strength
        strength = self.propagation_engine.calculate_propagation_strength(
            source_risk,
            target_risk,
            correlation
        )
        
        # Calculate propagated metrics
        propagated_metrics = BusinessMetrics(
            revenue=source_impact.direct_impact.revenue * strength * 0.7,
            costs=source_impact.direct_impact.costs * strength * 0.8,
            market_share=source_impact.direct_impact.market_share * strength * 0.6,
            operational_efficiency=source_impact.direct_impact.operational_efficiency * strength * 0.9,
            supply_chain_disruption=source_impact.direct_impact.supply_chain_disruption * strength * 0.85
        )
        
        # Get propagation delay
        delay = self.propagation_engine.get_propagation_delay(source_risk, target_risk)
        
        return BusinessImpact(
            direct_impact=BusinessMetrics(),  # No direct impact
            propagated_impacts=[propagated_metrics],
            time_to_effect=source_impact.time_to_effect + delay,
            duration=int(source_impact.duration * 0.7),  # 70% of source duration
            confidence=0.6  # Lower confidence in propagated impacts
        )