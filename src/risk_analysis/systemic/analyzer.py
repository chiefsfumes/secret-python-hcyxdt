from typing import Dict, List, Optional
import networkx as nx
import numpy as np
from dataclasses import dataclass
from src.models import Risk, Scenario
from src.risk_analysis.cascade.business_impact import BusinessMetrics, CascadeNode

@dataclass
class SystemicRiskMetrics:
    risk_concentration: float
    network_stability: float
    intervention_priority: float
    cascade_probability: float
    system_vulnerability: float

class SystemicRiskAnalyzer:
    def __init__(
        self,
        risks: List[Risk],
        risk_network: nx.Graph,
        cascade_results: Dict[int, CascadeNode],
        scenario: Scenario
    ):
        self.risks = risks
        self.risk_network = risk_network
        self.cascade_results = cascade_results
        self.scenario = scenario
        
    def analyze_systemic_risks(self) -> Dict[str, SystemicRiskMetrics]:
        """Perform comprehensive systemic risk analysis"""
        
        # Calculate network metrics
        centrality = nx.eigenvector_centrality(self.risk_network, weight='weight')
        clustering = nx.average_clustering(self.risk_network, weight='weight')
        
        # Analyze cascade patterns
        cascade_metrics = self._analyze_cascade_patterns()
        
        # Calculate risk concentration
        risk_concentration = self._calculate_risk_concentration(centrality)
        
        # Assess network stability
        network_stability = self._assess_network_stability(clustering)
        
        # Identify intervention points
        intervention_priorities = self._identify_intervention_points(
            centrality,
            cascade_metrics
        )
        
        # Calculate cascade probabilities
        cascade_probs = self._calculate_cascade_probabilities()
        
        # Assess system vulnerability
        vulnerability = self._assess_system_vulnerability(
            risk_concentration,
            network_stability,
            cascade_probs
        )
        
        return SystemicRiskMetrics(
            risk_concentration=risk_concentration,
            network_stability=network_stability,
            intervention_priority=max(intervention_priorities.values()),
            cascade_probability=max(cascade_probs.values()),
            system_vulnerability=vulnerability
        )
        
    def _analyze_cascade_patterns(self) -> Dict:
        """Analyze patterns in cascade results"""
        metrics = {}
        
        for risk_id, cascade in self.cascade_results.items():
            # Calculate cascade depth
            depth = self._calculate_cascade_depth(cascade)
            
            # Calculate total impact
            total_impact = cascade.get_total_impact()
            
            # Calculate impact amplification
            amplification = self._calculate_impact_amplification(
                cascade.business_impact.direct_impact,
                total_impact
            )
            
            metrics[risk_id] = {
                "depth": depth,
                "amplification": amplification,
                "total_impact": total_impact
            }
            
        return metrics
        
    def _calculate_cascade_depth(self, node: CascadeNode) -> int:
        """Calculate maximum depth of cascade tree"""
        if not node.children:
            return 1
        return 1 + max(self._calculate_cascade_depth(child) for child in node.children)
        
    def _calculate_impact_amplification(
        self,
        direct_impact: BusinessMetrics,
        total_impact: BusinessMetrics
    ) -> float:
        """Calculate impact amplification factor"""
        # Use revenue as primary metric for amplification
        if direct_impact.revenue == 0:
            return 1.0
        return abs(total_impact.revenue / direct_impact.revenue)
        
    def _calculate_risk_concentration(self, centrality: Dict[int, float]) -> float:
        """Calculate risk concentration based on network centrality"""
        values = list(centrality.values())
        return np.std(values) / np.mean(values) if values else 0
        
    def _assess_network_stability(self, clustering: float) -> float:
        """Assess network stability using clustering coefficient"""
        # Higher clustering generally indicates more stable network
        return clustering
        
    def _identify_intervention_points(
        self,
        centrality: Dict[int, float],
        cascade_metrics: Dict[int, Dict]
    ) -> Dict[int, float]:
        """Identify critical intervention points"""
        intervention_priorities = {}
        
        for risk_id in centrality:
            # Combine centrality with cascade metrics
            cascade_impact = cascade_metrics[risk_id]["amplification"]
            intervention_priorities[risk_id] = centrality[risk_id] * cascade_impact
            
        return intervention_priorities
        
    def _calculate_cascade_probabilities(self) -> Dict[int, float]:
        """Calculate probability of cascade failures"""
        probs = {}
        
        for risk_id, cascade in self.cascade_results.items():
            # Calculate based on network structure and impact amplification
            risk = next(r for r in self.risks if r.id == risk_id)
            base_prob = risk.likelihood
            
            # Adjust for scenario
            if self.scenario.name == "Delayed Transition":
                base_prob *= 1.3  # 30% higher probability
            
            # Adjust for cascade metrics
            metrics = self._analyze_cascade_patterns()[risk_id]
            cascade_factor = metrics["amplification"] * 0.1  # 10% per amplification
            
            probs[risk_id] = min(base_prob * (1 + cascade_factor), 1.0)
            
        return probs
        
    def _assess_system_vulnerability(
        self,
        risk_concentration: float,
        network_stability: float,
        cascade_probs: Dict[int, float]
    ) -> float:
        """Assess overall system vulnerability"""
        # Combine metrics into overall vulnerability score
        max_cascade_prob = max(cascade_probs.values())
        
        vulnerability = (
            0.4 * risk_concentration +  # 40% weight
            0.3 * (1 - network_stability) +  # 30% weight
            0.3 * max_cascade_prob  # 30% weight
        )
        
        return vulnerability