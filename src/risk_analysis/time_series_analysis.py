from typing import List, Dict, Tuple, Any
from src.models import Risk, ExternalData
from src.config import TIME_SERIES_HORIZON
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from scipy.stats import genextreme, gamma, expon
import logging

logger = logging.getLogger(__name__)

def time_series_analysis(risks_by_entity: Dict[str, List[Risk]], external_data: Dict[str, ExternalData]) -> Dict[str, Dict[int, List[float]]]:
    logger.info(f"Starting time series analysis for {len(risks_by_entity)} entities")
    time_series_results_by_entity = {}
    
    for entity, risks in risks_by_entity.items():
        logger.debug(f"Processing entity: {entity} with {len(risks)} risks")
        if not risks:
            logger.warning(f"No risks provided for entity: {entity}")
            continue
            
        time_series_results = {}
        for risk in risks:
            try:
                projections = project_risk_impact_advanced(risk, external_data)
                time_series_results[risk.id] = projections
                logger.debug(f"Generated projections for risk {risk.id}: {projections[:3]}...")
            except Exception as e:
                logger.error(f"Error projecting risk {risk.id}: {str(e)}")
                continue
                
        if time_series_results:
            time_series_results_by_entity[entity] = time_series_results
        else:
            logger.warning(f"No valid time series results generated for entity: {entity}")
            
    return time_series_results_by_entity

def project_risk_impact_advanced(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Advanced risk projection using category-specific modeling techniques.
    """
    logger.debug(f"Starting advanced projection for risk {risk.id}")
    
    try:
        if risk.category == "Physical Risks" and risk.subcategory == "Acute Physical Risks":
            return project_extreme_events(risk, external_data)
            
        elif risk.category == "Physical Risks" and risk.subcategory == "Chronic Physical Risks":
            return project_chronic_physical(risk, external_data)
            
        elif risk.category == "Transition Risks":
            return project_transition_risk(risk, external_data)
            
        elif risk.category == "Nature-related risks":
            return project_nature_risk(risk, external_data)
            
        elif risk.category == "Systemic Risks":
            return project_systemic_risk(risk, external_data)
            
        else:
            # Fallback to standard ARIMA
            return project_risk_impact_arima(risk, external_data)
            
    except Exception as e:
        logger.error(f"Error in advanced projection for risk {risk.id}: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def project_extreme_events(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Project acute physical risks using Extreme Value Theory (EVT).
    Uses Generalized Extreme Value (GEV) distribution for modeling extreme events.
    """
    try:
        # Calculate historical impacts
        historical_impacts = calculate_physical_acute_impact(risk, external_data)
        
        # Fit GEV distribution to historical extremes
        shape, loc, scale = genextreme.fit(historical_impacts)
        
        # Generate future projections
        raw_projections = genextreme.rvs(
            shape, loc=loc, scale=scale, 
            size=TIME_SERIES_HORIZON
        )
        
        # Apply trend adjustment based on temperature anomaly
        temp_trend = np.array([data.global_temp_anomaly for data in external_data.values()])
        trend_coefficient = np.polyfit(range(len(temp_trend)), temp_trend, 1)[0]
        
        # Adjust projections with trend and constraints
        adjusted_projections = []
        for i, proj in enumerate(raw_projections):
            trend_factor = 1 + (trend_coefficient * i * 0.1)  # 10% impact per trend unit
            adjusted_proj = proj * trend_factor
            adjusted_projections.append(adjusted_proj)
        
        # Normalize and validate
        return validate_and_normalize_forecast(adjusted_projections, risk)
        
    except Exception as e:
        logger.error(f"Error in extreme event projection: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def project_chronic_physical(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Project chronic physical risks using trend analysis and gamma distribution.
    Suitable for gradually increasing impacts.
    """
    try:
        historical_impacts = calculate_physical_chronic_impact(risk, external_data)
        
        # Fit gamma distribution for positive-skewed, continuous impacts
        alpha, loc, beta = gamma.fit(historical_impacts)
        
        # Generate base projections
        base_projections = gamma.rvs(
            alpha, loc=loc, scale=beta, 
            size=TIME_SERIES_HORIZON
        )
        
        # Add long-term trend based on multiple factors
        temp_trend = np.array([data.global_temp_anomaly for data in external_data.values()])
        pop_trend = np.array([data.population for data in external_data.values()])
        
        # Calculate combined trend
        temp_coef = np.polyfit(range(len(temp_trend)), temp_trend, 1)[0]
        pop_coef = np.polyfit(range(len(pop_trend)), pop_trend/1e9, 1)[0]  # normalized
        
        # Apply trends
        adjusted_projections = []
        for i, proj in enumerate(base_projections):
            temp_factor = 1 + (temp_coef * i * 0.15)
            pop_factor = 1 + (pop_coef * i * 0.05)
            adjusted_proj = proj * temp_factor * pop_factor
            adjusted_projections.append(adjusted_proj)
        
        return validate_and_normalize_forecast(adjusted_projections, risk)
        
    except Exception as e:
        logger.error(f"Error in chronic physical projection: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def project_transition_risk(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Project transition risks using regime-switching models.
    Accounts for policy changes and market transitions.
    """
    try:
        historical_impacts = calculate_category_specific_impacts(risk, external_data)
        
        # Identify potential regime changes based on policy and market indicators
        policy_changes = detect_regime_changes([d.policy_stringency_index for d in external_data.values()])
        market_changes = detect_regime_changes([d.market_volatility for d in external_data.values()])
        
        # Combine with ARIMA for base projection
        arima_params = get_category_arima_params(risk)
        model = ARIMA(historical_impacts, order=arima_params['order'])
        model_fit = model.fit()
        base_forecast = model_fit.forecast(steps=TIME_SERIES_HORIZON)
        
        # Adjust for regime changes
        adjusted_projections = adjust_for_regime_changes(
            base_forecast, 
            policy_changes, 
            market_changes,
            risk.subcategory
        )
        
        return validate_and_normalize_forecast(adjusted_projections, risk)
        
    except Exception as e:
        logger.error(f"Error in transition risk projection: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def detect_regime_changes(data: List[float]) -> List[Tuple[int, float]]:
    """Detect significant changes in time series data."""
    changes = []
    window_size = 3
    
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        current = data[i]
        
        # Calculate z-score for change detection
        z_score = abs(current - np.mean(window)) / np.std(window)
        if z_score > 2:  # Significant change threshold
            changes.append((i, current))
            
    return changes

def adjust_for_regime_changes(
    base_forecast: np.ndarray,
    policy_changes: List[Tuple[int, float]],
    market_changes: List[Tuple[int, float]],
    subcategory: str
) -> List[float]:
    """Adjust forecasts based on detected regime changes."""
    adjusted = base_forecast.copy()
    
    # Apply different adjustments based on subcategory
    if subcategory == "Policy and Legal Risks":
        policy_weight = 0.7
        market_weight = 0.3
    elif subcategory == "Market Risks":
        policy_weight = 0.3
        market_weight = 0.7
    else:
        policy_weight = 0.5
        market_weight = 0.5
        
    # Apply adjustments
    for i in range(len(adjusted)):
        policy_factor = calculate_change_factor(i, policy_changes)
        market_factor = calculate_change_factor(i, market_changes)
        
        combined_factor = (policy_factor * policy_weight + 
                         market_factor * market_weight)
        adjusted[i] *= combined_factor
        
    return list(adjusted)

def calculate_change_factor(index: int, changes: List[Tuple[int, float]]) -> float:
    """Calculate adjustment factor based on regime changes."""
    if not changes:
        return 1.0
        
    # Find relevant changes
    recent_changes = [c for c in changes if c[0] <= index]
    if not recent_changes:
        return 1.0
        
    # Calculate decay factor based on time since change
    factors = []
    for change_idx, change_val in recent_changes:
        time_since_change = index - change_idx
        decay = np.exp(-0.1 * time_since_change)  # Exponential decay
        factors.append(change_val * decay)
        
    return 1.0 + np.mean(factors) * 0.1  # 10% maximum impact

def validate_and_normalize_forecast(forecast: List[float], risk: Risk) -> List[float]:
    """Validate and normalize forecasted values."""
    # Apply category-specific constraints
    forecast = validate_arima_forecast(forecast, risk)
    
    # Ensure values are between 0 and 1
    forecast = np.clip(forecast, 0, 1)
    
    return list(forecast)

def project_risk_impact_arima(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    logger.debug(f"Projecting ARIMA impact for risk {risk.id}")
    
    if not external_data:
        logger.warning(f"No external data provided for risk {risk.id}")
        return [risk.impact] * TIME_SERIES_HORIZON
    
    try:
        # Get category-specific ARIMA parameters and historical impacts
        arima_params = get_category_arima_params(risk)
        historical_impacts = calculate_category_specific_impacts(risk, external_data)
        
        if len(historical_impacts) < arima_params['min_samples']:
            logger.warning(f"Insufficient historical data for risk {risk.id}. Using simple projection.")
            return [risk.impact] * TIME_SERIES_HORIZON
        
        # Fit ARIMA model with category-specific parameters
        model = ARIMA(historical_impacts, order=arima_params['order'])
        model_fit = model.fit()
        
        # Make future projections
        forecast = model_fit.forecast(steps=TIME_SERIES_HORIZON)
        forecast = np.clip(forecast, 0, 1)  # Ensure values stay between 0 and 1
        
        return list(forecast)
        
    except Exception as e:
        logger.error(f"Error in ARIMA projection for risk {risk.id}: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def get_category_arima_params(risk: Risk) -> Dict:
    """
    Get ARIMA parameters based on risk category.
    
    Parameters:
    - p: The number of lag observations (lag order)
    - d: The number of times the raw observations are differenced (degree of differencing)
    - q: The size of the moving average window (order of moving average)
    - min_samples: Minimum number of historical data points needed
    """
    
    if risk.category == "Physical Risks":
        if risk.subcategory == "Acute Physical Risks":
            # Higher p for autocorrelation, lower d as shocks are temporary
            return {
                'order': (2, 1, 1),  # More emphasis on recent patterns
                'min_samples': 5,    # Need more historical data for volatile risks
                'description': "Captures sudden changes and extreme events"
            }
        else:  # Chronic Physical Risks
            # Higher d for long-term trends, lower p as changes are gradual
            return {
                'order': (1, 2, 1),  # More emphasis on long-term trends
                'min_samples': 4,
                'description': "Models gradual, persistent changes"
            }
            
    elif risk.category == "Transition Risks":
        if risk.subcategory == "Policy and Legal Risks":
            # Higher q to capture policy implementation delays
            return {
                'order': (1, 1, 2),  # More emphasis on moving averages
                'min_samples': 4,
                'description': "Accounts for policy implementation lags"
            }
        elif risk.subcategory == "Technology Risks":
            # Higher p to capture technology adoption patterns
            return {
                'order': (2, 1, 1),
                'min_samples': 4,
                'description': "Models technology adoption patterns"
            }
        elif risk.subcategory == "Market Risks":
            # Balanced parameters for market dynamics
            return {
                'order': (1, 1, 1),
                'min_samples': 4,
                'description': "Balanced model for market dynamics"
            }
        else:  # Reputation Risks
            # Higher q to capture lasting effects of reputation changes
            return {
                'order': (1, 1, 2),
                'min_samples': 4,
                'description': "Models persistent reputation effects"
            }
            
    elif risk.category == "Nature-related risks":
        # Higher d for long-term environmental trends
        return {
            'order': (1, 2, 1),
            'min_samples': 5,
            'description': "Captures long-term environmental degradation"
        }
        
    elif risk.category == "Systemic Risks":
        # Complex model for interconnected risks
        return {
            'order': (2, 2, 2),
            'min_samples': 6,
            'description': "Models complex systemic interactions"
        }
        
    else:
        # Default parameters for undefined categories
        return {
            'order': (1, 1, 1),
            'min_samples': 4,
            'description': "Default balanced model"
        }

def validate_arima_forecast(forecast: List[float], risk: Risk) -> List[float]:
    """
    Validate and adjust ARIMA forecasts based on risk category constraints.
    """
    category_constraints = {
        "Physical Risks": {
            "max_step_change": 0.2,  # Maximum change between time steps
            "trend_direction": "increasing",  # Expected trend direction
            "volatility_threshold": 0.15  # Maximum allowed volatility
        },
        "Transition Risks": {
            "max_step_change": 0.15,
            "trend_direction": "variable",
            "volatility_threshold": 0.1
        },
        "Nature-related risks": {
            "max_step_change": 0.1,
            "trend_direction": "increasing",
            "volatility_threshold": 0.08
        },
        "Systemic Risks": {
            "max_step_change": 0.25,
            "trend_direction": "variable",
            "volatility_threshold": 0.2
        }
    }
    
    constraints = category_constraints.get(risk.category, {
        "max_step_change": 0.15,
        "trend_direction": "variable",
        "volatility_threshold": 0.12
    })
    
    # Apply constraints
    adjusted_forecast = forecast.copy()
    for i in range(1, len(adjusted_forecast)):
        # Limit step changes
        max_change = constraints["max_step_change"]
        current_change = abs(adjusted_forecast[i] - adjusted_forecast[i-1])
        if current_change > max_change:
            if adjusted_forecast[i] > adjusted_forecast[i-1]:
                adjusted_forecast[i] = adjusted_forecast[i-1] + max_change
            else:
                adjusted_forecast[i] = adjusted_forecast[i-1] - max_change
    
    # Ensure values stay within bounds
    adjusted_forecast = np.clip(adjusted_forecast, 0, 1)
    
    return list(adjusted_forecast)

def calculate_category_specific_impacts(risk: Risk, data: Dict[str, ExternalData]) -> List[float]:
    """Calculate historical impacts based on risk category."""
    impacts = []
    
    for period_data in data.values():
        if risk.category == "Physical Risks":
            if risk.subcategory == "Acute Physical Risks":
                # More sensitive to extreme changes
                impact = calculate_physical_acute_impact(risk, period_data)
            else:  # Chronic Physical Risks
                # More gradual response to long-term trends
                impact = calculate_physical_chronic_impact(risk, period_data)
                
        elif risk.category == "Transition Risks":
            if risk.subcategory == "Policy and Legal Risks":
                impact = calculate_policy_legal_impact(risk, period_data)
            elif risk.subcategory == "Technology Risks":
                impact = calculate_technology_impact(risk, period_data)
            elif risk.subcategory == "Market Risks":
                impact = calculate_market_impact(risk, period_data)
            else:  # Reputation Risks
                impact = calculate_reputation_impact(risk, period_data)
                
        elif risk.category == "Nature-related risks":
            impact = calculate_nature_related_impact(risk, period_data)
            
        elif risk.category == "Systemic Risks":
            impact = calculate_systemic_impact(risk, period_data)
            
        else:
            # Fallback to original calculation
            impact = calculate_default_impact(risk, period_data)
            
        impacts.append(impact)
    
    return impacts

def calculate_physical_acute_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for acute physical risks."""
    try:
        base_impact = risk.impact
        # Acute physical risks are more sensitive to rapid changes
        energy_volatility = abs(data.energy_demand / 1e5 - 0.5) * 0.2
        gdp_volatility = abs(data.gdp_growth - 2) * 0.1
        
        impact = base_impact * (1 + energy_volatility + gdp_volatility)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in acute physical impact calculation: {str(e)}")
        return risk.impact

def calculate_physical_chronic_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for chronic physical risks."""
    try:
        base_impact = risk.impact
        # Chronic risks respond more to long-term trends
        population_pressure = (data.population / 1e10) * 0.15
        energy_trend = (data.energy_demand / 1e5) * 0.1
        
        impact = base_impact * (1 + population_pressure + energy_trend)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in chronic physical impact calculation: {str(e)}")
        return risk.impact

def calculate_policy_legal_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for policy and legal risks."""
    try:
        base_impact = risk.impact
        # Policy risks are more sensitive to GDP changes
        gdp_factor = 1 + abs(data.gdp_growth - 2) * 0.15
        population_factor = 1 + (data.population / 1e10) * 0.05
        
        impact = base_impact * gdp_factor * population_factor
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in policy/legal impact calculation: {str(e)}")
        return risk.impact

# ... Additional category-specific calculation functions ...

def calculate_default_impact(risk: Risk, data: ExternalData) -> float:
    """Original calculation method as fallback."""
    try:
        base_impact = risk.impact
        gdp_factor = 1 + (data.gdp_growth - 2) * 0.05
        population_factor = 1 + (data.population / 1e10) * 0.1
        energy_factor = 1 + (data.energy_demand / 1e5) * 0.05
        
        impact = base_impact * gdp_factor * population_factor * energy_factor
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in default impact calculation: {str(e)}")
        return risk.impact

def analyze_impact_trends(time_series_results: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Analyze trends in time series data for each entity and risk.
    
    Args:
        time_series_results: Dictionary of time series data by entity and risk
        
    Returns:
        Dictionary of trend analysis results
    """
    trends = {}
    for entity_name, entity_risks in time_series_results.items():
        trends[entity_name] = {}
        for risk_id, time_series in entity_risks.items():
            if len(time_series) > 1:
                # Calculate linear trend
                x = np.arange(len(time_series))
                slope, intercept = np.polyfit(x, time_series, 1)
                
                # Calculate volatility
                volatility = np.std(time_series)
                
                # Calculate momentum (rate of change)
                momentum = np.mean(np.diff(time_series))
                
                trends[entity_name][risk_id] = {
                    "slope": slope,
                    "intercept": intercept,
                    "volatility": volatility,
                    "momentum": momentum
                }
    return trends

def identify_critical_periods(
    time_series_results: Dict[str, Dict[int, List[float]]], 
    threshold: float = 0.7
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Identify periods where risk levels exceed critical thresholds.
    
    Args:
        time_series_results: Dictionary of time series data by entity and risk
        threshold: Critical threshold level (0-1)
        
    Returns:
        Dictionary of critical periods by entity and risk
    """
    critical_periods = {}
    for entity_name, entity_risks in time_series_results.items():
        critical_periods[entity_name] = {}
        for risk_id, time_series in entity_risks.items():
            critical_periods[entity_name][risk_id] = []
            
            # Find periods where values exceed threshold
            critical_indices = np.where(np.array(time_series) > threshold)[0]
            
            # Group consecutive periods
            if len(critical_indices) > 0:
                current_period = {"start": critical_indices[0], "end": critical_indices[0]}
                
                for idx in critical_indices[1:]:
                    if idx == current_period["end"] + 1:
                        current_period["end"] = idx
                    else:
                        critical_periods[entity_name][risk_id].append(current_period)
                        current_period = {"start": idx, "end": idx}
                
                critical_periods[entity_name][risk_id].append(current_period)
    
    return critical_periods

def forecast_cumulative_impact(time_series_results: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate cumulative impact forecasts for each entity.
    
    Args:
        time_series_results: Dictionary of time series data by entity and risk
        
    Returns:
        Dictionary of cumulative impact metrics by entity
    """
    cumulative_impacts = {}
    for entity_name, entity_risks in time_series_results.items():
        # Calculate total impact over time
        total_impact = np.zeros(len(next(iter(entity_risks.values()))))
        for time_series in entity_risks.values():
            total_impact += np.array(time_series)
        
        # Calculate metrics
        cumulative_impacts[entity_name] = {
            "total": float(np.sum(total_impact)),
            "mean": float(np.mean(total_impact)),
            "max": float(np.max(total_impact)),
            "final": float(total_impact[-1])
        }
    
    return cumulative_impacts

def calculate_technology_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for technology transition risks."""
    try:
        base_impact = risk.impact
        # Technology risks are sensitive to renewable energy adoption and market volatility
        renewable_factor = (1 - data.renewable_energy_share/100) * 0.25  # inverse as higher adoption means lower risk
        market_volatility_factor = (data.market_volatility / 25) * 0.15  # normalized to VIX baseline
        gdp_factor = abs(data.gdp_growth - 2) * 0.1
        
        impact = base_impact * (1 + renewable_factor + market_volatility_factor + gdp_factor)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in technology impact calculation: {str(e)}")
        return risk.impact

def calculate_market_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for market transition risks."""
    try:
        base_impact = risk.impact
        # Market risks are sensitive to multiple economic indicators
        carbon_price_factor = (data.carbon_price / 100) * 0.2  # normalized to $100/tonne
        market_volatility_factor = (data.market_volatility / 25) * 0.2
        supply_chain_factor = data.supply_chain_disruption_index * 0.15
        gdp_factor = abs(data.gdp_growth - 2) * 0.15
        
        impact = base_impact * (1 + carbon_price_factor + market_volatility_factor + 
                              supply_chain_factor + gdp_factor)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in market impact calculation: {str(e)}")
        return risk.impact

def calculate_reputation_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for reputation transition risks."""
    try:
        base_impact = risk.impact
        # Reputation risks are sensitive to policy stringency and environmental factors
        policy_factor = data.policy_stringency_index * 0.2
        ecosystem_factor = (1 - data.ecosystem_health_index) * 0.15
        biodiversity_factor = (1 - data.biodiversity_index) * 0.15
        
        impact = base_impact * (1 + policy_factor + ecosystem_factor + biodiversity_factor)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in reputation impact calculation: {str(e)}")
        return risk.impact

def calculate_nature_related_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for nature-related risks."""
    try:
        base_impact = risk.impact
        # Nature risks are sensitive to biodiversity loss and ecosystem health
        biodiversity_factor = (1 - data.biodiversity_index) * 0.3
        ecosystem_factor = (1 - data.ecosystem_health_index) * 0.2
        deforestation_factor = data.deforestation_rate * 0.2
        temp_anomaly_factor = (data.global_temp_anomaly / 1.5) * 0.1
        
        impact = base_impact * (1 + biodiversity_factor + ecosystem_factor + 
                              deforestation_factor + temp_anomaly_factor)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in nature-related impact calculation: {str(e)}")
        return risk.impact

def calculate_systemic_impact(risk: Risk, data: ExternalData) -> float:
    """Calculate impact for systemic risks."""
    try:
        base_impact = risk.impact
        # Systemic risks are sensitive to multiple interconnected factors
        market_volatility_factor = (data.market_volatility / 25) * 0.2
        supply_chain_factor = data.supply_chain_disruption_index * 0.2
        gdp_volatility = abs(data.gdp_growth - 2) * 0.15
        policy_factor = data.policy_stringency_index * 0.15
        ecosystem_factor = (1 - data.ecosystem_health_index) * 0.1
        
        impact = base_impact * (1 + market_volatility_factor + supply_chain_factor + 
                              gdp_volatility + policy_factor + ecosystem_factor)
        return min(1.0, max(0.0, impact))
    except Exception as e:
        logger.error(f"Error in systemic impact calculation: {str(e)}")
        return risk.impact

def project_nature_risk(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Project nature-related risks using combined ecological indicators and threshold effects.
    Incorporates biodiversity loss, ecosystem health, and tipping points.
    """
    try:
        historical_impacts = calculate_nature_related_impact(risk, external_data)
        
        # Extract ecological indicators
        biodiversity_trend = np.array([data.biodiversity_index for data in external_data.values()])
        ecosystem_trend = np.array([data.ecosystem_health_index for data in external_data.values()])
        deforestation_trend = np.array([data.deforestation_rate for data in external_data.values()])
        
        # Detect ecological thresholds
        thresholds = detect_ecological_thresholds(
            biodiversity_trend,
            ecosystem_trend,
            deforestation_trend
        )
        
        # Fit base model (using gamma distribution for positive skew)
        alpha, loc, beta = gamma.fit(historical_impacts)
        base_projections = gamma.rvs(
            alpha, loc=loc, scale=beta, 
            size=TIME_SERIES_HORIZON
        )
        
        # Apply ecological trends and thresholds
        adjusted_projections = []
        for i, proj in enumerate(base_projections):
            # Calculate trend factors
            bio_factor = calculate_biodiversity_factor(biodiversity_trend, i)
            eco_factor = calculate_ecosystem_factor(ecosystem_trend, i)
            defor_factor = calculate_deforestation_factor(deforestation_trend, i)
            
            # Check for threshold crossings
            threshold_factor = calculate_threshold_impact(i, thresholds)
            
            # Combine factors with threshold effects
            combined_factor = (bio_factor * 0.4 + 
                             eco_factor * 0.3 + 
                             defor_factor * 0.3) * threshold_factor
            
            adjusted_proj = proj * combined_factor
            adjusted_projections.append(adjusted_proj)
        
        return validate_and_normalize_forecast(adjusted_projections, risk)
        
    except Exception as e:
        logger.error(f"Error in nature risk projection: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

def project_systemic_risk(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    """
    Project systemic risks using network effects and contagion modeling.
    Incorporates interconnected risk factors and cascade effects.
    """
    try:
        historical_impacts = calculate_systemic_impact(risk, external_data)
        
        # Extract systemic indicators
        market_vol = np.array([data.market_volatility for data in external_data.values()])
        supply_chain = np.array([data.supply_chain_disruption_index for data in external_data.values()])
        policy_stringency = np.array([data.policy_stringency_index for data in external_data.values()])
        
        # Detect systemic stress points
        stress_points = detect_systemic_stress(
            market_vol,
            supply_chain,
            policy_stringency
        )
        
        # Fit base model using ARIMA with higher orders for complex dynamics
        model = ARIMA(historical_impacts, order=(2, 2, 2))
        model_fit = model.fit()
        base_forecast = model_fit.forecast(steps=TIME_SERIES_HORIZON)
        
        # Apply systemic factors and contagion effects
        adjusted_projections = []
        for i, proj in enumerate(base_forecast):
            # Calculate interconnected factors
            market_factor = calculate_market_stress_factor(market_vol, i)
            supply_factor = calculate_supply_chain_factor(supply_chain, i)
            policy_factor = calculate_policy_factor(policy_stringency, i)
            
            # Calculate contagion multiplier
            contagion_factor = calculate_contagion_effect(i, stress_points)
            
            # Combine with network effects
            network_factor = calculate_network_amplification(
                market_factor,
                supply_factor,
                policy_factor
            )
            
            # Apply combined factors
            adjusted_proj = proj * network_factor * contagion_factor
            adjusted_projections.append(adjusted_proj)
        
        return validate_and_normalize_forecast(adjusted_projections, risk)
        
    except Exception as e:
        logger.error(f"Error in systemic risk projection: {str(e)}")
        return [risk.impact] * TIME_SERIES_HORIZON

# Helper functions for nature risk projections
def detect_ecological_thresholds(biodiversity: np.ndarray, ecosystem: np.ndarray, 
                               deforestation: np.ndarray) -> List[Tuple[int, float]]:
    """Detect potential ecological tipping points."""
    thresholds = []
    
    # Check for rapid biodiversity loss
    bio_changes = np.diff(biodiversity)
    eco_changes = np.diff(ecosystem)
    defor_changes = np.diff(deforestation)
    
    for i in range(len(bio_changes)):
        # Detect compound effects
        if (bio_changes[i] < -0.05 and  # Significant biodiversity loss
            eco_changes[i] < -0.05 and  # Ecosystem degradation
            defor_changes[i] > 0.05):   # Increased deforestation
            thresholds.append((i, 1.2))  # 20% impact increase
            
    return thresholds

def calculate_biodiversity_factor(trend: np.ndarray, index: int) -> float:
    """Calculate biodiversity impact factor with acceleration."""
    if index < len(trend):
        return 1 + (1 - trend[index]) * 0.3
    return 1.3  # Assume continuing degradation

def calculate_ecosystem_factor(trend: np.ndarray, index: int) -> float:
    """Calculate ecosystem health factor."""
    if index < len(trend):
        return 1 + (1 - trend[index]) * 0.25
    return 1.25

def calculate_deforestation_factor(trend: np.ndarray, index: int) -> float:
    """Calculate deforestation impact factor."""
    if index < len(trend):
        return 1 + trend[index] * 0.2
    return 1.2

# Helper functions for systemic risk projections
def detect_systemic_stress(market_vol: np.ndarray, supply_chain: np.ndarray,
                          policy: np.ndarray) -> List[Tuple[int, float]]:
    """Detect periods of systemic stress."""
    stress_points = []
    
    for i in range(1, len(market_vol)):
        # Detect compound stress conditions
        if (market_vol[i] > np.mean(market_vol) + 2*np.std(market_vol) and
            supply_chain[i] > np.mean(supply_chain) + np.std(supply_chain)):
            stress_points.append((i, 1.3))  # 30% impact increase
            
    return stress_points

def calculate_market_stress_factor(trend: np.ndarray, index: int) -> float:
    """Calculate market stress factor."""
    if index < len(trend):
        return 1 + (trend[index] / 50) * 0.3  # Normalized to VIX scale
    return 1.3

def calculate_supply_chain_factor(trend: np.ndarray, index: int) -> float:
    """Calculate supply chain disruption factor."""
    if index < len(trend):
        return 1 + trend[index] * 0.25
    return 1.25

def calculate_policy_factor(trend: np.ndarray, index: int) -> float:
    """Calculate policy impact factor."""
    if index < len(trend):
        return 1 + (1 - trend[index]) * 0.2  # Inverse as higher policy stringency reduces risk
    return 1.2

def calculate_network_amplification(*factors: float) -> float:
    """Calculate network effect amplification."""
    # Use geometric mean for multiplicative effects
    return np.exp(np.mean(np.log(factors)))

def calculate_contagion_effect(index: int, stress_points: List[Tuple[int, float]]) -> float:
    """Calculate contagion effect based on stress points."""
    if not stress_points:
        return 1.0
    
    # Calculate decay from each stress point
    effects = []
    for stress_idx, magnitude in stress_points:
        time_since_stress = index - stress_idx
        if time_since_stress >= 0:
            decay = magnitude * np.exp(-0.2 * time_since_stress)
            effects.append(decay)
    
    return 1.0 + np.sum(effects) if effects else 1.0

def calculate_threshold_impact(index: int, thresholds: List[Tuple[int, float]]) -> float:
    """
    Calculate impact multiplier based on ecological thresholds.
    
    Args:
        index: Current time index
        thresholds: List of (time_index, magnitude) tuples representing detected thresholds
    
    Returns:
        float: Impact multiplier (>= 1.0)
    """
    if not thresholds:
        return 1.0
    
    # Calculate cumulative effect of all thresholds
    threshold_effects = []
    for threshold_idx, magnitude in thresholds:
        # Calculate time distance from threshold
        time_since_threshold = index - threshold_idx
        
        if time_since_threshold >= 0:
            # Apply non-linear threshold effect with memory
            # Effect increases sharply after threshold and decays slowly
            immediate_effect = magnitude - 1.0  # Convert magnitude to excess impact
            decay_rate = 0.1  # Slower decay for persistent threshold effects
            memory_factor = np.exp(-decay_rate * time_since_threshold)
            
            # Add non-linear amplification for compound effects
            if len(threshold_effects) > 0:
                amplification = 1.0 + (len(threshold_effects) * 0.1)  # 10% extra per existing threshold
            else:
                amplification = 1.0
            
            threshold_effect = 1.0 + (immediate_effect * memory_factor * amplification)
            threshold_effects.append(threshold_effect)
    
    if not threshold_effects:
        return 1.0
    
    # Combine threshold effects (multiplicative)
    # Using geometric mean to prevent extreme amplification
    combined_effect = np.exp(np.mean(np.log(threshold_effects)))
    
    # Add extra impact for multiple simultaneous thresholds
    if len(threshold_effects) > 1:
        synergy_factor = 1.0 + (len(threshold_effects) - 1) * 0.05  # 5% extra per additional threshold
        combined_effect *= synergy_factor
    
    # Ensure reasonable bounds
    max_threshold_effect = 2.0  # Maximum doubling of impact
    return min(max_threshold_effect, max(1.0, combined_effect))

