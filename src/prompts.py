# Climate Risk Assessment Prompts

from src.config import COMPANY_INFO

RISK_ASSESSMENT_PROMPT = """
<instruction>
As an expert in climate risk assessment, evaluate the following risk for {company_name}, specifically for the {entity_name} entity:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<entity_context>
Entity: {entity_name}
Description: {entity_description}
Key Products: {entity_key_products}
Region: {entity_region}
</entity_context>

<risk>
<id>{risk_id}</id>
<description>{risk_description}</description>
<category>{risk_category}</category>
<subcategory>{risk_subcategory}</subcategory>
<current_likelihood>{risk_likelihood}</current_likelihood>
<current_impact>{risk_impact}</current_impact>
<time_horizon>{risk_time_horizon}</time_horizon>
</risk>

<scenario>
<name>{scenario_name}</name>
<temperature_increase>{temp_increase}</temperature_increase>
<carbon_price>{carbon_price}</carbon_price>
<renewable_energy>{renewable_energy}</renewable_energy>
<policy_stringency>{policy_stringency}</policy_stringency>
<biodiversity_loss>{biodiversity_loss}</biodiversity_loss>
<ecosystem_degradation>{ecosystem_degradation}</ecosystem_degradation>
<financial_stability>{financial_stability}</financial_stability>
<supply_chain_disruption>{supply_chain_disruption}</supply_chain_disruption>
</scenario>

<task>
Provide a detailed analysis addressing the following points:
1. How does this risk's likelihood and impact change under the given scenario, considering the specific context of the {entity_name} entity?
2. What are the potential financial implications for the {entity_name} entity over the next 5 years?
3. Are there any emerging opportunities related to this risk in this scenario, particularly for the {entity_name} entity's products or services?
4. What additional challenges might arise from this risk in this specific context, considering the {entity_name} entity's supply chain and operations?
5. Suggest 2-3 possible mitigation strategies tailored to this scenario for the {entity_name} entity

Please structure your response in JSON format as follows:
</task>

<output_format>
{{
  "risk_id": "string",
  "entity_name": "string",
  "likelihood_impact_change": {{
    "likelihood_change": "string (choose one: ['Significant Decrease', 'Moderate Decrease', 'Slight Decrease', 'No Change', 'Slight Increase', 'Moderate Increase', 'Significant Increase'])",
    "impact_change": "string (choose one: ['Significant Decrease', 'Moderate Decrease', 'Slight Decrease', 'No Change', 'Slight Increase', 'Moderate Increase', 'Significant Increase'])",
    "explanation": "string"
  }},
  "financial_implications": {{
    "short_term": "string (choose one: ['Highly Negative', 'Moderately Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Moderately Positive', 'Highly Positive'])",
    "medium_term": "string (choose one: ['Highly Negative', 'Moderately Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Moderately Positive', 'Highly Positive'])",
    "long_term": "string (choose one: ['Highly Negative', 'Moderately Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Moderately Positive', 'Highly Positive'])"
  }},
  "emerging_opportunities": [
    {{
      "opportunity": "string",
      "description": "string",
      "potential": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "additional_challenges": [
    {{
      "challenge": "string",
      "description": "string",
      "severity": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "mitigation_strategies": [
    {{
      "strategy": "string",
      "description": "string",
      "potential_impact": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])",
      "feasibility": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ]
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

RISK_NARRATIVE_PROMPT = """
<instruction>
As an expert in climate risk assessment, create a concise narrative summary for the following risk across different scenarios, specifically for {company_name}. Focus on key trends, variations in impact and likelihood, and overarching mitigation strategies.
</instruction>

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk>
<id>{risk_id}</id>
<description>{risk_description}</description>
<category>{risk_category}</category>
<subcategory>{risk_subcategory}</subcategory>
<current_likelihood>{risk_likelihood}</current_likelihood>
<current_impact>{risk_impact}</current_impact>
<time_horizon>{risk_time_horizon}</time_horizon>
</risk>

<scenario_analyses>
{scenario_analyses}
</scenario_analyses>

<task>
Provide a concise narrative summary (about 200 words) that synthesizes the insights from all scenarios, highlighting key trends and strategic implications for {company_name}. Structure your response in JSON format as follows:
</task>

<output_format>
{{
  "risk_id": "string",
  "narrative_summary": "string",
  "key_trends": [
    {{
      "trend": "string",
      "description": "string",
      "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "strategic_implications": [
    {{
      "implication": "string",
      "description": "string",
      "impact": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "overarching_mitigation_strategies": [
    {{
      "strategy": "string",
      "description": "string",
      "effectiveness": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])",
      "feasibility": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ]
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

EXECUTIVE_INSIGHTS_PROMPT = """
<instruction>
As a senior climate risk analyst, review the following comprehensive risk analyses across multiple scenarios for {company_name}, focusing on the {entity_name} entity. Provide high-level executive insights.
</instruction>

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<entity_context>
Entity: {entity_name}
Description: {entity_description}
Key Products: {entity_key_products}
Region: {entity_region}
</entity_context>

<analyses>
{all_analyses}
</analyses>

<task>
1. Key overarching trends across scenarios specific to {entity_name}'s context within {company_name}
2. Most critical risks requiring immediate attention for {entity_name}
3. Potential strategic opportunities arising from climate change for {entity_name}'s products/services
4. Recommendations for enhancing {entity_name}'s overall climate resilience

Provide a concise executive summary (about 400 words) with key insights and strategic recommendations tailored to {entity_name} within {company_name}. Structure your response in JSON format as follows:
</task>

<output_format>
{{
  "executive_summary": "string",
  "key_trends": [
    {{
      "trend": "string",
      "description": "string",
      "implications": "string",
      "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "critical_risks": [
    {{
      "risk": "string",
      "urgency": "string (choose one: ['Low', 'Medium', 'High', 'Very High', 'Critical'])",
      "potential_impact": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "strategic_opportunities": [
    {{
      "opportunity": "string",
      "description": "string",
      "potential_benefits": "string",
      "feasibility": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "resilience_recommendations": [
    {{
      "recommendation": "string",
      "description": "string",
      "expected_outcome": "string",
      "priority": "string (choose one: ['Low', 'Medium', 'High', 'Very High', 'Critical'])"
    }}
  ]
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

INTERACTION_ANALYSIS_PROMPT = """
<instruction>
As an expert in climate risk assessment, analyze the potential interaction between two risks for {company_name}:

<company_context>
Company: {company_name}
Industry: {company_industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk_1>
ID: {risk1_id}
Description: {risk1_description}
Category: {risk1_category}
Subcategory: {risk1_subcategory}
</risk_1>

<risk_2>
ID: {risk2_id}
Description: {risk2_description}
Category: {risk2_category}
Subcategory: {risk2_subcategory}
</risk_2>

Analyze how these risks might interact, considering:
1. The company's industry context
2. The company's operations and dependencies
3. Potential compounding effects
4. Possible mitigating factors

Provide your analysis in the following format:
</instruction>

<output_format>
{{
  "interaction_score": {{
    "score": float,  # 0-1 where 0 is no interaction and 1 is strongest possible interaction
    "confidence": float  # 0-1 indicating confidence in the assessment
  }},
  "interaction_explanation": "string",
  "compounding_effects": [
    {{
      "effect": "string",
      "severity": "string"
    }}
  ],
  "mitigating_factors": [
    {{
      "factor": "string",
      "effectiveness": "string"
    }}
  ],
  "recommendations": [
    {{
      "action": "string",
      "priority": "string",
      "scope": "string"
    }}
  ]
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""


SYSTEMIC_RISK_PROMPT = """
<instruction>
As an expert in systemic risk analysis, evaluate the following risk in the context of broader systems, specifically for {company_name}:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk>
<id>{risk_id}</id>
<description>{risk_description}</description>
<category>{risk_category}</category>
<subcategory>{risk_subcategory}</subcategory>
</risk>

<context>
<company_industry>{company_industry}</company_industry>
<key_dependencies>{key_dependencies}</key_dependencies>
</context>

Please provide an analysis addressing the following points:
1. How might this risk contribute to or be affected by systemic vulnerabilities in the {company_industry} sector, particularly for {company_name}?
2. Identify potential trigger points or critical thresholds related to this risk that could lead to cascading effects across systems, considering {company_name}'s position in the industry.
3. Assess {company_name}'s potential role in mitigating this systemic risk, considering its position in the industry and sustainability goals.
4. Suggest collaborative initiatives or policy engagements that could help address this risk at a systemic level, with {company_name} playing a key role.

Structure your response in JSON format as follows:
</instruction>

<output_format>
{{
  "risk_id": "string",
  "systemic_vulnerabilities": {{
    "contribution": "string",
    "affected_by": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "trigger_points": [
    {{
      "description": "string",
      "threshold": "string",
      "potential_cascading_effects": "string",
      "likelihood": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "company_mitigation_role": {{
    "potential_actions": "string",
    "industry_position_leverage": "string",
    "effectiveness": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "collaborative_initiatives": [
    {{
      "initiative": "string",
      "description": "string",
      "potential_impact": "string",
      "feasibility": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "policy_engagements": [
    {{
      "engagement": "string",
      "description": "string",
      "expected_outcome": "string",
      "importance": "string (choose one: ['Low', 'Medium', 'High', 'Very High', 'Critical'])"
    }}
  ]
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

MITIGATION_STRATEGY_PROMPT = """
<instruction>
As an expert in climate risk mitigation and adaptation strategies, analyze the following risk and its assessment across different scenarios for {company_name}:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk>
<id>{risk_id}</id>
<description>{risk_description}</description>
<category>{risk_category}</category>
<subcategory>{risk_subcategory}</subcategory>
</risk>

<scenario_analyses>
{scenario_analyses}
</scenario_analyses>

Based on this information, please provide:

1. 3-5 concrete mitigation strategies that address this risk across multiple scenarios, tailored to {company_name}'s context and sustainability goals.
2. For each strategy, briefly explain its potential effectiveness and any challenges in implementation specific to {company_name}.
3. Prioritize these strategies based on their potential impact and feasibility for {company_name}.

Structure your response in JSON format as follows:
</instruction>

<output_format>
{{
  "risk_id": "{risk_id}",
  "mitigation_strategies": [
    {{
      "strategy": "string",
      "description": "string",
      "potential_effectiveness": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])",
      "implementation_challenges": "string",
      "challenge_level": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])",
      "priority": int,
      "impact": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])",
      "feasibility": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
    }}
  ],
  "overall_recommendation": "string"
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

# Add the new PESTEL_ANALYSIS_PROMPT
PESTEL_ANALYSIS_PROMPT = """
<instruction>
As an expert in PESTEL (Political, Economic, Social, Technological, Environmental, Legal) analysis, evaluate the following risk for {company_name} considering each PESTEL factor:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk>
<id>{risk_id}</id>
{risk_description}
</risk>

Please provide an analysis addressing how this risk relates to each PESTEL factor, considering {company_name}'s specific context. For each factor, provide:
1. A brief explanation of how the risk relates to that factor.
2. Potential impacts on {company_name}.
3. Possible opportunities or challenges arising from this factor.

Structure your response in JSON format as follows:
</instruction>

<output_format>
{{
  "risk_id": "string",
  "political": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "economic": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "social": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "technological": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "environmental": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "legal": {{
    "explanation": "string",
    "potential_impacts": "string",
    "opportunities_challenges": "string",
    "significance": "string (choose one: ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])"
  }},
  "overall_assessment": "string"
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

RISK_INTERACTION_SUMMARY_PROMPT = """
<instruction>
As an expert in climate risk assessment and network analysis, summarize the key findings from the risk interaction analysis for {company_name}:

<analysis_data>
1. Top 5 strongest interactions:
{top_interactions}

2. Top 3 central risks:
{central_risks}

3. Risk clusters:
{risk_clusters}
</analysis_data>

Provide a concise summary addressing the following points:
1. The most critical risk interactions and their potential implications
2. The role of central risks in the overall risk landscape
3. Insights from the risk clustering and what it reveals about the company's risk profile
4. Recommendations for risk management based on these findings

Structure your response in JSON format as follows:
</instruction>

<output_format>
{{
  "critical_interactions": [
    {{
      "interaction": "string",
      "implications": "string",
      "severity": "string (choose one: ['Low', 'Medium', 'High', 'Very High', 'Critical'])"
    }}
  ],
  "central_risks_analysis": {{
    "summary": "string",
    "impact_on_risk_landscape": "string"
  }},
  "risk_clustering_insights": {{
    "key_findings": "string",
    "implications_for_company": "string"
  }},
  "risk_management_recommendations": [
    {{
      "recommendation": "string",
      "rationale": "string",
      "priority": "string (choose one: ['Low', 'Medium', 'High', 'Very High', 'Critical'])"
    }}
  ],
  "overall_summary": "string"
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""

# Add these new prompts at the end of the file

RISK_CATEGORIZATION_PROMPT = """
<instruction>
As an expert in climate risk assessment, categorize the following risk according to the provided taxonomy, considering the specific context of {company_name}:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<risk>
<id>{risk_id}</id>
{risk_description}
</risk>

<taxonomy>
1. Climate-Related Risks
   1.1 Physical Risks
       1.1.1 Acute Physical Risks
       1.1.2 Chronic Physical Risks
   1.2 Transition Risks
       1.2.1 Policy and Legal Risks
       1.2.2 Technology Risks
       1.2.3 Market Risks
       1.2.4 Reputation Risks
2. Nature-related risks
   2.1 Biodiversity Loss
   2.2 Ecosystem Degradation
   2.3 Natural Resource Depletion
   2.4 Nature-Based Carbon Sinks Loss
   2.5 Invasive Species and Pests
3. Systemic Risks
   3.1 Financial System Risks
   3.2 Supply Chain Disruptions
</taxonomy>

Provide the main category, subcategory, and tertiary category (if applicable) for this risk, considering how it specifically relates to {company_name}'s industry and operations. Use the following format:
</instruction>

<output_format>
{{
  "risk_id": "{risk_id}",
  "category": "string",
  "subcategory": "string",
  "tertiary_category": "string"
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted.
</output_format>
"""


RISK_BASELINE_ASSESSMENT_PROMPT = """
<instruction>
As an expert in climate risk assessment, evaluate the likelihood and impact of the following risk for {company_name}, specifically for the {entity_name} entity:

<company_context>
Company: {company_name}
Industry: {industry}
Region: {company_region}
Key Products: {key_products}
</company_context>

<entity_context>
Entity: {entity_name}
Description: {entity_description}
Key Products: {entity_key_products}
Region: {entity_region}
</entity_context>

<risk>
<id>{risk_id}</id>
{risk_description}
</risk>

Considering the specific context of the {entity_name} entity within {company_name}, provide the likelihood and impact scores on a scale of 0 to 1, where 0 is the lowest and 1 is the highest. Use the following Likert-style references:

Likelihood Scale:
0.0 - 0.2: Very Unlikely (Rare event, less than 5% chance of occurrence)
0.2 - 0.4: Unlikely (Occasional event, 5-25% chance of occurrence)
0.4 - 0.6: Possible (Could occur, 25-50% chance of occurrence)
0.6 - 0.8: Likely (Probable event, 50-75% chance of occurrence)
0.8 - 1.0: Very Likely (Frequent event, more than 75% chance of occurrence)

Impact Scale:
0.0 - 0.2: Negligible (Minimal effect on the entity's operations, finances, or reputation)
0.2 - 0.4: Minor (Limited effect, easily managed within normal entity operations)
0.4 - 0.6: Moderate (Noticeable effect, requires some changes to entity operations)
0.6 - 0.8: Major (Significant effect, requires substantial changes to entity operations)
0.8 - 1.0: Severe (Critical effect, threatens the entity's existence or major operations)

Provide your assessment using the following format:
</instruction>

<output_format>
{{
  "risk_id": "{risk_id}",
  "entity_name": "{entity_name}",
  "likelihood": float,
  "impact": float,
  "explanation": "string"
}}

Important: Return the JSON object only, without any backticks or "json" prefix. Make sure to include all the required keys and values. Ensure that the JSON object is properly closed. Double check that the JSON object is correctly formatted. Provide a brief explanation for your assessment, focusing on how this risk specifically affects the {entity_name} entity.
</output_format>
"""



