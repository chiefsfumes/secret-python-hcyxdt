from typing import List, Dict
from src.models import Risk, ExternalData, PESTELAnalysis, Entity, Company
from src.config import LLM_MODEL, LLM_API_KEY, COMPANY_INFO
from src.prompts import (
    RISK_CATEGORIZATION_PROMPT, 
    RISK_BASELINE_ASSESSMENT_PROMPT, 
    PESTEL_ANALYSIS_PROMPT
)
from src.utils.llm_util import get_llm_response
import logging
import json
import os
from datetime import datetime

def categorize_risks(risks: List[Risk], entity: Entity) -> Dict[str, Dict[str, List[Risk]]]:
    logger = logging.getLogger(__name__)
    categorized_risks = {}
    
    # Create checkpoint directory for the company - remove any trailing periods
    company_name = COMPANY_INFO.name.lower().replace(' ', '_').rstrip('.')
    checkpoint_dir = os.path.join('checkpoints', company_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for risk in risks:
        # Create checkpoint filename using just the risk ID
        checkpoint_file = os.path.join(
            checkpoint_dir, 
            f'risk_{risk.id}_categorization.json'
        )
        
        # Check if we have a checkpoint
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    categorization_data = json.load(f)
                    logger.info(f"Loaded categorization from checkpoint for risk {risk.id}")
            except Exception as e:
                logger.error(f"Error loading checkpoint for risk {risk.id}: {e}")
                categorization_data = None
        else:
            categorization_data = None
        
        # If no checkpoint or loading failed, query the LLM
        if not categorization_data:
            prompt = RISK_CATEGORIZATION_PROMPT.format(
                risk_id=risk.id,
                risk_description=risk.description,
                company_name=COMPANY_INFO.name,
                industry=COMPANY_INFO.industry,
                company_region=", ".join(COMPANY_INFO.region),
                key_products=", ".join(COMPANY_INFO.key_products),
                entity_name=entity.name,
                entity_description=entity.description,
                entity_key_products=", ".join(entity.key_products),
                entity_region=", ".join(entity.region)
            )
            logger.info(f"Sending prompt to LLM for risk categorization:\n{json.dumps({'prompt': prompt}, indent=2)}")

            system_message = "You are an expert in climate risk assessment. Always respond with valid JSON."
            categorization_data = get_llm_response(prompt, system_message)
            logger.info(f"Received categorization data from LLM:\n{json.dumps(categorization_data, indent=2)}")
            
            # Save the checkpoint
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(categorization_data, f, indent=2)
                logger.info(f"Saved categorization checkpoint for risk {risk.id}")
            except Exception as e:
                logger.error(f"Error saving checkpoint for risk {risk.id}: {e}")

        # Fallback if response is empty
        if not categorization_data:
            categorization_data = {"category": "Uncategorized", "subcategory": "Unknown", "tertiary_category": "Unknown"}

        category = categorization_data.get('category', "Uncategorized")
        subcategory = categorization_data.get('subcategory', "Unknown")
        tertiary_category = categorization_data.get('tertiary_category', "Unknown")
        
        if category not in categorized_risks:
            categorized_risks[category] = {}
        if subcategory not in categorized_risks[category]:
            categorized_risks[category][subcategory] = []
        
        risk.category = category
        risk.subcategory = subcategory
        risk.tertiary_category = tertiary_category
        categorized_risks[category][subcategory].append(risk)
    
    return categorized_risks

def assess_risks(risks: List[Risk], entity: Entity) -> List[Risk]:
    logger = logging.getLogger(__name__)
    
    # Create checkpoint directory for the company - remove any trailing periods
    company_name = COMPANY_INFO.name.lower().replace(' ', '_').rstrip('.')
    checkpoint_dir = os.path.join('checkpoints', company_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for risk in risks:
        # Create checkpoint filename using risk ID
        checkpoint_file = os.path.join(
            checkpoint_dir, 
            f'risk_{risk.id}_assessment.json'
        )
        
        # Check if we have a checkpoint
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    assessment_data = json.load(f)
                    logger.info(f"Loaded assessment from checkpoint for risk {risk.id}")
            except Exception as e:
                logger.error(f"Error loading checkpoint for risk {risk.id}: {e}")
                assessment_data = None
        else:
            assessment_data = None
            
        # If no checkpoint or loading failed, query the LLM
        if not assessment_data:
            prompt = RISK_BASELINE_ASSESSMENT_PROMPT.format(
                risk_id=risk.id,
                risk_description=risk.description,
                company_name=COMPANY_INFO.name,
                industry=COMPANY_INFO.industry,
                company_region=", ".join(COMPANY_INFO.region),
                key_products=", ".join(COMPANY_INFO.key_products),
                entity_name=entity.name,
                entity_description=entity.description,
                entity_key_products=", ".join(entity.key_products),
                entity_region=", ".join(entity.region)
            )
            logger.info(f"Sending prompt to LLM for risk assessment:\n{json.dumps({'prompt': prompt}, indent=2)}")

            system_message = "You are an expert in climate risk assessment. Always respond with valid JSON."
            assessment_data = get_llm_response(prompt, system_message)
            logger.info(f"Received assessment data from LLM:\n{json.dumps(assessment_data, indent=2)}")
            
            # Save the checkpoint
            if assessment_data:
                try:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(assessment_data, f, indent=2)
                    logger.info(f"Saved assessment checkpoint for risk {risk.id}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint for risk {risk.id}: {e}")

        # Fallback if response is empty
        if not assessment_data:
            assessment_data = {"likelihood": 0.5, "impact": 0.5, "explanation": "Error in parsing response"}

        risk.likelihood = assessment_data.get('likelihood', 0.5)
        risk.impact = assessment_data.get('impact', 0.5)
        risk.assessment_explanation = assessment_data.get('explanation', '')
    
    return risks

def prioritize_risks(risks: List[Risk]) -> Dict[str, List[Risk]]:
    priorities = {"High": [], "Medium": [], "Low": []}
    for risk in risks:
        if risk.impact > 0.7 and risk.likelihood > 0.7:
            priorities["High"].append(risk)
        elif risk.impact > 0.3 and risk.likelihood > 0.3:
            priorities["Medium"].append(risk)
        else:
            priorities["Low"].append(risk)
    return priorities

def perform_pestel_analysis(risks: List[Risk], external_data: Dict[str, ExternalData]) -> Dict[int, PESTELAnalysis]:
    logger = logging.getLogger(__name__)
    pestel_analyses = {}
    
    # Create checkpoint directory for the company - remove any trailing periods
    company_name = COMPANY_INFO.name.lower().replace(' ', '_').rstrip('.')
    checkpoint_dir = os.path.join('checkpoints', company_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for risk in risks:
        # Create checkpoint filename using risk ID
        checkpoint_file = os.path.join(
            checkpoint_dir, 
            f'risk_{risk.id}_pestel.json'
        )
        
        # Check if we have a checkpoint
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    pestel_data = json.load(f)
                    logger.info(f"Loaded PESTEL analysis from checkpoint for risk {risk.id}")
            except Exception as e:
                logger.error(f"Error loading PESTEL checkpoint for risk {risk.id}: {e}")
                pestel_data = None
        else:
            pestel_data = None
            
        # If no checkpoint or loading failed, query the LLM
        if not pestel_data:
            prompt = PESTEL_ANALYSIS_PROMPT.format(
                risk_id=risk.id,
                risk_description=risk.description,
                company_name=COMPANY_INFO.name,
                industry=COMPANY_INFO.industry,
                company_region=", ".join(COMPANY_INFO.region),
                key_products=", ".join(COMPANY_INFO.key_products)
            )
            logger.info(f"Sending prompt to LLM for PESTEL analysis:\n{json.dumps({'prompt': prompt}, indent=2)}")

            system_message = "You are an expert in PESTEL analysis. Always respond with valid JSON."
            pestel_data = get_llm_response(prompt, system_message)
            logger.info(f"Received PESTEL data from LLM:\n{json.dumps(pestel_data, indent=2)}")
            
            # Save the checkpoint
            if pestel_data:
                try:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(pestel_data, f, indent=2)
                    logger.info(f"Saved PESTEL checkpoint for risk {risk.id}")
                except Exception as e:
                    logger.error(f"Error saving PESTEL checkpoint for risk {risk.id}: {e}")
        
        if pestel_data:
            try:
                pestel_analysis = PESTELAnalysis(**pestel_data)
                pestel_analyses[risk.id] = pestel_analysis
                logger.info(f"Created PESTELAnalysis object for risk {risk.id}.")
            except ValueError as e:
                logger.error(f"Error creating PESTELAnalysis object for risk {risk.id}: {e}")
        else:
            logger.error(f"No PESTEL data received for risk {risk.id}.")
    
    return pestel_analyses
