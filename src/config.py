import os
import logging
from datetime import datetime
from typing import Dict
from src.models import Company, Entity, Scenario
from dotenv import load_dotenv

load_dotenv()

# Scenario definitions
SCENARIOS: Dict[str, Scenario] = {
    "Net Zero 2050": Scenario(
        name="Net Zero 2050",
        description="A scenario where net-zero emissions are achieved by 2050",
        time_horizon="2050",
        temp_increase=1.5,
        carbon_price=250,
        renewable_energy=0.75,
        policy_stringency=0.9,
        biodiversity_loss=0.1,
        ecosystem_degradation=0.2,
        financial_stability=0.8,
        supply_chain_disruption=0.3
    ),
    "Delayed Transition": Scenario(
        name="Delayed Transition",
        description="A scenario with delayed climate action",
        time_horizon="2050",
        temp_increase=2.5,
        carbon_price=125,
        renewable_energy=0.55,
        policy_stringency=0.6,
        biodiversity_loss=0.3,
        ecosystem_degradation=0.4,
        financial_stability=0.6,
        supply_chain_disruption=0.5
    ),
    # "Current Policies": Scenario(
    #     name="Current Policies",
    #     description="A scenario with current climate policies",
    #     time_horizon="2050",
    #     temp_increase=3.5,
    #     carbon_price=35,
    #     renewable_energy=0.35,
    #     policy_stringency=0.2,
    #     biodiversity_loss=0.5,
    #     ecosystem_degradation=0.6,
    #     financial_stability=0.4,
    #     supply_chain_disruption=0.7
    # ),
    # "Nature Positive": Scenario(
    #     name="Nature Positive",
    #     description="A scenario with net positive impacts on nature",
    #     time_horizon="2050",
    #     temp_increase=1.8,
    #     carbon_price=200,
    #     renewable_energy=0.7,
    #     policy_stringency=0.8,
    #     biodiversity_loss=-0.1,  # Net gain
    #     ecosystem_degradation=-0.2,  # Net restoration
    #     financial_stability=0.75,
    #     supply_chain_disruption=0.4
    # ),
    # "Systemic Crisis": Scenario(
    #     name="Systemic Crisis",
    #     description="A scenario with systemic climate crisis",
    #     time_horizon="2050",
    #     temp_increase=4.0,
    #     carbon_price=50,
    #     renewable_energy=0.4,
    #     policy_stringency=0.3,
    #     biodiversity_loss=0.6,
    #     ecosystem_degradation=0.7,
    #     financial_stability=0.2,
    #     supply_chain_disruption=0.8
    # ),
    # "Cascading Failures": Scenario(
    #     name="Cascading Failures",
    #     description="A scenario with cascading climate failures",
    #     time_horizon="2050",
    #     temp_increase=3.5,
    #     carbon_price=75,
    #     renewable_energy=0.45,
    #     policy_stringency=0.4,
    #     biodiversity_loss=0.5,
    #     ecosystem_degradation=0.6,
    #     financial_stability=0.3,
    #     supply_chain_disruption=0.8
    # ),
    # "Global Instability": Scenario(
    #     name="Global Instability",
    #     description="A scenario with global climate instability",
    #     time_horizon="2050",
    #     temp_increase=4.0,
    #     carbon_price=50,
    #     renewable_energy=0.4,
    #     policy_stringency=0.3,
    #     biodiversity_loss=0.6,
    #     ecosystem_degradation=0.7,
    #     financial_stability=0.2,
    #     supply_chain_disruption=0.8
    # )
}

# Monte Carlo simulation parameters
NUM_SIMULATIONS = 10000

# Clustering parameters
NUM_CLUSTERS = 3

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# LLM configuration
LLM_MODEL = "gpt-4o-mini"  # Replace with the actual model you're using
LLM_API_KEY = os.getenv("OPENAI_API_KEY")

if not LLM_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Visualization settings
VIZ_DPI = 300
HEATMAP_CMAP = 'YlOrRd'

# Time series configuration
TIME_SERIES_HORIZON = 50  # Number of time periods to forecast

# Sensitivity analysis parameters
SENSITIVITY_VARIABLES = ['temp_increase', 'carbon_price', 'renewable_energy', 'policy_stringency', 'biodiversity_loss', 'ecosystem_degradation', 'financial_stability', 'supply_chain_disruption']
SENSITIVITY_RANGE = 0.2  # +/- 20%

def setup_logging(log_level: str = "INFO") -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Generate a unique log file name with datetime stamp
    log_file_name = f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(LOGS_DIR, log_file_name)
    
    # Set up logging configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Set OpenAI logger to WARNING to reduce noise
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized. Log file: {log_file_path}")
    logging.debug("Debug logging enabled")

    # Set the root logger to DEBUG to capture all log levels
    logging.getLogger().setLevel(logging.DEBUG)

COMPANY_INFO = Company(
    name="Apple Inc.",
    industry="Technology",
    region=["North America", "Europe", "Asia"],
    key_products=["Consumer Electronics", "Software", "Cloud Services", "Wearables"],
    key_dependencies=[
        "Semiconductor supply chain",
        "Rare earth materials",
        "Global logistics network",
        "App Store ecosystem",
        "Cloud infrastructure"
    ],
    entities={
        "iPhone Division": Entity(
            name="iPhone Division",
            description="Responsible for iPhone development and production",
            key_products=["iPhone"],
            region=["Global"],
            parent_entity=None,
            weight=0.6  # Represents 50% of company's risk weight
        ),
        "Mac Division": Entity(
            name="Mac Division",
            description="Responsible for Mac computers development and production",
            key_products=["MacBook", "iMac", "Mac Pro"],
            region=["Global"],
            parent_entity=None,
            weight=0.4  # Represents 30% of company's risk weight
        )
    }
)

# Update sub-entities with weights
COMPANY_INFO.add_entity(Entity(
    name="iPhone Hardware",
    description="Responsible for iPhone hardware development",
    key_products=["iPhone Hardware"],
    region=["Global"],
    parent_entity="iPhone Division",
    weight=0.8  # 30% of iPhone Division's weight
))

COMPANY_INFO.add_entity(Entity(
    name="iPhone Software",
    description="Responsible for iPhone software development",
    key_products=["iOS"],
    region=["Global"],
    parent_entity="iPhone Division",
    weight=0.2  # 20% of iPhone Division's weight
))

COMPANY_INFO.add_entity(Entity(
    name="Mac Hardware",
    description="Responsible for Mac hardware development",
    key_products=["Mac Hardware"],
    region=["Global"],
    parent_entity="Mac Division",
    weight=0.9  # 20% of Mac Division's weight
))

COMPANY_INFO.add_entity(Entity(
    name="Mac Software",
    description="Responsible for Mac software development",
    key_products=["macOS"],
    region=["Global"],
    parent_entity="Mac Division",
    weight=0.1  # 10% of Mac Division's weight
))
