from openai import OpenAI
from src.config import LLM_MODEL, LLM_API_KEY
import json
import logging

client = OpenAI(api_key=LLM_API_KEY)
logger = logging.getLogger(__name__)

def get_llm_response(prompt: str, system_message: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        logger.info(f"Received response from LLM:\n{json.dumps(response.choices[0].message.content, indent=2)}")
        
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        logger.error(f"Raw response: {response.choices[0].message.content}")
        return {}
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
        return {}
