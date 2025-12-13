import os
from dotenv import load_dotenv
from src.utils.logger import logger

# Load from .env file
load_dotenv()

def load_hf_token(env_var_name: str = "HF_TOKEN") -> str:
    token = os.getenv(env_var_name)

    if not token:
        logger.error(f"HuggingFace token not found in .env file: '{env_var_name}'")
        raise EnvironmentError("Missing HF token in .env")

    logger.info("HuggingFace token loaded successfully from .env")
    return token
