import logging
import os
from pathlib import Path

# Create logs directory if not exists
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

# Logging format
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  # print to terminal
    ]
)

# Main logger object accessible across project
logger = logging.getLogger("RAG_Project")

