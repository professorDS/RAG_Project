# RAG_project/src/utils/validate_setup.py
import sys
import importlib
import os
from pathlib import Path
from typing import List

# Relative import of the project logger
from logger import logger

# Minimum python version
MIN_PYTHON = (3, 8)

# Packages that are mandatory for the pipeline to work
REQUIRED_PACKAGES = [
    ("faiss", "faiss"),  # faiss (faiss-cpu on pip)
    ("numpy", "numpy"),
    ("sentence-transformers", "sentence_transformers"),
    ("pypdf", "pypdf"),
    ("requests", "requests"),
]

# Packages that are optional but recommended
OPTIONAL_PACKAGES = [
    ("torch", "torch"),
    ("python-dotenv (for .env support)", "dotenv"),
]


def _check_python_version() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    if ok:
        logger.info("Python version OK: %s", sys.version.split()[0])
    else:
        logger.error(
            "Python version too old: %s. Minimum required is %s.%s",
            sys.version.split()[0],
            MIN_PYTHON[0],
            MIN_PYTHON[1],
        )
    return ok


def _importable(pkg_name: str) -> bool:
    try:
        importlib.import_module(pkg_name)
        return True
    except Exception:
        return False


def check_required_packages() -> List[str]:
    missing = []
    for nice, pkg in REQUIRED_PACKAGES:
        if not _importable(pkg):
            logger.error("Required package missing: %s (import name: %s)", nice, pkg)
            missing.append(pkg)
        else:
            logger.info("Found required package: %s", pkg)
    return missing


def check_optional_packages() -> List[str]:
    missing = []
    for nice, pkg in OPTIONAL_PACKAGES:
        if not _importable(pkg):
            logger.warning("Optional package not found: %s (import name: %s)", nice, pkg)
            missing.append(pkg)
        else:
            logger.info("Found optional package: %s", pkg)
    return missing


def try_load_dotenv() -> None:
    """
    Try to load .env into environment if python-dotenv is installed
    (non-fatal). This helps detect HF_TOKEN from .env files.
    """
    try:
        from dotenv import load_dotenv

        repo_root = Path(__file__).resolve().parents[2]
        dotenv_path = repo_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.info("Loaded environment variables from %s", dotenv_path)
        else:
            logger.debug(".env file not found at %s", dotenv_path)
    except Exception:
        logger.debug("python-dotenv not installed; skipping .env auto-load")


def check_hf_token(env_var_name: str = "HF_TOKEN") -> bool:
    """
    Checks that an HF token is available in environment variables.
    Returns True if token present, False otherwise.
    """
    try_load_dotenv()
    token = os.getenv(env_var_name)
    if token:
        logger.info("HuggingFace token detected in environment variable '%s'.", env_var_name)
        return True
    # Also check for a token file fallback (non recommended) e.g. .hf_token
    repo_root = Path(__file__).resolve().parents[2]
    token_file = repo_root / ".hf_token"
    if token_file.exists():
        logger.warning(
            "Found token file %s — this is less secure than using environment variables or .env. Consider moving to env/.env and adding %s to .gitignore.",
            token_file,
            env_var_name,
        )
        return True
    logger.error(
        "No HuggingFace token found. Set environment variable %s or create a .env with %s in the project root.",
        env_var_name,
        env_var_name,
    )
    return False


def validate(fail_on_missing_token: bool = True) -> None:
    """
    Run all checks. Exits the program with non-zero status on fatal errors.
    Set fail_on_missing_token False to skip stopping on token absence.
    """
    logger.info("Running environment validation checks...")

    if not _check_python_version():
        logger.error("Please install Python %s.%s or newer.", MIN_PYTHON[0], MIN_PYTHON[1])
        sys.exit(1)

    missing_required = check_required_packages()
    missing_optional = check_optional_packages()

    if missing_required:
        logger.error(
            "Missing required packages: %s. Run: pip install -r requirements.txt",
            ", ".join(missing_required),
        )
        sys.exit(1)

    has_token = check_hf_token()
    if not has_token and fail_on_missing_token:
        logger.error("HuggingFace token is required to proceed. Aborting.")
        sys.exit(1)

    logger.info("Environment validation passed. You're ready to run the pipeline.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate environment for RAG project")
    parser.add_argument(
        "--no-token-fail",
        action="store_true",
        help="Do not abort if HF_TOKEN is missing (useful for offline index building)",
    )
    args = parser.parse_args()
    validate(fail_on_missing_token=not args.no_token_fail)
