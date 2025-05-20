import configparser
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FIREWORKS_CONFIG_DIR = Path.home() / ".fireworks"
AUTH_INI_FILE = FIREWORKS_CONFIG_DIR / "auth.ini"


def get_fireworks_api_key() -> Optional[str]:
    """
    Retrieves the Fireworks API key.

    The key is sourced in the following order:
    1. FIREWORKS_API_KEY environment variable.
    2. 'api_key' from the [fireworks] section of ~/.fireworks/auth.ini.

    Returns:
        The API key if found, otherwise None.
    """
    # 1. Try environment variable
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        logger.debug("Using FIREWORKS_API_KEY from environment variable.")
        return api_key

    # 2. Try auth.ini file
    if AUTH_INI_FILE.exists():
        try:
            config = configparser.ConfigParser()
            config.read(AUTH_INI_FILE)
            if "fireworks" in config and "api_key" in config["fireworks"]:
                api_key_from_file = config["fireworks"]["api_key"]
                if api_key_from_file:
                    logger.debug(f"Using api_key from {AUTH_INI_FILE}.")
                    return api_key_from_file
        except configparser.Error as e:
            logger.warning(f"Error parsing {AUTH_INI_FILE}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error reading {AUTH_INI_FILE}: {e}")

    logger.debug("Fireworks API key not found in environment variables or auth.ini.")
    return None


def get_fireworks_account_id() -> Optional[str]:
    """
    Retrieves the Fireworks Account ID.

    The Account ID is sourced in the following order:
    1. FIREWORKS_ACCOUNT_ID environment variable.
    2. 'account_id' from the [fireworks] section of ~/.fireworks/auth.ini.

    Returns:
        The Account ID if found, otherwise None.
    """
    # 1. Try environment variable
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if account_id:
        logger.debug("Using FIREWORKS_ACCOUNT_ID from environment variable.")
        return account_id

    # 2. Try auth.ini file
    if AUTH_INI_FILE.exists():
        try:
            config = configparser.ConfigParser()
            config.read(AUTH_INI_FILE)
            if "fireworks" in config and "account_id" in config["fireworks"]:
                account_id_from_file = config["fireworks"]["account_id"]
                if account_id_from_file:
                    logger.debug(f"Using account_id from {AUTH_INI_FILE}.")
                    return account_id_from_file
        except configparser.Error as e:
            logger.warning(f"Error parsing {AUTH_INI_FILE}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error reading {AUTH_INI_FILE}: {e}")

    # Handle Dev API special case for account ID if FIREWORKS_API_BASE is set
    # This logic was present in the original file and might be relevant
    # if no account_id is found yet and a dev environment is detected.
    # However, the plan is to return None if not found.
    # For now, strictly adhere to the plan.
    # If this dev-specific logic is still needed, it should be handled by the caller
    # or re-evaluated if it belongs in this module.

    # api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    # if "dev.api.fireworks.ai" in api_base and account_id == "fireworks":
    # logger.info("Using development API base, defaulting to pyroworks-dev account")
    # account_id = "pyroworks-dev" # Default dev account

    logger.debug("Fireworks Account ID not found in environment variables or auth.ini.")
    return None
