import os
import logging
import configparser
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_authentication() -> Tuple[str, str]:
    """
    Get authentication information for the Fireworks API.

    This function attempts to retrieve the account_id and API key from:
    1. Environment variables (FIREWORKS_ACCOUNT_ID and FIREWORKS_API_KEY)
    2. Configuration files (~/.fireworks/settings.ini and ~/.fireworks/auth.ini)

    Returns:
        Tuple of (account_id, auth_token)

    Raises:
        ValueError: If either account_id or auth_token couldn't be found
    """
    # Try to get API key from environment
    auth_token = os.environ.get("FIREWORKS_API_KEY")
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")

    # Check if we have both account_id and auth_token from environment
    if auth_token and account_id:
        logger.debug("Using authentication from environment variables")
        return account_id, auth_token

    # If not found, try config files
    if not auth_token or not account_id:
        auth_token = get_auth_token() or auth_token
        account_id = get_account_id() or account_id

    # We need real authentication credentials
    if not account_id:
        raise ValueError(
            "Account ID not found. Set FIREWORKS_ACCOUNT_ID environment variable "
            "or configure ~/.fireworks/settings.ini"
        )

    if not auth_token:
        raise ValueError(
            "Auth token not found. Set FIREWORKS_API_KEY environment variable "
            "or configure ~/.fireworks/auth.ini"
        )

    # Handle Dev API special case for account ID
    api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    if "dev.api.fireworks.ai" in api_base and account_id == "fireworks":
        logger.info("Using development API base, defaulting to pyroworks-dev account")
        account_id = "pyroworks-dev"  # Default dev account

    return account_id, auth_token


def get_auth_token() -> Optional[str]:
    """
    Get auth token from config files.

    Returns:
        The auth token if found, None otherwise
    """
    auth_path = Path.home() / ".fireworks" / "auth.ini"
    if not auth_path.exists():
        return None

    # Try to read auth.ini with standard configparser
    try:
        auth_config = configparser.ConfigParser()
        auth_config.read(auth_path)
        if "default" in auth_config and "id_token" in auth_config["default"]:
            return auth_config["default"]["id_token"]
    except Exception:
        # If standard parsing fails, try to read as key-value pairs
        try:
            with open(auth_path, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "id_token":
                            return value
        except Exception as e:
            logger.warning(f"Error reading auth.ini as key-value pairs: {str(e)}")

    return None


def get_account_id() -> Optional[str]:
    """
    Get account ID from config files.

    Returns:
        The account ID if found, None otherwise
    """
    settings_path = Path.home() / ".fireworks" / "settings.ini"
    if not settings_path.exists():
        return None

    # Try to read settings.ini with standard configparser
    try:
        settings_config = configparser.ConfigParser()
        settings_config.read(settings_path)
        if "default" in settings_config and "account_id" in settings_config["default"]:
            return settings_config["default"]["account_id"]
    except Exception:
        # If standard parsing fails, try to read as key-value pairs
        try:
            with open(settings_path, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "account_id":
                            return value
        except Exception as e:
            logger.warning(f"Error reading settings.ini as key-value pairs: {str(e)}")

    return None