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
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        logger.debug("Using FIREWORKS_API_KEY from environment variable.")
        return api_key

    if AUTH_INI_FILE.exists():
        try:
            config = configparser.ConfigParser()
            config.read(AUTH_INI_FILE)
            # Try [fireworks] section first
            if "fireworks" in config and config.has_option("fireworks", "api_key"):
                api_key_from_file = config.get("fireworks", "api_key")
                if api_key_from_file:
                    logger.debug(
                        f"Using api_key from [fireworks] section in {AUTH_INI_FILE}."
                    )
                    return api_key_from_file
            # If not found in [fireworks] section, try the default section
            if config.has_option(config.default_section, "api_key"):
                api_key_from_default = config.get(config.default_section, "api_key")
                if api_key_from_default:
                    logger.debug(
                        f"Using api_key from default section [{config.default_section}] in {AUTH_INI_FILE}."
                    )
                    return api_key_from_default
        except configparser.MissingSectionHeaderError:
            # If MissingSectionHeaderError occurs, try parsing as simple key-value pairs
            logger.warning(
                f"{AUTH_INI_FILE} has no section headers. Attempting simple key-value parsing for 'api_key'."
            )
            try:
                with open(AUTH_INI_FILE, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            if key == "api_key" and value:
                                logger.debug(
                                    f"Using api_key from simple key-value parsing of {AUTH_INI_FILE}."
                                )
                                return value
            except Exception as e_simple:
                logger.warning(
                    f"Error during simple key-value parsing of {AUTH_INI_FILE}: {e_simple}"
                )
        except configparser.Error as e_config:
            logger.warning(f"Configparser error reading {AUTH_INI_FILE}: {e_config}")
        except Exception as e_general:
            logger.warning(f"Unexpected error reading {AUTH_INI_FILE}: {e_general}")

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
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if account_id:
        logger.debug("Using FIREWORKS_ACCOUNT_ID from environment variable.")
        return account_id

    if AUTH_INI_FILE.exists():
        try:
            config = configparser.ConfigParser()
            config.read(AUTH_INI_FILE)
            # Try [fireworks] section first
            if "fireworks" in config and config.has_option("fireworks", "account_id"):
                account_id_from_file = config.get("fireworks", "account_id")
                if account_id_from_file:
                    logger.debug(
                        f"Using account_id from [fireworks] section in {AUTH_INI_FILE}."
                    )
                    return account_id_from_file
            # If not found in [fireworks] section, try the default section
            if config.has_option(config.default_section, "account_id"):
                account_id_from_default = config.get(
                    config.default_section, "account_id"
                )
                if account_id_from_default:
                    logger.debug(
                        f"Using account_id from default section [{config.default_section}] in {AUTH_INI_FILE}."
                    )
                    return account_id_from_default
        except configparser.MissingSectionHeaderError:
            logger.warning(
                f"{AUTH_INI_FILE} has no section headers. Attempting simple key-value parsing for 'account_id'."
            )
            try:
                with open(AUTH_INI_FILE, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            if key == "account_id" and value:
                                logger.debug(
                                    f"Using account_id from simple key-value parsing of {AUTH_INI_FILE}."
                                )
                                return value
            except Exception as e_simple:
                logger.warning(
                    f"Error during simple key-value parsing of {AUTH_INI_FILE}: {e_simple}"
                )
        except configparser.Error as e_config:
            logger.warning(f"Configparser error reading {AUTH_INI_FILE}: {e_config}")
        except Exception as e_general:
            logger.warning(f"Unexpected error reading {AUTH_INI_FILE}: {e_general}")

    logger.debug("Fireworks Account ID not found in environment variables or auth.ini.")
    return None


def get_fireworks_api_base() -> str:
    """
    Retrieves the Fireworks API base URL.

    The base URL is sourced from the FIREWORKS_API_BASE environment variable.
    If not set, it defaults to "https://api.fireworks.ai".

    Returns:
        The API base URL.
    """
    api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    if os.environ.get("FIREWORKS_API_BASE"):
        logger.debug("Using FIREWORKS_API_BASE from environment variable.")
    else:
        logger.debug(
            f"FIREWORKS_API_BASE not set in environment, defaulting to {api_base}."
        )
    return api_base
