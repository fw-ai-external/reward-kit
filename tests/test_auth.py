import configparser  # Import the original for type hinting if needed, but not for spec.
import os

# Import the original ConfigParser for use in spec if absolutely necessary,
# though direct configuration of the mock instance is preferred.
from configparser import ConfigParser as OriginalConfigParser
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import the SUT
from reward_kit.auth import (
    AUTH_INI_FILE,
    get_fireworks_account_id,
    get_fireworks_api_key,
)

# Test data
TEST_ENV_API_KEY = "test_env_api_key_123"
TEST_ENV_ACCOUNT_ID = "test_env_account_id_456"
INI_API_KEY = "ini_api_key_abc"
INI_ACCOUNT_ID = "ini_account_id_def"


@pytest.fixture(autouse=True)
def clear_env_vars_fixture():
    env_vars_to_clear = ["FIREWORKS_API_KEY", "FIREWORKS_ACCOUNT_ID"]
    original_values = {var: os.environ.get(var) for var in env_vars_to_clear}
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    yield
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


# --- Tests for get_fireworks_api_key ---


def test_get_api_key_from_env():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY
    assert get_fireworks_api_key() == TEST_ENV_API_KEY


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")  # Mocks the ConfigParser class
def test_get_api_key_from_ini(mock_ConfigParser_class, mock_path_exists):
    # Configure the instance that configparser.ConfigParser() will return
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    # Simulate key found in [fireworks] section
    def has_option_fireworks_true(section, option):
        if section == "fireworks":
            return option == "api_key"
        return False  # Not in default or other sections for this test

    def get_fireworks_value(section, option):
        if section == "fireworks" and option == "api_key":
            return INI_API_KEY
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_fireworks_true
    mock_parser_instance.get.side_effect = get_fireworks_value
    # Ensure 'fireworks' section itself exists
    mock_parser_instance.__contains__.side_effect = (
        lambda item: item == "fireworks"
    )  # For "fireworks" in config check

    with patch(
        "builtins.open", mock_open(read_data="[fireworks]\napi_key = foo")
    ):  # Actual read_data not used by mock parser
        assert get_fireworks_api_key() == INI_API_KEY

    mock_path_exists.assert_called_once_with()
    mock_ConfigParser_class.assert_called_once_with()  # Class was instantiated
    mock_parser_instance.read.assert_called_once_with(AUTH_INI_FILE)


def test_get_api_key_env_overrides_ini():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY
    with patch("pathlib.Path.exists") as mock_path_exists, patch(
        "configparser.ConfigParser"
    ) as mock_ConfigParser_class:
        assert get_fireworks_api_key() == TEST_ENV_API_KEY
        mock_path_exists.assert_not_called()
        mock_ConfigParser_class.assert_not_called()


@patch("pathlib.Path.exists", return_value=False)
def test_get_api_key_not_found(mock_path_exists):
    assert get_fireworks_api_key() is None
    mock_path_exists.assert_called_once_with()


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_exists_no_section(mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]
    # "fireworks" section does not exist
    mock_parser_instance.__contains__.side_effect = (
        lambda item: item != "fireworks"
    )  # For "fireworks" in config check
    # Simulate default section also not having the key
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Simulate MissingSectionHeaderError to trigger fallback
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError(
        "file", 1, "line"
    )
    # Fallback parsing will not find 'api_key' in this data
    with patch(
        "builtins.open",
        mock_open(read_data="other_key = some_val_but_no_section_header\nanother=val"),
    ):
        assert get_fireworks_api_key() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_exists_no_key_option(  # Tests [fireworks] exists, but no api_key in it, and not in default, configparser parses OK (no fallback)
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [
        str(AUTH_INI_FILE)
    ]  # Simulate successful read

    # "fireworks" section exists
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    # but no "api_key" option in it, and not in default section
    mock_parser_instance.has_option.side_effect = lambda section, option: False

    with patch("builtins.open", mock_open(read_data="[fireworks]\nsome_other_key=foo")):
        assert get_fireworks_api_key() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_empty_value(
    mock_ConfigParser_class, mock_path_exists
):  # Key exists in [fireworks] but value is empty
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    # "fireworks" section exists and has "api_key"
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = (
        lambda section, option: section == "fireworks" and option == "api_key"
    )
    # but its value is empty
    mock_parser_instance.get.side_effect = lambda section, option: (
        ""
        if section == "fireworks" and option == "api_key"
        else configparser.NoOptionError(option, section)
    )

    with patch("builtins.open", mock_open(read_data="[fireworks]\napi_key=")):
        assert (
            get_fireworks_api_key() is None
        )  # Empty string treated as not found by SUT


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_from_ini_default_section_success(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [
        str(AUTH_INI_FILE)
    ]  # Simulate successful read

    def has_option_logic(section, option):
        if section == "fireworks":  # Not in [fireworks]
            return False
        # Is in default section
        return section == mock_parser_instance.default_section and option == "api_key"

    def get_logic(section, option):
        if section == mock_parser_instance.default_section and option == "api_key":
            return INI_API_KEY
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_logic
    mock_parser_instance.get.side_effect = get_logic
    mock_parser_instance.__contains__.side_effect = (
        lambda item: item != "fireworks"
    )  # No 'fireworks' section

    with patch("builtins.open", mock_open(read_data="api_key = ini_api_key_abc")):
        assert get_fireworks_api_key() == INI_API_KEY


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_from_ini_fallback_parsing_success(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Simulate MissingSectionHeaderError to trigger fallback
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError(
        "mocked file", 1, "mocked line content"
    )

    file_content = f"api_key = {INI_API_KEY}\nother_key = value"
    with patch("builtins.open", mock_open(read_data=file_content)):
        assert get_fireworks_api_key() == INI_API_KEY


# --- Tests for get_fireworks_account_id ---


def test_get_account_id_from_env():
    os.environ["FIREWORKS_ACCOUNT_ID"] = TEST_ENV_ACCOUNT_ID
    assert get_fireworks_account_id() == TEST_ENV_ACCOUNT_ID


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_from_ini(mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    # Simulate key found in [fireworks] section
    def has_option_fireworks_true(section, option):
        if section == "fireworks":
            return option == "account_id"
        return False  # Not in default or other sections for this test

    def get_fireworks_value(section, option):
        if section == "fireworks" and option == "account_id":
            return INI_ACCOUNT_ID
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_fireworks_true
    mock_parser_instance.get.side_effect = get_fireworks_value
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"

    with patch("builtins.open", mock_open(read_data="[fireworks]\naccount_id = foo")):
        assert get_fireworks_account_id() == INI_ACCOUNT_ID

    mock_path_exists.assert_called_once_with()
    mock_ConfigParser_class.assert_called_once_with()
    mock_parser_instance.read.assert_called_once_with(AUTH_INI_FILE)


def test_get_account_id_env_overrides_ini():
    os.environ["FIREWORKS_ACCOUNT_ID"] = TEST_ENV_ACCOUNT_ID
    with patch("pathlib.Path.exists") as mock_path_exists, patch(
        "configparser.ConfigParser"
    ) as mock_ConfigParser_class:
        assert get_fireworks_account_id() == TEST_ENV_ACCOUNT_ID
        mock_path_exists.assert_not_called()
        mock_ConfigParser_class.assert_not_called()


@patch("pathlib.Path.exists", return_value=False)
def test_get_account_id_not_found(mock_path_exists):
    assert get_fireworks_account_id() is None
    mock_path_exists.assert_called_once_with()


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_exists_no_section(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Simulate MissingSectionHeaderError to trigger fallback
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError(
        "file", 1, "line"
    )
    # Fallback parsing will not find 'account_id' in this data
    with patch(
        "builtins.open",
        mock_open(read_data="other_key = some_val_but_no_section_header\nanother=val"),
    ):
        assert get_fireworks_account_id() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_exists_no_id_option(  # Tests [fireworks] exists, but no account_id in it, and not in default, configparser parses OK
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [
        str(AUTH_INI_FILE)
    ]  # Simulate successful read

    mock_parser_instance.__contains__.side_effect = (
        lambda item: item == "fireworks"
    )  # Has [fireworks] section
    # No "account_id" option in [fireworks] or default section
    mock_parser_instance.has_option.side_effect = lambda section, option: False

    with patch("builtins.open", mock_open(read_data="[fireworks]\nsome_other_key=foo")):
        assert get_fireworks_account_id() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_empty_value(
    mock_ConfigParser_class, mock_path_exists
):  # Key exists in [fireworks] but value is empty
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = (
        lambda section, option: section == "fireworks" and option == "account_id"
    )
    mock_parser_instance.get.side_effect = lambda section, option: (
        ""
        if section == "fireworks" and option == "account_id"
        else configparser.NoOptionError(option, section)
    )

    with patch("builtins.open", mock_open(read_data="[fireworks]\naccount_id=")):
        assert (
            get_fireworks_account_id() is None
        )  # Empty string treated as not found by SUT


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_from_ini_default_section_success(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    def has_option_logic(section, option):
        if section == "fireworks":  # Not in [fireworks]
            return False
        # Is in default section
        return (
            section == mock_parser_instance.default_section and option == "account_id"
        )

    def get_logic(section, option):
        if section == mock_parser_instance.default_section and option == "account_id":
            return INI_ACCOUNT_ID
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_logic
    mock_parser_instance.get.side_effect = get_logic
    mock_parser_instance.__contains__.side_effect = (
        lambda item: item != "fireworks"
    )  # No 'fireworks' section

    with patch("builtins.open", mock_open(read_data="account_id = ini_account_id_def")):
        assert get_fireworks_account_id() == INI_ACCOUNT_ID


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_from_ini_fallback_parsing_success(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError(
        "mocked file", 1, "mocked line content"
    )

    file_content = f"account_id = {INI_ACCOUNT_ID}\nother_key = value"
    with patch("builtins.open", mock_open(read_data=file_content)):
        assert get_fireworks_account_id() == INI_ACCOUNT_ID


# --- Tests for error handling ---


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_parse_error(mock_ConfigParser_class, mock_path_exists, caplog):
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Use the original ConfigParser's Error for side_effect
    mock_parser_instance.read.side_effect = configparser.Error("Mocked Parsing Error")

    with patch("builtins.open", mock_open(read_data="malformed ini content")):
        assert get_fireworks_api_key() is None
    # Adjusted to match the new log message format
    assert "Configparser error reading" in caplog.text
    assert "Mocked Parsing Error" in caplog.text


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_parse_error(
    mock_ConfigParser_class, mock_path_exists, caplog
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = configparser.Error("Mocked Parsing Error")

    with patch("builtins.open", mock_open(read_data="malformed ini content")):
        assert get_fireworks_account_id() is None
    # Adjusted to match the new log message format
    assert "Configparser error reading" in caplog.text
    assert "Mocked Parsing Error" in caplog.text


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_unexpected_error_reading_ini(
    mock_ConfigParser_class, mock_path_exists, caplog
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = Exception("Unexpected Read Error")

    with patch("builtins.open", mock_open(read_data="ini content")):
        assert get_fireworks_api_key() is None
    assert "Unexpected error reading" in caplog.text
    assert "Unexpected Read Error" in caplog.text
