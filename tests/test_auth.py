import os
import configparser  # Import the original for type hinting if needed, but not for spec.
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

# Import the SUT
from reward_kit.auth import (
    get_fireworks_api_key,
    get_fireworks_account_id,
    AUTH_INI_FILE,
)

# Import the original ConfigParser for use in spec if absolutely necessary,
# though direct configuration of the mock instance is preferred.
from configparser import ConfigParser as OriginalConfigParser

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

    # Simulate config["fireworks"]["api_key"]
    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        True  # "api_key" in config["fireworks"]
    )
    fireworks_section_mock.__getitem__.return_value = (
        INI_API_KEY  # config["fireworks"]["api_key"]
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch(
        "builtins.open", mock_open(read_data="")
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
    mock_parser_instance.__contains__.return_value = False  # "fireworks" not in config

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_api_key() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_exists_no_key_option(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        False  # "api_key" not in config["fireworks"]
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_api_key() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_empty_value(mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        True  # "api_key" in config["fireworks"]
    )
    fireworks_section_mock.__getitem__.return_value = (
        ""  # config["fireworks"]["api_key"] is empty
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_api_key() is None


# --- Tests for get_fireworks_account_id ---


def test_get_account_id_from_env():
    os.environ["FIREWORKS_ACCOUNT_ID"] = TEST_ENV_ACCOUNT_ID
    assert get_fireworks_account_id() == TEST_ENV_ACCOUNT_ID


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_from_ini(mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        True  # "account_id" in config["fireworks"]
    )
    fireworks_section_mock.__getitem__.return_value = (
        INI_ACCOUNT_ID  # config["fireworks"]["account_id"]
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch("builtins.open", mock_open(read_data="")):
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
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]
    mock_parser_instance.__contains__.return_value = False  # "fireworks" not in config

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_account_id() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_exists_no_id_option(
    mock_ConfigParser_class, mock_path_exists
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        False  # "account_id" not in config["fireworks"]
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_account_id() is None


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_account_id_ini_empty_value(mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    fireworks_section_mock = MagicMock()
    fireworks_section_mock.__contains__.return_value = (
        True  # "account_id" in config["fireworks"]
    )
    fireworks_section_mock.__getitem__.return_value = (
        ""  # config["fireworks"]["account_id"] is empty
    )

    mock_parser_instance.__contains__.return_value = True  # "fireworks" in config
    mock_parser_instance.__getitem__.return_value = fireworks_section_mock

    with patch("builtins.open", mock_open(read_data="")):
        assert get_fireworks_account_id() is None


# --- Tests for error handling ---


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
def test_get_api_key_ini_parse_error(mock_ConfigParser_class, mock_path_exists, caplog):
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Use the original ConfigParser's Error for side_effect
    mock_parser_instance.read.side_effect = configparser.Error("Mocked Parsing Error")

    with patch("builtins.open", mock_open(read_data="malformed ini content")):
        assert get_fireworks_api_key() is None
    assert "Error parsing" in caplog.text
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
    assert "Error parsing" in caplog.text
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
