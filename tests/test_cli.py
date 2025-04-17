import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from typer.testing import CliRunner

from reward_kit.cli import app, validate_function_path


@pytest.fixture
def cli_runner():
    """Fixture for CLI runner."""
    return CliRunner()


class TestCLI:
    """Tests for the CLI functionality."""
    
    def test_version_command(self, cli_runner):
        """Test the version command."""
        with patch("reward_kit.cli.__version__", "0.1.0"):
            result = cli_runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "0.1.0" in result.stdout
    
    def test_serve_command(self, cli_runner):
        """Test the serve command exists and has the expected help text."""
        # Just check that the command is registered and has the expected help text
        result = cli_runner.invoke(app, ["serve-app", "--help"])
        
        # Command should return help text successfully
        assert result.exit_code == 0
        assert "Serve a reward function as an HTTP API" in result.stdout
        assert "FUNCTION_PATH" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
    
    def test_validate_function_path(self):
        """Test the validation of function paths."""
        # We'll mock imports separately to test just the validate_function_path function
        with patch("reward_kit.cli.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.test_func = MagicMock()
            mock_import.return_value = mock_module
            
            # Test the function directly with a module:function format
            result = validate_function_path("test_module:test_func")
            
            # Module import should be called with correct name
            mock_import.assert_called_once_with("test_module")
            
            # Verify the result is the function object
            assert result is mock_module.test_func
    
    @patch("reward_kit.cli.validate_function_path")
    @patch("reward_kit.cli.uvicorn.run")
    def test_serve_error_handling(self, mock_run, mock_validate, cli_runner):
        """Test error handling in the serve command."""
        # Mock the validation function to raise an error
        mock_validate.side_effect = ImportError("Function not found")
        
        result = cli_runner.invoke(app, [
            "serve-app", 
            "invalid.path"
        ])
        
        # Should exit with error
        assert result.exit_code != 0
        # Error message should be displayed
        assert "Function not found" in result.stdout