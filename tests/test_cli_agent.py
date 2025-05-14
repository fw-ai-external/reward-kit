"""
Tests for the agent-eval CLI command.
"""
import pytest
import argparse
import asyncio # Added import
from unittest.mock import patch, MagicMock, mock_open, AsyncMock # Added AsyncMock
import json
import yaml 

from reward_kit.cli_commands.agent_eval_cmd import agent_eval_command
from reward_kit.models import TaskDefinitionModel

MINIMAL_TASK_DEF_CONTENT_DICT = {
    "name": "CLI Test Task",
    "description": "Task for CLI test.",
    "resource_type": "PythonStateResource",
    "base_resource_config": {},
    "reward_function_path": "test_module.test_reward",
    "goal_description": "Test goal.",
    "poc_max_turns": 1
}
MINIMAL_TASK_DEF_YAML_CONTENT = yaml.dump(MINIMAL_TASK_DEF_CONTENT_DICT)
MINIMAL_TASK_DEF_JSON_CONTENT = json.dumps(MINIMAL_TASK_DEF_CONTENT_DICT)


class TestAgentEvalCommand:
    """Tests for the agent_eval_command function."""

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("reward_kit.cli_commands.agent_eval_cmd.Orchestrator")
    @patch("builtins.open", new_callable=mock_open, read_data=MINIMAL_TASK_DEF_YAML_CONTENT)
    def test_agent_eval_success_yaml(self, mock_file_open, MockOrchestrator, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True

        mock_orchestrator_instance = MockOrchestrator.return_value
        mock_orchestrator_instance.execute_task_poc = AsyncMock(return_value={"score": 1.0}) # Use AsyncMock
        mock_orchestrator_instance.setup_base_resource = AsyncMock() # Use AsyncMock

        args = argparse.Namespace(task_def="dummy_task.yaml", verbose=False, debug=False)
        result = agent_eval_command(args)

        assert result == 0
        MockPath.assert_called_once_with("dummy_task.yaml")
        mock_file_open.assert_called_once_with(mock_path_instance, 'r')
        MockOrchestrator.assert_called_once()
        # Orchestrator is called with task_definition as a keyword argument
        assert 'task_definition' in MockOrchestrator.call_args.kwargs
        assert isinstance(MockOrchestrator.call_args.kwargs['task_definition'], TaskDefinitionModel)
        assert MockOrchestrator.call_args.kwargs['task_definition'].name == "CLI Test Task"
        mock_orchestrator_instance.execute_task_poc.assert_awaited_once()
        assert "agent-eval command finished successfully" in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("reward_kit.cli_commands.agent_eval_cmd.Orchestrator")
    @patch("builtins.open", new_callable=mock_open, read_data=MINIMAL_TASK_DEF_JSON_CONTENT)
    @patch("reward_kit.cli_commands.agent_eval_cmd.yaml", None) 
    def test_agent_eval_success_json_no_yaml_lib(self, mock_file_open, MockOrchestrator, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True

        mock_orchestrator_instance = MockOrchestrator.return_value
        mock_orchestrator_instance.execute_task_poc = AsyncMock(return_value={"score": 1.0}) # Use AsyncMock
        mock_orchestrator_instance.setup_base_resource = AsyncMock() # Use AsyncMock

        args = argparse.Namespace(task_def="dummy_task.json")
        result = agent_eval_command(args)

        assert result == 0
        assert "PyYAML not installed. Attempting to parse task definition as JSON." in caplog.text
        MockOrchestrator.assert_called_once()
        mock_orchestrator_instance.execute_task_poc.assert_awaited_once()

    def test_agent_eval_no_task_def_arg(self, caplog):
        args = argparse.Namespace(task_def=None) 
        result = agent_eval_command(args)
        assert result == 1
        assert "Error: --task-def (path to task definition YAML file) is required." in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_task_def_file_not_found(self, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = False
        mock_path_instance.__str__ = MagicMock(return_value="non_existent_task.yaml") # For logging
        
        args = argparse.Namespace(task_def="non_existent_task.yaml")
        result = agent_eval_command(args)
        assert result == 1
        assert "Error: Task definition file not found or is not a file: non_existent_task.yaml" in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("builtins.open", new_callable=mock_open, read_data="invalid yaml content: %&^")
    def test_agent_eval_invalid_yaml_content(self, mock_file_open, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="invalid_task.yaml") # For logging
        
        args = argparse.Namespace(task_def="invalid_task.yaml")
        result = agent_eval_command(args)
        assert result == 1
        assert "Error parsing YAML file invalid_task.yaml" in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("builtins.open", new_callable=mock_open, read_data='{"name": "Only Name"}') 
    def test_agent_eval_pydantic_validation_error(self, mock_file_open, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        
        args = argparse.Namespace(task_def="incomplete_task.yaml")
        result = agent_eval_command(args)
        assert result == 1
        assert "Invalid task definition file structure" in caplog.text
        assert "resource_type" in caplog.text 
        assert "Field required" in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("reward_kit.cli_commands.agent_eval_cmd.Orchestrator")
    @patch("builtins.open", new_callable=mock_open, read_data=MINIMAL_TASK_DEF_YAML_CONTENT)
    def test_agent_eval_orchestrator_instantiation_fails(self, mock_file_open, MockOrchestrator, MockPath, caplog):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        MockOrchestrator.side_effect = RuntimeError("Orchestrator init failed")
        args = argparse.Namespace(task_def="dummy_task.yaml")
        result = agent_eval_command(args)
        assert result == 1
        assert "Error instantiating Orchestrator: Orchestrator init failed" in caplog.text

    @patch("reward_kit.cli_commands.agent_eval_cmd.Path")
    @patch("reward_kit.cli_commands.agent_eval_cmd.Orchestrator")
    @patch("builtins.open", new_callable=mock_open, read_data=MINIMAL_TASK_DEF_YAML_CONTENT)
    def test_agent_eval_orchestrator_execution_fails(self, mock_file_open, MockOrchestrator, MockPath, caplog): # Removed async
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True

        mock_orchestrator_instance = MockOrchestrator.return_value
        mock_orchestrator_instance.execute_task_poc = AsyncMock(side_effect=RuntimeError("POC execution failed")) # Use AsyncMock
        mock_orchestrator_instance.setup_base_resource = AsyncMock() # Use AsyncMock

        args = argparse.Namespace(task_def="dummy_task.yaml")
        result = agent_eval_command(args) 
        assert result == 1
        assert "Error during agent-eval execution: POC execution failed" in caplog.text
