import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Optional
from reward_kit.reward_function import RewardFunction, reward_function
from reward_kit.models import RewardOutput, MetricRewardOutput


def simple_reward_func(
    messages: List[Dict[str, str]], 
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """Example reward function for testing."""
    metrics = {
        "length": MetricRewardOutput(
            score=0.5,
            reason="Length-based score"
        )
    }
    return RewardOutput(score=0.5, metrics=metrics)


@reward_function
def decorated_reward_func(
    messages: List[Dict[str, str]], 
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """Example decorated reward function."""
    metrics = {
        "test": MetricRewardOutput(
            score=0.7,
            reason="Test score"
        )
    }
    return RewardOutput(score=0.7, metrics=metrics)


class TestRewardFunction:
    """Tests for the RewardFunction class."""
    
    def test_local_mode_function_path(self):
        """Test RewardFunction in local mode with function path."""
        with patch("reward_kit.reward_function.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.simple_reward_func = simple_reward_func
            mock_import.return_value = mock_module
            
            reward_fn = RewardFunction(
                func_path="test_module.simple_reward_func",
                mode="local"
            )
            
            test_msgs = [{"role": "user", "content": "Hello"}, 
                        {"role": "assistant", "content": "Hi there"}]
            orig_msgs = [test_msgs[0]]
            
            result = reward_fn(messages=test_msgs, original_messages=orig_msgs)
            assert result.score == 0.5
            assert "length" in result.metrics
            assert result.metrics["length"].score == 0.5
    
    def test_local_mode_function(self):
        """Test RewardFunction in local mode with direct function."""
        reward_fn = RewardFunction(
            func=simple_reward_func,
            mode="local"
        )
        
        test_msgs = [{"role": "user", "content": "Hello"}, 
                    {"role": "assistant", "content": "Hi there"}]
        orig_msgs = [test_msgs[0]]
        
        result = reward_fn(messages=test_msgs, original_messages=orig_msgs)
        assert result.score == 0.5
        assert "length" in result.metrics
        assert result.metrics["length"].score == 0.5
    
    def test_remote_mode(self):
        """Test RewardFunction in remote mode."""
        with patch("reward_kit.reward_function.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "score": 0.8,
                "metrics": {
                    "remote": {
                        "score": 0.8,
                        "reason": "Remote score"
                    }
                }
            }
            mock_post.return_value = mock_response
            
            reward_fn = RewardFunction(
                endpoint="https://example.com/reward",
                mode="remote"
            )
            
            test_msgs = [{"role": "user", "content": "Hello"}, 
                        {"role": "assistant", "content": "Hi there"}]
            orig_msgs = [test_msgs[0]]
            
            result = reward_fn(messages=test_msgs, original_messages=orig_msgs)
            assert result.score == 0.8
            assert "remote" in result.metrics
            assert result.metrics["remote"].score == 0.8
    
    def test_fireworks_hosted_mode(self):
        """Test RewardFunction in fireworks_hosted mode."""
        with patch("reward_kit.reward_function.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "score": 0.9,
                "metrics": {
                    "hosted": {
                        "score": 0.9,
                        "reason": "Hosted score"
                    }
                }
            }
            mock_post.return_value = mock_response
            
            reward_fn = RewardFunction(
                model_id="fireworks/test-model",
                mode="fireworks_hosted"
            )
            
            test_msgs = [{"role": "user", "content": "Hello"}, 
                        {"role": "assistant", "content": "Hi there"}]
            orig_msgs = [test_msgs[0]]
            
            result = reward_fn(messages=test_msgs, original_messages=orig_msgs)
            assert result.score == 0.9
            assert "hosted" in result.metrics
            assert result.metrics["hosted"].score == 0.9
    
    def test_get_trl_adapter(self):
        """Test getting a TRL adapter from a RewardFunction."""
        reward_fn = RewardFunction(
            func=simple_reward_func,
            mode="local"
        )
        
        trl_adapter = reward_fn.get_trl_adapter()
        assert callable(trl_adapter)
        
        # Test the adapter with a batch input
        test_msgs = [
            [{"role": "user", "content": "Hello"}, 
             {"role": "assistant", "content": "Hi there"}]
        ]
        orig_msgs = [[test_msgs[0][0]]]
        
        result = trl_adapter(test_msgs, orig_msgs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 0.5  # Just the score


class TestRewardFunctionDecorator:
    """Tests for the @reward_function decorator."""
    
    def test_decorator_basic_functionality(self):
        """Test basic functionality of the reward_function decorator."""
        test_msgs = [{"role": "user", "content": "Hello"}, 
                    {"role": "assistant", "content": "Hi there"}]
        orig_msgs = [test_msgs[0]]
        
        # Call the decorated function directly
        result = decorated_reward_func(messages=test_msgs, original_messages=orig_msgs)
        assert result.score == 0.7
        assert "test" in result.metrics
        assert result.metrics["test"].score == 0.7
    
    def test_decorator_deploy_method(self):
        """Test that the decorator adds a deploy method."""
        assert hasattr(decorated_reward_func, "deploy")
        assert callable(decorated_reward_func.deploy)
        
        # Directly patch the requests.post call for simplicity
        with patch("reward_kit.reward_function.requests.post") as mock_post:
            # Configure the response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "accounts/test-account/evaluators/test-123"}
            mock_post.return_value = mock_response
            
            # Test deploy method by providing account_id directly in the config
            deploy_result = decorated_reward_func.deploy(
                name="test-deployment", 
                account_id="test-account",  # Provide account_id directly
                auth_token="fake-token"     # Provide token directly
            )
            
            # Check the result is the evaluation ID
            assert deploy_result == "test-123"
            
            # Verify the API was called
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert "accounts/test-account/evaluators" in args[0]
            assert kwargs["headers"]["Authorization"] == "Bearer fake-token"