import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from omegaconf import DictConfig

from reward_kit.generation.clients import FireworksModelClient


@pytest.fixture
def mock_session():
    """Fixture for a mocked aiohttp.ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.post = AsyncMock()

    # Default successful response
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"choices": [{"message": {"content": "Test response content"}}]}
    )
    mock_response.text = AsyncMock(
        return_value='{"choices": [{"message": {"content": "Test response content"}}]}'
    )
    session.post.return_value.__aenter__.return_value = mock_response  # For async with
    session.post.return_value.__aexit__.return_value = None
    return session


@pytest.mark.asyncio
async def test_fireworks_client_generate_with_reasoning_effort(mock_session: AsyncMock):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
            "reasoning_effort": "low",  # Test with reasoning_effort
            "api_params": {"max_retries": 1},
        }
    )
    api_key = "test_api_key"
    client = FireworksModelClient(client_config, api_key)

    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(messages, mock_session)

    mock_session.post.assert_called_once()
    called_args, called_kwargs = mock_session.post.call_args

    assert "json" in called_kwargs
    payload = called_kwargs["json"]
    assert "reasoning_effort" in payload
    assert payload["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_fireworks_client_generate_without_reasoning_effort(
    mock_session: AsyncMock,
):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
            # reasoning_effort is omitted
            "api_params": {"max_retries": 1},
        }
    )
    api_key = "test_api_key"
    client = FireworksModelClient(client_config, api_key)

    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(messages, mock_session)

    mock_session.post.assert_called_once()
    called_args, called_kwargs = mock_session.post.call_args

    assert "json" in called_kwargs
    payload = called_kwargs["json"]
    assert "reasoning_effort" not in payload  # Should not be present if not in config


@pytest.mark.asyncio
async def test_fireworks_client_generate_reasoning_effort_none(mock_session: AsyncMock):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
            "reasoning_effort": None,  # Explicitly None
            "api_params": {"max_retries": 1},
        }
    )
    api_key = "test_api_key"
    client = FireworksModelClient(client_config, api_key)

    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(messages, mock_session)

    mock_session.post.assert_called_once()
    called_args, called_kwargs = mock_session.post.call_args

    assert "json" in called_kwargs
    payload = called_kwargs["json"]
    # The client logic currently adds reasoning_effort to payload if self.reasoning_effort is not None.
    # So if config has it as None, it should not be in payload.
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_fireworks_client_generate_default_params(mock_session: AsyncMock):
    # Test that other defaultable params are correctly set if not in config
    client_config = DictConfig(
        {
            "model_name": "test-model",
            # temperature, max_tokens, top_p, top_k, min_p, reasoning_effort omitted
            "api_params": {"max_retries": 1},
        }
    )
    api_key = "test_api_key"
    client = FireworksModelClient(client_config, api_key)

    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(messages, mock_session)

    mock_session.post.assert_called_once()
    _, called_kwargs = mock_session.post.call_args
    payload = called_kwargs["json"]

    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.0  # Default from ModelClient
    assert payload["max_tokens"] == 1024  # Default from ModelClient
    assert payload["top_p"] == 0.95  # Default from ModelClient
    assert payload["top_k"] == 20  # Default from ModelClient
    assert payload["min_p"] == 0.0  # Default from ModelClient
    assert "reasoning_effort" not in payload  # Default from ModelClient is None
