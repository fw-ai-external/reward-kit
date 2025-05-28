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
async def test_fireworks_client_generate_with_basic_params(mock_session: AsyncMock):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
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
    # Check only the basic parameters that are actually sent
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 50
    assert payload["messages"] == messages


@pytest.mark.asyncio
async def test_fireworks_client_generate_minimal_config(
    mock_session: AsyncMock,
):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            # Only model_name specified, others should use defaults
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
    # Verify minimal payload structure
    assert payload["model"] == "test-model"
    assert "temperature" in payload  # Should have default
    assert "max_tokens" in payload  # Should have default
    assert "messages" in payload


@pytest.mark.asyncio
async def test_fireworks_client_generate_payload_structure(mock_session: AsyncMock):
    client_config = DictConfig(
        {
            "model_name": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
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
    # Verify the current minimal payload structure (no optional params)
    expected_keys = {"model", "messages", "temperature", "max_tokens"}
    assert set(payload.keys()) == expected_keys


@pytest.mark.asyncio
async def test_fireworks_client_generate_default_params(mock_session: AsyncMock):
    # Test that default params are correctly set if not in config
    client_config = DictConfig(
        {
            "model_name": "test-model",
            # temperature, max_tokens omitted - should use defaults
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
    # Verify optional parameters are NOT in the payload (minimal approach)
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert "min_p" not in payload
    assert "reasoning_effort" not in payload
