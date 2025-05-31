"""
Model clients for generating responses from various LLM APIs.
"""

import abc
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ModelClient(abc.ABC):
    """Abstract base class for model clients."""

    def __init__(self, client_config: DictConfig):
        self.client_config = client_config
        self.model_name = client_config.get("model_name", "unknown_model")
        self.temperature = client_config.get("temperature", 0.0)
        self.max_tokens = client_config.get("max_tokens", 1024)
        self.top_p = client_config.get("top_p", 0.95)
        self.top_k = client_config.get("top_k", 20)
        self.min_p = client_config.get("min_p", 0.0)
        # Add reasoning_effort, defaulting to None if not specified in config
        self.reasoning_effort = client_config.get("reasoning_effort", None)

    @abc.abstractmethod
    async def generate(
        self, messages: List[Dict[str, str]], session: aiohttp.ClientSession
    ) -> Optional[str]:
        """Generates a response from the model given a list of messages."""
        pass


class FireworksModelClient(ModelClient):
    """Client for Fireworks AI models."""

    def __init__(self, client_config: DictConfig, api_key: str):
        super().__init__(client_config)
        self.api_key = api_key
        self.api_base = client_config.get(
            "api_base", "https://api.fireworks.ai/inference/v1"
        )
        # TODO: Initialize rate limiter, retry policy from client_config.api_params

    async def generate(
        self, messages: List[Dict[str, str]], session: aiohttp.ClientSession
    ) -> Optional[str]:
        url = f"{self.api_base}/chat/completions"
        # TEMPORARY FIX: Use minimal parameters to avoid API issues
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Create debug payload with properly truncated content to avoid modifying original
        import json

        debug_payload_log = json.loads(json.dumps(payload))  # Deep copy
        if "messages" in debug_payload_log and debug_payload_log["messages"]:
            debug_payload_log["messages"][-1]["content"] = (
                debug_payload_log["messages"][-1]["content"][:50] + "..."
            )
        logger.debug(f"Calling Fireworks API: {url}, Payload: {debug_payload_log}")

        # TODO: Implement robust retries (e.g., with tenacity) and rate limiting.
        # The following is a simplified version.
        try:
            for attempt in range(
                self.client_config.get("api_params", {}).get("max_retries", 3) + 1
            ):
                # TODO: Implement rate limiting before the call
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("choices") and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if choice.get("message") and choice["message"].get(
                                "content"
                            ):
                                return choice["message"]["content"].strip()
                        logger.warning(f"Fireworks API response malformed: {data}")
                        return None
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        logger.warning(
                            f"Rate limited. Retrying after {retry_after}s (attempt {attempt+1})."
                        )
                        await asyncio.sleep(retry_after)
                    elif response.status in [401, 403]:  # Auth errors
                        error_text = await response.text()
                        logger.error(
                            f"Fireworks API Auth Error ({response.status}): {error_text}"
                        )
                        # raise ModelAuthError(f"Fireworks API Auth Error ({response.status}): {error_text}")
                        return None  # Or raise specific error
                    elif response.status >= 500:  # Server errors
                        logger.warning(
                            f"Fireworks API Server Error ({response.status}). Retrying (attempt {attempt+1})."
                        )
                        await asyncio.sleep(2**attempt)
                    else:  # Other client errors
                        error_text = await response.text()
                        logger.error(
                            f"Fireworks API request failed ({response.status}): {error_text}"
                        )
                        return None
            logger.error("Max retries reached for Fireworks API call.")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP client error: {e}")
            # raise ModelGenerationError(f"AIOHTTP client error: {e}") from e
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error in FireworksModelClient: {e}", exc_info=True
            )
            # raise ModelGenerationError(f"Unexpected error: {e}") from e
            return None
