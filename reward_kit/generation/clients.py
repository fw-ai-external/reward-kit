"""
Model clients for generating responses from various LLM APIs.
"""

import abc
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig

# Assuming FireworksAuthError might be a common exception to raise from clients
# from ..exceptions import ModelGenerationError, ModelAuthError, RateLimitError # Define these later

logger = logging.getLogger(__name__)


class ModelClient(abc.ABC):
    """Abstract base class for model clients."""

    def __init__(self, client_config: DictConfig):
        self.client_config = client_config
        # Common config might include model_name, temperature, max_tokens, api_base, etc.
        self.model_name = client_config.get("model_name", "unknown_model")
        self.temperature = client_config.get("temperature", 0.0)
        self.max_tokens = client_config.get("max_tokens", 1024)

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

        logger.debug(
            f"Calling Fireworks API: {url}, Model: {self.model_name}, Temp: {self.temperature}, Prompt: {messages[-1]['content'][:50]}..."
        )

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


# Example usage (for testing this module, not for direct use by pipeline)
# async def main_test():
#     from reward_kit.auth import get_fireworks_api_key # Relative import if this is a module
#     api_key = get_fireworks_api_key()
#     if not api_key:
#         print("API key not found for test.")
#         return

#     client_cfg = DictConfig({
#         "model_name": "accounts/fireworks/models/llama-v2-7b-chat",
#         "temperature": 0.1,
#         "max_tokens": 50,
#         "api_params": {"max_retries": 2}
#     })
#     client = FireworksModelClient(client_cfg, api_key)

#     async with aiohttp.ClientSession() as session:
#         response = await client.generate(
#             messages=[{"role": "user", "content": "What is 2+2?"}],
#             session=session
#         )
#         print(f"Test response: {response}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     # Need to load .env for get_fireworks_api_key if testing standalone
#     # from dotenv import load_dotenv
#     # load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env.dev"))
#     asyncio.run(main_test())
