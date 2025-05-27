"""
Caching for model-generated responses.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, cache_config: DictConfig):
        self.cache_config = cache_config
        self.cache_dir = cache_config.get(
            "cache_dir", ".reward_kit_cache/generated_responses"
        )
        # Ensure cache_dir is absolute or resolvable from a known root
        # For now, assume it's relative to CWD or an absolute path.
        # If used within a Hydra app, CWD is the output dir, so this might need adjustment
        # or be made relative to hydra.utils.get_original_cwd().
        if not os.path.isabs(self.cache_dir):
            # Defaulting to make it under CWD/.reward_kit_cache if not absolute
            # This might need to be configurable to be relative to project root.
            self.cache_dir = os.path.join(os.getcwd(), self.cache_dir)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Response cache directory: {self.cache_dir}")
        except OSError as e:
            logger.error(
                f"Failed to create cache directory {self.cache_dir}: {e}. Caching will be disabled."
            )
            self.cache_dir = None  # Disable caching if dir creation fails

    def _generate_key(
        self,
        sample_id: str,
        system_prompt: Optional[str],
        user_query: str,  # Or full messages list for more robustness
        model_name: str,
        temperature: float,
        # Potentially add other generation params like max_tokens, top_p if they affect output
    ) -> str:
        """Generates a cache key."""
        key_material = (
            f"{sample_id}-{system_prompt}-{user_query}-{model_name}-{temperature}"
        )
        return hashlib.md5(key_material.encode()).hexdigest()

    def get(
        self,
        sample_id: str,
        system_prompt: Optional[str],
        user_query: str,
        model_name: str,
        temperature: float,
    ) -> Optional[str]:
        """Retrieves an item from the cache. Returns None if not found or error."""
        if not self.cache_dir:
            return None

        if (
            temperature != 0.0
        ):  # Only cache deterministic (temp=0) generations by default
            # Could add a config option to allow caching for non-zero temps if desired
            return None

        cache_key = self._generate_key(
            sample_id, system_prompt, user_query, model_name, temperature
        )
        cache_file_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    # Assuming cached data is stored like {"assistant_response": "..."}
                    response = cached_data.get("assistant_response")
                    if response is not None:
                        logger.debug(
                            f"Cache hit for key {cache_key} (sample {sample_id})"
                        )
                        return response
                    else:
                        logger.warning(
                            f"Cache file {cache_file_path} for key {cache_key} is malformed."
                        )
            except json.JSONDecodeError:
                logger.warning(
                    f"Error decoding JSON from cache file {cache_file_path} for key {cache_key}."
                )
            except Exception as e:
                logger.warning(f"Error reading from cache file {cache_file_path}: {e}")
        else:
            logger.debug(f"Cache miss for key {cache_key} (sample {sample_id})")
        return None

    def put(
        self,
        sample_id: str,
        system_prompt: Optional[str],
        user_query: str,
        model_name: str,
        temperature: float,
        response: str,
    ) -> None:
        """Stores an item in the cache."""
        if not self.cache_dir:
            return

        if temperature != 0.0:  # Only cache deterministic (temp=0) generations
            return

        cache_key = self._generate_key(
            sample_id, system_prompt, user_query, model_name, temperature
        )
        cache_file_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            # Store as {"assistant_response": "..."} to match example main.py
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump({"assistant_response": response}, f)
            logger.debug(f"Cached response for key {cache_key} (sample {sample_id})")
        except Exception as e:
            logger.warning(f"Error writing to cache file {cache_file_path}: {e}")
