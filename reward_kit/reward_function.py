from typing import Dict, List, Optional, Any, Union, Callable, Type, TypeVar, cast
import os
import importlib
import importlib.util
import inspect
import json
import requests
from pathlib import Path
from functools import wraps
import logging

from .models import RewardOutput, MetricRewardOutput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type for reward function
T = TypeVar('T', bound=Callable[..., RewardOutput])

class RewardFunction:
    """
    A wrapper for reward functions that allows them to be run locally or remotely.
    
    The RewardFunction class wraps a reward function (either a local function or a remote endpoint)
    and provides a unified interface for calling it. It supports:
    
    - Local functions (mode="local")
    - Remote endpoints (mode="remote")
    - Fireworks-hosted models (mode="fireworks_hosted")
    
    Args:
        func: The local function to use (for mode="local")
        func_path: A string path to a function (e.g., "module.submodule:function_name")
        mode: The mode of operation ("local", "remote", or "fireworks_hosted")
        endpoint: The URL of the remote endpoint (for mode="remote")
        model_id: The ID of the Fireworks-hosted model (for mode="fireworks_hosted")
        **kwargs: Additional keyword arguments to pass to the function
    """
    
    def __init__(
        self,
        func: Optional[Callable] = None,
        func_path: Optional[str] = None,
        mode: str = "local",
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ):
        self.mode = mode
        self.func = func
        self.func_path = func_path
        self.endpoint = endpoint
        self.name = name
        self.model_id = model_id
        self.kwargs = kwargs
        
        if mode == "local":
            if func is None and func_path is None:
                raise ValueError("Either 'func' or 'func_path' must be provided for local mode")
            if func_path and func is None:
                self.func = self._load_function_from_path(func_path)
        elif mode == "remote":
            if endpoint is None and name is None:
                raise ValueError("Either 'endpoint' or 'name' must be provided for remote mode")
            if name and endpoint is None:
                # Construct endpoint URL from name (in a real implementation, 
                # this would fetch from the Fireworks API)
                self.endpoint = f"https://api.fireworks.ai/v1/reward/{name}"
        elif mode == "fireworks_hosted":
            if model_id is None:
                raise ValueError("'model_id' must be provided for fireworks_hosted mode")
            # Construct endpoint for the Fireworks-hosted model
            self.endpoint = f"https://api.fireworks.ai/v1/models/{model_id}/reward"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _load_function_from_path(self, func_path: str) -> Callable:
        """Load a function from a path string (e.g., 'module.submodule:function_name')."""
        if ":" not in func_path:
            raise ValueError(f"Invalid func_path format: {func_path}, expected 'module.path:function_name'")
        
        module_path, func_name = func_path.split(":", 1)
        
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load function from path {func_path}: {str(e)}")
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        original_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> RewardOutput:
        """
        Call the reward function with the provided messages.
        
        Args:
            messages: List of conversation messages, each with 'role' and 'content' keys
            original_messages: Original conversation messages (for context)
            **kwargs: Additional keyword arguments to pass to the function
                
        Returns:
            RewardOutput object with score and metrics
        """
        if original_messages is None:
            original_messages = messages[:-1] if messages else []
        
        # Combine instance kwargs with call kwargs
        combined_kwargs = {**self.kwargs, **kwargs}
        
        if self.mode == "local":
            if self.func is None:
                raise ValueError("No function provided for local mode")
            
            # Call the local function
            try:
                result = self.func(messages=messages, original_messages=original_messages, **combined_kwargs)
                
                # Ensure the result is a RewardOutput
                if isinstance(result, RewardOutput):
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handle legacy (score, components) tuple format
                    score, components = result
                    metrics = {
                        k: MetricRewardOutput(score=v, reason=None)
                        for k, v in components.items()
                    }
                    return RewardOutput(score=score, metrics=metrics)
                else:
                    raise TypeError(f"Invalid return type from reward function: {type(result)}")
                
            except Exception as e:
                logger.error(f"Error calling local reward function: {str(e)}")
                raise
        
        elif self.mode in ["remote", "fireworks_hosted"]:
            if self.endpoint is None:
                raise ValueError(f"No endpoint provided for {self.mode} mode")
            
            # Prepare the payload
            payload = {
                "messages": messages,
                "original_messages": original_messages,
                **combined_kwargs
            }
            
            # Get API key
            api_key = os.environ.get("FIREWORKS_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
            
            try:
                response = requests.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # Convert the result to RewardOutput
                if isinstance(result, dict) and "score" in result:
                    return RewardOutput.from_dict(result)
                else:
                    raise ValueError(f"Invalid response from remote endpoint: {result}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling remote endpoint: {str(e)}")
                raise
        
        raise ValueError(f"Invalid mode: {self.mode}")
    
    def get_trl_adapter(self) -> Callable:
        """
        Create an adapter function for use with TRL library.
        
        The TRL library expects a function that takes a batch of texts (list of strings)
        and returns a batch of reward values (list of floats).
        
        Returns:
            A callable function compatible with TRL
        """
        def adapter(texts: List[str]) -> List[float]:
            results = []
            for text in texts:
                # TRL typically provides just the completion, so we wrap it in a message
                messages = [{"role": "assistant", "content": text}]
                # Call the reward function and extract the score
                try:
                    reward_output = self(messages=messages)
                    results.append(reward_output.score)
                except Exception as e:
                    logger.error(f"Error in TRL adapter: {str(e)}")
                    # In case of error, provide a neutral reward to avoid breaking training
                    results.append(0.0)
            return results
        
        return adapter


def reward_function(func: T) -> T:
    """
    Decorator for reward functions that adds deployment capabilities.
    
    This decorator wraps a function to ensure it returns a RewardOutput and adds
    a .deploy() method that can be used to deploy the function to Fireworks.
    
    Args:
        func: The reward function to decorate
        
    Returns:
        The decorated function with added deployment capabilities
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> RewardOutput:
        result = func(*args, **kwargs)
        
        # Ensure the result is a RewardOutput
        if isinstance(result, RewardOutput):
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Handle legacy (score, components) tuple format
            score, components = result
            metrics = {
                k: MetricRewardOutput(score=v, reason=None)
                for k, v in components.items()
            }
            return RewardOutput(score=score, metrics=metrics)
        else:
            raise TypeError(
                f"Invalid return type from reward function: {type(result)}. "
                f"Expected RewardOutput or (float, Dict[str, float]) tuple."
            )
    
    def deploy(**config) -> str:
        """
        Deploy the reward function to Fireworks.
        
        Args:
            **config: Configuration options for deployment
                
        Returns:
            A string deployment handle/ID
        """
        # In a real implementation, this would send the function code to Fireworks
        # for deployment. For now, we just log the action.
        name = config.get("name", func.__name__)
        logger.info(f"Deploying reward function '{func.__name__}' as '{name}'...")
        
        # Get function source code and dependencies
        source = inspect.getsource(func)
        module = inspect.getmodule(func)
        module_path = module.__file__ if module else None
        
        # Create a mock deployment ID
        deployment_id = f"deployment_{name}"
        
        logger.info(f"Deployment submitted. Handle/ID: {deployment_id}")
        return deployment_id
    
    # Add the deploy method to the function
    wrapper.deploy = deploy  # type: ignore
    
    return cast(T, wrapper)