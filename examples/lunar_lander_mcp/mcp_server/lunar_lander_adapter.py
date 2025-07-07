#!/usr/bin/env python3
"""
LunarLander Adapter for Gymnasium Environment

This adapter handles the specific mechanics of the LunarLander environment
including discrete action space, rendering, and state interpretation.
"""

import base64
import io
import tempfile
from typing import Any, Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
from PIL import Image


class LunarLanderAdapter:
    """Adapter for LunarLander environment with rendering support."""

    def __init__(self):
        self.action_map = {
            "NOTHING": 0,
            "LEFT": 1,
            "MAIN": 2,
            "RIGHT": 3,
            "FIRE_LEFT": 1,
            "FIRE_MAIN": 2,
            "FIRE_RIGHT": 3,
        }
        self.action_names = ["NOTHING", "FIRE_LEFT", "FIRE_MAIN", "FIRE_RIGHT"]

    def get_default_config(self) -> str:
        """Get the default environment configuration."""
        return "LunarLander-v3"

    def create_environment(self, env_id: str) -> gym.Env:
        """Create and configure the LunarLander environment."""
        env = gym.make(env_id, render_mode="rgb_array")
        return env

    def parse_action(self, action: str) -> int:
        """Parse string action to environment action integer."""
        action = action.strip().upper()
        if action not in self.action_map:
            valid_actions = list(self.action_map.keys())
            raise ValueError(
                f"Invalid action '{action}'. Valid actions: {valid_actions}"
            )
        return self.action_map[action]

    def step_environment(
        self, env: gym.Env, action: int
    ) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        return env.step(action)

    def reset_environment(self, env: gym.Env, seed: int = None) -> Tuple[Any, Dict]:
        """Reset the environment."""
        if seed is not None:
            return env.reset(seed=seed)
        return env.reset()

    def render_frame(self, env: gym.Env) -> str:
        """Render the current environment state as base64 encoded image."""
        try:
            # Get RGB array from environment
            rgb_array = env.render()

            if rgb_array is None:
                return None

            # Convert numpy array to PIL Image
            image = Image.fromarray(rgb_array.astype(np.uint8))

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode as base64 string
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return f"data:image/png;base64,{image_b64}"

        except Exception as e:
            print(f"Error rendering frame: {e}")
            return None

    def format_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """Format observation array into structured data."""
        # LunarLander observation is 8-dimensional:
        # [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]

        if len(obs) != 8:
            raise ValueError(f"Expected 8-dimensional observation, got {len(obs)}")

        return {
            "position": {"x": float(obs[0]), "y": float(obs[1])},
            "velocity": {"x": float(obs[2]), "y": float(obs[3])},
            "orientation": {"angle": float(obs[4]), "angular_velocity": float(obs[5])},
            "legs": {"left_contact": bool(obs[6]), "right_contact": bool(obs[7])},
            "raw_observation": obs.tolist(),
        }

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        return {
            "type": "discrete",
            "n": 4,
            "actions": {
                0: "NOTHING - Do nothing",
                1: "FIRE_LEFT - Fire left orientation engine",
                2: "FIRE_MAIN - Fire main engine",
                3: "FIRE_RIGHT - Fire right orientation engine",
            },
        }

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get information about the observation space."""
        return {
            "type": "box",
            "shape": [8],
            "description": {
                "0": "x coordinate",
                "1": "y coordinate",
                "2": "x velocity",
                "3": "y velocity",
                "4": "angle",
                "5": "angular velocity",
                "6": "left leg contact",
                "7": "right leg contact",
            },
        }

    def is_successful_landing(
        self, obs: np.ndarray, reward: float, terminated: bool
    ) -> bool:
        """Check if the landing was successful."""
        if not terminated:
            return False

        # Successful landing typically gives positive reward
        # and both legs should be in contact
        legs_contact = obs[6] and obs[7] if len(obs) >= 8 else False

        return reward > 0 and legs_contact

    def get_landing_status(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> str:
        """Get a human-readable status of the landing attempt."""
        if not (terminated or truncated):
            return "🚀 Flying"

        if truncated:
            return "⏰ Time limit reached"

        if self.is_successful_landing(obs, reward, terminated):
            return "🎯 Successful landing!"
        elif reward < -50:  # Crashed
            return "💥 Crashed!"
        else:
            return "❌ Landing failed"
