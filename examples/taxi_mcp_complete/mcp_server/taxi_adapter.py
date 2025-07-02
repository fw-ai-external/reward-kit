"""
Taxi Environment Adapter

Adapter for the Taxi-v3 gymnasium environment for MCP framework integration.
The Taxi environment involves navigating to passengers, picking them up, and
dropping them off at designated locations in a 5x5 grid world.
"""

from typing import Any, Dict, Optional, Tuple

from gymnasium.envs.toy_text.taxi import TaxiEnv

from reward_kit.mcp import EnvironmentAdapter


class TaxiAdapter(EnvironmentAdapter):
    """Taxi environment adapter for MCP framework."""

    ACTION_NAMES = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]

    def decode_state(self, state: int) -> Dict[str, Any]:
        """
        Decode the Taxi state integer into human-readable components.

        State encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

        Returns:
            Dict with taxi_row, taxi_col, passenger_location, destination
        """
        # Reverse the encoding formula
        destination = state % 4
        state //= 4

        passenger_location = state % 5
        state //= 5

        taxi_col = state % 5
        taxi_row = state // 5

        return {
            "taxi_row": taxi_row,
            "taxi_col": taxi_col,
            "passenger_location": passenger_location,  # 0-3: locations, 4: in taxi
            "destination": destination,  # 0-3: Red, Green, Yellow, Blue
        }

    def get_state_description(self, state: int) -> str:
        """Get human-readable description of the current state."""
        decoded = self.decode_state(state)

        locations = ["Red", "Green", "Yellow", "Blue"]
        location_symbols = ["R", "G", "Y", "B"]
        destination_symbols = ["r", "g", "y", "b"]

        # Determine taxi visual symbol
        taxi_symbol = "T" if decoded["passenger_location"] == 4 else "t"

        # Passenger description
        if decoded["passenger_location"] == 4:
            passenger_desc = "in taxi"
        else:
            passenger_location_name = locations[decoded["passenger_location"]]
            passenger_symbol = location_symbols[decoded["passenger_location"]]
            passenger_desc = f"at {passenger_symbol} ({passenger_location_name})"

        # Destination description
        destination_name = locations[decoded["destination"]]
        destination_symbol = destination_symbols[decoded["destination"]]

        # Add action guidance
        if decoded["passenger_location"] == 4:
            # Passenger in taxi - need to dropoff
            action_guidance = (
                f"must dropoff passenger at {destination_symbol} ({destination_name})"
            )
        else:
            # Passenger not in taxi - need to pickup
            action_guidance = "must pickup passenger"

        return (
            f"Taxi at {taxi_symbol} ({decoded['taxi_row']}, {decoded['taxi_col']}), "
            f"Passenger {passenger_desc}, "
            f"Destination: {destination_symbol} ({destination_name}), "
            f"{action_guidance}"
        )

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> TaxiEnv:
        """
        Create Taxi environment.

        Args:
            config: Optional configuration dict. Can include:
                - is_raining: If True, movement has 80% success rate (default: False)
                - fickle_passenger: If True, passenger may change destinations (default: False)
        """
        is_raining = False
        fickle_passenger = False

        # Extract config options
        if config:
            is_raining = config["is_raining"]
            fickle_passenger = config["fickle_passenger"]

        # Create environment (TaxiEnv doesn't accept these parameters directly)
        # TODO: The parameters would need to be handled differently in gymnasium
        env = TaxiEnv(is_rainy=is_raining, fickle_passenger=fickle_passenger)

        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[TaxiEnv, int, Dict[str, Any]]:
        """
        Create Taxi environment with proper seeding and return initial state.

        Returns:
            Tuple of (environment, initial_observation, initial_info)
        """
        # Create environment
        env = self.create_environment(config)

        # Reset with seed to get initial state
        obs, info = env.reset(seed=seed)

        return env, obs, info

    def reset_environment(
        self, env: TaxiEnv, seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset environment to initial state."""
        return env.reset(seed=seed)

    def step_environment(
        self, env: TaxiEnv, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute environment step."""
        return env.step(action)

    def close_environment(self, env: TaxiEnv) -> None:
        """Close environment."""
        pass  # TaxiEnv doesn't need explicit cleanup

    def parse_action(self, action_str: str) -> int:
        """Parse action string to integer."""
        action_str = action_str.upper().strip()
        if action_str not in self.ACTION_NAMES:
            raise ValueError(
                f"Invalid action '{action_str}'. Must be one of {self.ACTION_NAMES}"
            )
        return self.ACTION_NAMES.index(action_str)

    def format_observation(self, observation: int) -> int:
        """Format observation for JSON."""
        return int(observation)

    def get_action_space_description(self) -> Dict[str, Any]:
        """Get action space description."""
        return {
            "type": "discrete",
            "num_actions": 6,
            "actions": self.ACTION_NAMES,
            "description": "Move actions: SOUTH(0), NORTH(1), EAST(2), WEST(3), PICKUP(4), DROPOFF(5)",
        }

    def get_observation_space_description(self) -> Dict[str, Any]:
        """Get observation space description."""
        return {
            "type": "discrete",
            "num_states": 500,
            "description": "Encoded state: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination",
            "components": {
                "taxi_position": "25 possible positions in 5x5 grid",
                "passenger_location": "5 possible locations (0-3: Red/Green/Yellow/Blue, 4: in taxi)",
                "destination": "4 possible destinations (0-3: Red/Green/Yellow/Blue)",
            },
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Taxi."""
        return {
            "is_raining": False,  # Deterministic movement
            "fickle_passenger": False,  # Passenger doesn't change destinations
        }

    # def get_game_info(self) -> Dict[str, Any]:
    #     """Get general information about the Taxi game."""
    #     return {
    #         "name": "Taxi-v3",
    #         "description": "Navigate to passengers, pick them up, and drop them off at designated locations",
    #         "grid_size": "5x5",
    #         "locations": {
    #             "Red": "Top-left area",
    #             "Green": "Top-right area",
    #             "Yellow": "Bottom-left area",
    #             "Blue": "Bottom-right area"
    #         },
    #         "rewards": {
    #             "step": -1,
    #             "successful_dropoff": +20,
    #             "illegal_pickup_dropoff": -10
    #         },
    #         "episode_length": "Max 200 steps (with time limit wrapper)"
    #     }
