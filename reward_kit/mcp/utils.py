"""
MCP Gym Utilities

This module provides shared utilities for MCP Gym classes to reduce code duplication
and standardize common patterns across GymProductionServer and McpGym.
"""

import hashlib
import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


class SessionIDGenerator:
    """Utility class for generating stable session IDs from MCP context."""
    
    @staticmethod
    def generate_session_id(ctx: Context) -> str:
        """
        Generate a stable session ID from MCP context.
        
        Uses client info (seed, config, name, version) to create a deterministic
        session ID that will be the same across reconnections with same parameters.
        
        Args:
            ctx: MCP context from FastMCP
            
        Returns:
            Stable session ID string
        """
        logger.debug(f"🔍 Generating session ID from context: {type(ctx)}")
        
        # Use stable session ID based on client info
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            logger.debug(f"🔍 Client params type: {type(client_params)}")
            
            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                logger.debug(f"🔍 Client info: {client_info}")
                
                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    logger.debug(f"🔍 Extra data: {extra_data}")
                    
                    if extra_data and isinstance(extra_data, dict):
                        # Create stable session ID based on seed and config
                        seed_value = extra_data.get("seed")
                        config_value = extra_data.get("config", {})
                        
                        stable_data = {
                            "seed": seed_value,
                            "config": config_value,
                            "name": client_info.name,
                            "version": client_info.version,
                        }
                        
                        logger.debug(f"🔍 Stable data for session ID: {stable_data}")
                        stable_str = json.dumps(stable_data, sort_keys=True)
                        session_id = hashlib.md5(stable_str.encode()).hexdigest()
                        logger.info(f"🎯 Generated stable session ID: {session_id[:16]}... for seed: {seed_value}")
                        return session_id
        
        # Fallback for testing or other scenarios
        session_id = f"gym_{id(ctx)}"
        logger.warning(f"🎯 Generated fallback session ID: {session_id}")
        return session_id

    @staticmethod
    def extract_seed_from_context(ctx: Context) -> Optional[int]:
        """
        Extract seed value from MCP client info.
        
        Args:
            ctx: MCP context from FastMCP
            
        Returns:
            Seed value if found, None otherwise
        """
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    if extra_data and isinstance(extra_data, dict):
                        seed = extra_data.get("seed")
                        if seed is not None:
                            logger.info(f"🌱 Extracted seed from context: {seed}")
                            return seed
        return None

    @staticmethod
    def extract_config_from_context(ctx: Context, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from MCP client info.
        
        Args:
            ctx: MCP context from FastMCP
            default_config: Default configuration to extend
            
        Returns:
            Configuration dictionary (default + client overrides)
        """
        config = default_config.copy()
        
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    if extra_data and isinstance(extra_data, dict):
                        client_config = extra_data.get("config", {})
                        if client_config:
                            config.update(client_config)
                            logger.debug(f"🔧 Updated config with client overrides: {config}")
        
        return config


class ControlPlaneState:
    """Utility class for managing control plane state."""
    
    @staticmethod
    def create_initial_state() -> Dict[str, Any]:
        """Create initial control plane state."""
        return {
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
            "step_count": 0,
            "total_reward": 0.0,
        }
    
    @staticmethod
    def update_state(
        state: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """
        Update control plane state after environment step.
        
        Args:
            state: Control plane state dictionary to update
            reward: Reward from environment step
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary from environment
        """
        state["reward"] = reward
        state["terminated"] = terminated
        state["truncated"] = truncated
        state["info"] = info
        state["step_count"] += 1
        state["total_reward"] += reward
    
    @staticmethod
    def log_update(
        session_id: str,
        reward: float,
        terminated: bool,
        step_count: int,
        prefix: str = "🎛️"
    ) -> None:
        """
        Log control plane state update.
        
        Args:
            session_id: Session identifier (will be truncated for display)
            reward: Current step reward
            terminated: Whether episode terminated
            step_count: Current step count
            prefix: Log message prefix emoji/text
        """
        session_short = session_id[:16] + "..." if len(session_id) > 16 else session_id
        logger.info(f"{prefix} Session {session_short} control plane: reward={reward}, terminated={terminated}, step={step_count}")


class SessionManager:
    """Utility class for managing session lifecycle and thread-safe operations."""
    
    def __init__(self):
        """Initialize session manager with thread-safe storage."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        with self.session_lock:
            return self.sessions.get(session_id)
    
    def create_session(
        self,
        session_id: str,
        env: Any,
        obs: Any,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create new session with environment and observation.
        
        Args:
            session_id: Session identifier
            env: Environment instance
            obs: Initial observation
            additional_data: Additional session-specific data
            
        Returns:
            Created session data dictionary
        """
        with self.session_lock:
            session_data = {
                "env": env,
                "obs": obs,
                "session_data": additional_data or {},
                "session_id": session_id,
            }
            self.sessions[session_id] = session_data
            
            logger.info(f"🎮 Created session {session_id[:16]}... with initial obs: {obs}")
            return session_data
    
    def update_session_obs(self, session_id: str, obs: Any) -> None:
        """
        Update session observation.
        
        Args:
            session_id: Session identifier
            obs: New observation
        """
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]["obs"] = obs
    
    def get_or_create_session_data(
        self,
        session_id: str,
        data_key: str,
        factory_func: callable
    ) -> Any:
        """
        Get or create session-specific data using factory function.
        
        Args:
            session_id: Session identifier
            data_key: Key within session_data to get/create
            factory_func: Function to create initial data if not exists
            
        Returns:
            Session data value
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            
            session_data = self.sessions[session_id]["session_data"]
            if data_key not in session_data:
                session_data[data_key] = factory_func()
            
            return session_data[data_key]
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        with self.session_lock:
            return session_id in self.sessions
    
    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        with self.session_lock:
            return len(self.sessions)


class EnvironmentUtils:
    """Utility class for environment creation and management."""
    
    @staticmethod
    def create_environment_with_seed(adapter, config: Dict[str, Any], seed: Optional[int]) -> Tuple[Any, Any, Dict]:
        """
        Create environment with optional seed, handling both adapter patterns.
        
        Args:
            adapter: Environment adapter instance
            config: Environment configuration
            seed: Optional seed for environment
            
        Returns:
            Tuple of (env, obs, info)
        """
        if hasattr(adapter, "create_environment_with_seed") and seed is not None:
            logger.debug(f"🏗️ Creating environment with seed using adapter method: seed={seed}")
            return adapter.create_environment_with_seed(config, seed=seed)
        else:
            logger.debug(f"🏗️ Creating environment and resetting with seed: seed={seed}")
            env = adapter.create_environment(config)
            obs, info = adapter.reset_environment(env, seed=seed)
            return env, obs, info
    
    @staticmethod
    def log_environment_creation(session_id: str, seed: Optional[int], obs: Any) -> None:
        """
        Log environment creation for debugging.
        
        Args:
            session_id: Session identifier
            seed: Seed used for environment (if any)
            obs: Initial observation
        """
        session_short = session_id[:16] + "..." if len(session_id) > 16 else session_id
        logger.info(f"🎮 Session {session_short} environment created with seed {seed}, initial obs: {obs}")
