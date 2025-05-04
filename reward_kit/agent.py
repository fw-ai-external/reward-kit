"""
Agent evaluation framework for Reward Kit.

This module provides tools for evaluating agent models that use tool-augmented reasoning.
It implements a Task Bundle architecture where reward functions and tools live side-by-side.
"""

import os
import importlib
import json
import inspect
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Type, TypeVar
from enum import Enum
from dataclasses import dataclass, field

import fastapi
from fastapi import FastAPI, APIRouter, Request, Body, HTTPException
from pydantic import BaseModel, create_model, Field
import sqlalchemy
from sqlalchemy import create_engine, text
import aiosqlite

# Type definitions
ToolFunc = TypeVar('ToolFunc', bound=Callable)


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Definition of a tool for agent evaluation."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    func: Optional[Callable] = None
    is_async: bool = False


class ToolRegistry:
    """
    Registry for agent tools.
    
    The ToolRegistry class manages a collection of tool functions for agent evaluation.
    It provides decorators for registering tools and methods for generating OpenAI-compatible
    tool specifications.
    
    Args:
        name: A name for this tool registry
        description: Optional description of this tool collection
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Tool registry for {name}"
        self.tools: Dict[str, ToolDefinition] = {}
        self._app: Optional[FastAPI] = None
    
    def tool(self, 
             description: str, 
             parameters: Dict[str, Type] = None,
             name: Optional[str] = None) -> Callable[[ToolFunc], ToolFunc]:
        """
        Decorator to register a function as a tool.
        
        Args:
            description: Description of what the tool does
            parameters: Dictionary mapping parameter names to types
            name: Optional name for the tool (defaults to function name)
            
        Returns:
            The decorated function
        """
        def decorator(func: ToolFunc) -> ToolFunc:
            func_name = name or func.__name__
            
            # Determine if function is async
            is_async = asyncio.iscoroutinefunction(func)
            
            # Process parameters
            params = {}
            if parameters:
                for param_name, param_type in parameters.items():
                    # Convert Python types to JSON schema types
                    if param_type == str:
                        type_str = "string"
                    elif param_type == int:
                        type_str = "integer"
                    elif param_type == float:
                        type_str = "number"
                    elif param_type == bool:
                        type_str = "boolean"
                    elif param_type == list:
                        type_str = "array"
                    elif param_type == dict:
                        type_str = "object"
                    else:
                        type_str = "string"  # Default to string for complex types
                    
                    params[param_name] = {
                        "type": type_str,
                        "description": f"{param_name} parameter"
                    }
            
            # Get additional parameter info from function signature and docstring
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                # Skip 'self', 'cls', and internal parameters like 'db'
                if param_name in ('self', 'cls', 'db'):
                    continue
                
                if param_name not in params and param.default == inspect.Parameter.empty:
                    # Add missing required parameters
                    params[param_name] = {
                        "type": "string",  # Default type
                        "description": f"{param_name} parameter"
                    }
            
            # Store the tool
            self.tools[func_name] = ToolDefinition(
                name=func_name,
                description=description,
                parameters=params,
                func=func,
                is_async=is_async
            )
            
            return func
        
        return decorator
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        openai_tools = []
        
        for tool_name, tool in self.tools.items():
            properties = {}
            required = []
            
            for param_name, param_info in tool.parameters.items():
                properties[param_name] = {
                    "type": param_info["type"],
                    "description": param_info.get("description", f"{param_name} parameter")
                }
                
                if param_info.get("enum"):
                    properties[param_name]["enum"] = param_info["enum"]
                
                # Add to required list unless it has a default value
                if param_info.get("required", True):
                    required.append(param_name)
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        return openai_tools
    
    def create_fastapi_app(self) -> FastAPI:
        """
        Create a FastAPI application for the tools.
        
        Returns:
            FastAPI application
        """
        if self._app:
            return self._app
        
        app = FastAPI(title=self.name, description=self.description)
        router = APIRouter()
        
        for tool_name, tool in self.tools.items():
            # Create a Pydantic model for request validation
            fields = {}
            for param_name, param_info in tool.parameters.items():
                param_type = param_info["type"]
                
                # Map JSON schema types to Python types
                if param_type == "string":
                    python_type = str
                elif param_type == "integer":
                    python_type = int
                elif param_type == "number":
                    python_type = float
                elif param_type == "boolean":
                    python_type = bool
                elif param_type == "array":
                    python_type = list
                elif param_type == "object":
                    python_type = dict
                else:
                    python_type = str  # Default
                
                # Define the field with proper typing
                default = param_info.get("default", ... if param_info.get("required", True) else None)
                fields[param_name] = (python_type, Field(default=default, description=param_info.get("description")))
            
            # Create dynamic model
            model_name = f"{tool_name.capitalize()}Request"
            request_model = create_model(model_name, **fields)
            
            # Define the endpoint
            @router.post(f"/tools/{tool_name}", summary=tool.description)
            async def tool_endpoint(request_data: request_model, tool=tool):
                # Convert Pydantic model to dict
                params = request_data.dict()
                
                try:
                    # Add db connection if available in request context
                    # This will be implemented later with dependency injection
                    
                    # Execute the tool function (handle both async and sync)
                    if tool.is_async:
                        result = await tool.func(**params)
                    else:
                        result = tool.func(**params)
                    
                    return {"result": result}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")
        
        app.include_router(router)
        self._app = app
        return app
    
    async def execute_tool(self, 
                          tool_name: str, 
                          params: Dict[str, Any],
                          db_conn: Optional[Any] = None) -> Any:
        """
        Execute a tool by name with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            db_conn: Optional database connection to inject
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool doesn't exist
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        tool = self.tools[tool_name]
        tool_params = params.copy()
        
        # Inject database connection if provided and the function accepts it
        sig = inspect.signature(tool.func)
        if "db" in sig.parameters and db_conn is not None:
            tool_params["db"] = db_conn
        
        # Execute the function (sync or async)
        if tool.is_async:
            return await tool.func(**tool_params)
        else:
            return tool.func(**tool_params)


class Database:
    """
    Database manager for agent evaluation.
    
    Provides methods to set up, access, and modify a SQLite database
    for agent evaluation tasks.
    """
    
    def __init__(self, 
                base_path: str, 
                seed_sql: Optional[str] = None,
                seed_file: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            base_path: Path to the database file
            seed_sql: Optional SQL to initialize the database
            seed_file: Optional path to a SQL file to initialize the database
        """
        self.base_path = base_path
        self.seed_sql = seed_sql
        self.seed_file = seed_file
        self._engine = None
        self._connection = None
    
    async def setup(self):
        """Set up the database with the seed data."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
        
        # Create new database file
        async with aiosqlite.connect(self.base_path) as db:
            if self.seed_sql:
                await db.executescript(self.seed_sql)
            elif self.seed_file:
                with open(self.seed_file, 'r') as f:
                    seed_sql = f.read()
                await db.executescript(seed_sql)
            
            await db.commit()
    
    async def get_connection(self):
        """Get a database connection."""
        if not os.path.exists(self.base_path):
            await self.setup()
        
        # Create a fresh connection each time with timeout
        conn = await asyncio.wait_for(
            aiosqlite.connect(self.base_path),
            timeout=5.0
        )
        
        # Enable foreign keys and set busy timeout
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA busy_timeout = 5000")
        
        # Add helper methods directly to the connection object
        # to make it compatible with our tools
        
        # Add fetch_all method
        async def fetch_all(query, params=None):
            try:
                if params:
                    cursor = await conn.execute(query, params)
                else:
                    cursor = await conn.execute(query)
                
                # Convert rows to dictionaries
                columns = [col[0] for col in cursor.description]
                rows = await cursor.fetchall()
                
                # Use list comprehension for performance
                return [dict(zip(columns, row)) for row in rows]
            except asyncio.TimeoutError:
                print(f"Query timed out: {query}")
                raise asyncio.TimeoutError(f"Database query timed out: {query}")
        
        # Add fetch_one method
        async def fetch_one(query, params=None):
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        
        # Add fetch_val method
        async def fetch_val(query, params=None):
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            
            row = await cursor.fetchone()
            return row[0] if row else None
        
        # Attach methods to the connection
        conn.fetch_all = fetch_all
        conn.fetch_one = fetch_one
        conn.fetch_val = fetch_val
        
        return conn
    
    def get_sync_engine(self):
        """Get a SQLAlchemy engine for synchronous access."""
        if not self._engine:
            self._engine = create_engine(f"sqlite:///{self.base_path}")
        return self._engine
    
    def get_sync_connection(self):
        """Get a SQLAlchemy connection for synchronous access."""
        if not self._connection:
            self._connection = self.get_sync_engine().connect()
        return self._connection
    
    async def create_snapshot(self, snapshot_path: str):
        """
        Create a snapshot of the database at the given path.
        
        Args:
            snapshot_path: Path to save the snapshot
        """
        import shutil
        shutil.copy2(self.base_path, snapshot_path)
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            Query result
        """
        async with await self.get_connection() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            
            result = await cursor.fetchall()
            await conn.commit()
            return result
    
    async def execute_script(self, script: str):
        """
        Execute a SQL script.
        
        Args:
            script: SQL script to execute
        """
        async with await self.get_connection() as conn:
            await conn.executescript(script)
            await conn.commit()
    
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Fetch all results from a query.
        
        Args:
            query: SQL query
            params: Optional parameters
            
        Returns:
            List of rows
        """
        async with await self.get_connection() as conn:
            try:
                # Set timeout and optimize query
                await conn.execute("PRAGMA busy_timeout = 5000")
                
                # Execute query with timeout
                if params:
                    cursor = await asyncio.wait_for(
                        conn.execute(query, params),
                        timeout=5.0
                    )
                else:
                    cursor = await asyncio.wait_for(
                        conn.execute(query),
                        timeout=5.0
                    )
                
                # Convert rows to dictionaries
                columns = [col[0] for col in cursor.description]
                rows = await cursor.fetchall()
                
                # Use list comprehension for performance
                return [dict(zip(columns, row)) for row in rows]
            except asyncio.TimeoutError:
                # Log and re-raise with more context
                print(f"Query timed out: {query}")
                raise asyncio.TimeoutError(f"Database query timed out: {query}")
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Fetch one result from a query.
        
        Args:
            query: SQL query
            params: Optional parameters
            
        Returns:
            Row as dictionary or None
        """
        async with await self.get_connection() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
    
    async def fetch_val(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Fetch a single value from a query.
        
        Args:
            query: SQL query
            params: Optional parameters
            
        Returns:
            Single value or None
        """
        async with await self.get_connection() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            
            row = await cursor.fetchone()
            return row[0] if row else None
    
    async def close(self):
        """Close database connections."""
        if self._connection:
            self._connection.close()
            self._connection = None
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
            
        # Force garbage collection to release file handles
        import gc
        gc.collect()


class AgentEvaluator:
    """
    Orchestrator for agent evaluation.
    
    Manages the interaction between an agent model, tools, and a reward function.
    """
    
    def __init__(self, 
                 task_id: str,
                 toolset_path: str,
                 reward_path: str,
                 base_dir: str = "./runs",
                 seed_sql: Optional[str] = None,
                 seed_file: Optional[str] = None):
        """
        Initialize the agent evaluator.
        
        Args:
            task_id: Unique identifier for this task
            toolset_path: Import path to the toolset (e.g., "my_task.tools")
            reward_path: Import path to the reward function (e.g., "my_task.reward")
            base_dir: Base directory for evaluation runs
            seed_sql: Optional SQL to initialize the database
            seed_file: Optional path to a SQL file to initialize the database
        """
        self.task_id = task_id
        self.toolset_path = toolset_path
        self.reward_path = reward_path
        self.base_dir = base_dir
        self.seed_sql = seed_sql
        self.seed_file = seed_file
        
        # Paths for database files
        task_dir = os.path.join(base_dir, task_id)
        self.db_path = os.path.join(task_dir, "base.db")
        
        # Imported components (to be loaded)
        self.tool_registry = None
        self.reward_function = None
    
    async def setup(self):
        """Set up the evaluation environment."""
        # Create directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Import toolset
        try:
            tool_module = importlib.import_module(self.toolset_path)
            self.tool_registry = tool_module.R  # Convention: registry is named R
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import tool registry from {self.toolset_path}: {str(e)}")
        
        # Import reward function
        try:
            reward_module = importlib.import_module(self.reward_path)
            self.reward_function = reward_module.evaluate  # Convention: function is named evaluate
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import reward function from {self.reward_path}: {str(e)}")
        
        # Set up database
        self.db = Database(self.db_path, self.seed_sql, self.seed_file)
        await self.db.setup()
    
    async def create_run(self, run_id: str) -> str:
        """
        Create a new evaluation run.
        
        Args:
            run_id: Unique identifier for this run
            
        Returns:
            Path to the run database
        """
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        
        # Create a snapshot of the base database
        await self.db.create_snapshot(run_db_path)
        
        return run_db_path
    
    async def execute_tool(self, 
                         run_id: str, 
                         tool_name: str, 
                         params: Dict[str, Any]) -> Any:
        """
        Execute a tool in a specific run.
        
        Args:
            run_id: Run identifier
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        # Get the run database
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        
        if not os.path.exists(run_db_path):
            raise ValueError(f"Run {run_id} does not exist")
        
        # Create database connection for this run
        run_db = Database(run_db_path)
        db_conn = await run_db.get_connection()
        
        try:
            # Execute the tool with the database connection
            result = await self.tool_registry.execute_tool(tool_name, params, db_conn)
            return result
        finally:
            await db_conn.close()
    
    async def evaluate(self, 
                     run_id: str, 
                     messages: List[Dict[str, Any]],
                     **kwargs) -> Dict[str, Any]:
        """
        Evaluate a conversation using the reward function.
        
        Args:
            run_id: Run identifier
            messages: Conversation messages
            **kwargs: Additional parameters for the reward function
            
        Returns:
            Evaluation result
        """
        # Get the run database
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        
        if not os.path.exists(run_db_path):
            raise ValueError(f"Run {run_id} does not exist")
        
        # Create database connection for this run
        run_db = Database(run_db_path)
        db_conn = run_db.get_sync_connection()
        
        try:
            # Add database connection to kwargs
            eval_kwargs = kwargs.copy()
            eval_kwargs["db"] = db_conn
            
            # Call the reward function
            result = self.reward_function(messages=messages, **eval_kwargs)
            
            # Convert to dict if it's not already
            if hasattr(result, "to_dict"):
                return result.to_dict()
            elif hasattr(result, "__dict__"):
                return result.__dict__
            else:
                return {"score": result}
        finally:
            db_conn.close()


# Helper functions for loading task definitions
def load_task_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load task definitions from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of task definitions
    """
    tasks = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            task = json.loads(line)
            tasks.append(task)
    
    return tasks


def load_sql_from_file(file_path: str) -> str:
    """
    Load SQL from a file.
    
    Args:
        file_path: Path to the SQL file
        
    Returns:
        SQL string
    """
    with open(file_path, "r") as f:
        return f.read()