"""
Agent evaluation framework for Reward Kit (V1 Components).

This module provides tools for evaluating agent models that use tool-augmented reasoning.
It implements a Task Bundle architecture where reward functions and tools live side-by-side.
The V2 components (ForkableResource, Orchestrator, etc.) have been moved to the
`reward_kit.agent_v2` package.
"""

import os
import importlib
import json
import inspect
import asyncio
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Callable,
    Type,
    TypeVar,
)

from pydantic import BaseModel, Field # create_model is used by ToolRegistry
# FastAPI is forward-referenced in ToolRegistry, actual import is local in create_fastapi_app
# from fastapi import FastAPI, APIRouter, HTTPException 

from sqlalchemy import create_engine # Used by V1 Database
import aiosqlite # Used by V1 Database

# Type definitions
ToolFunc = TypeVar("ToolFunc", bound=Callable[..., Any])


# --- V1 Agent Framework Components ---

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
    func: Optional[Callable[..., Any]] = None
    is_async: bool = False


class ToolRegistry:
    """
    Registry for agent tools (V1).
    """
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Tool registry for {name}"
        self.tools: Dict[str, ToolDefinition] = {}
        self._app: Optional['FastAPI'] = None # Forward ref FastAPI, type as string

    def tool(
        self,
        description: str,
        parameters: Optional[Dict[str, Type[Any]]] = None,
        name: Optional[str] = None,
    ) -> Callable[[ToolFunc], ToolFunc]:
        def decorator(func: ToolFunc) -> ToolFunc:
            func_name = name or func.__name__
            is_async = asyncio.iscoroutinefunction(func)
            processed_params: Dict[str, Dict[str, Any]] = {}
            if parameters:
                for param_name, param_type in parameters.items():
                    type_str: str
                    if param_type == str: type_str = "string"
                    elif param_type == int: type_str = "integer"
                    elif param_type == float: type_str = "number"
                    elif param_type == bool: type_str = "boolean"
                    elif param_type == list: type_str = "array"
                    elif param_type == dict: type_str = "object"
                    else: type_str = "string" # Default
                    processed_params[param_name] = {"type": type_str, "description": f"{param_name} parameter"}

            sig = inspect.signature(func)
            for param_name_sig, param_obj in sig.parameters.items():
                if param_name_sig in ("self", "cls", "db"): continue
                if param_name_sig not in processed_params and param_obj.default == inspect.Parameter.empty:
                    processed_params[param_name_sig] = {"type": "string", "description": f"{param_name_sig} parameter (auto-detected)"}
            
            self.tools[func_name] = ToolDefinition(
                name=func_name, description=description, parameters=processed_params, func=func, is_async=is_async
            )
            return func
        return decorator

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        openai_tools_list = []
        for tool_name, tool_item in self.tools.items():
            properties = {}
            required_list = []
            for param_name, param_info in tool_item.parameters.items():
                properties[param_name] = {"type": param_info["type"], "description": param_info.get("description", f"{param_name} parameter")}
                if param_info.get("enum"): properties[param_name]["enum"] = param_info["enum"]
                # Ensure 'required' defaults to True if not present in param_info, consistent with ToolParameter model
                if param_info.get("required", True): required_list.append(param_name)
            openai_tools_list.append({
                "type": "function",
                "function": {
                    "name": tool_name, "description": tool_item.description,
                    "parameters": {"type": "object", "properties": properties, "required": required_list},
                },
            })
        return openai_tools_list

    def create_fastapi_app(self) -> 'FastAPI':
        from fastapi import FastAPI as FastAPIForApp, APIRouter, HTTPException # Local import
        from pydantic import create_model as create_model_for_app # Local import

        if self._app: return self._app
        app_instance = FastAPIForApp(title=self.name, description=self.description)
        router = APIRouter()
        for tool_name, tool_def_item in self.tools.items():
            fields_for_model = {}
            for param_name, param_info_item in tool_def_item.parameters.items():
                param_type_str = param_info_item["type"]
                py_type: Type[Any]
                if param_type_str == "string": py_type = str
                elif param_type_str == "integer": py_type = int
                elif param_type_str == "number": py_type = float
                elif param_type_str == "boolean": py_type = bool
                elif param_type_str == "array": py_type = list
                elif param_type_str == "object": py_type = dict
                else: py_type = str # Default
                default_val = param_info_item.get("default", ... if param_info_item.get("required", True) else None)
                fields_for_model[param_name] = (py_type, Field(default=default_val, description=param_info_item.get("description")))
            
            RequestModelCls = create_model_for_app(f"{tool_name.capitalize()}Request", **fields_for_model)

            def create_endpoint_closure(captured_tool_def: ToolDefinition):
                async def tool_endpoint_func(request_data: RequestModelCls): # type: ignore
                    # For Pydantic v1, use .dict(). For v2, .model_dump()
                    # Assuming Pydantic v1 based on other model code.
                    params_dict = request_data.dict() 
                    try:
                        if captured_tool_def.func is None: raise ValueError("Tool function not set")
                        if captured_tool_def.is_async: tool_result = await captured_tool_def.func(**params_dict)
                        else: tool_result = captured_tool_def.func(**params_dict)
                        return {"result": tool_result}
                    except Exception as e_exec:
                        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e_exec)}")
                return tool_endpoint_func

            router.post(f"/tools/{tool_name}", summary=tool_def_item.description)(create_endpoint_closure(tool_def_item))
        
        app_instance.include_router(router)
        self._app = app_instance
        return app_instance

    async def execute_tool(self, tool_name: str, params: Dict[str, Any], db_conn: Optional[Any] = None) -> Any:
        if tool_name not in self.tools: raise ValueError(f"Tool '{tool_name}' not found in registry")
        tool_item = self.tools[tool_name]
        if tool_item.func is None: raise ValueError(f"Tool function for '{tool_name}' is not set.")
        
        tool_params_copy = params.copy()
        sig = inspect.signature(tool_item.func)
        if "db" in sig.parameters and db_conn is not None: tool_params_copy["db"] = db_conn
        
        if tool_item.is_async: return await tool_item.func(**tool_params_copy)
        else: return tool_item.func(**tool_params_copy)

class Database: # V1 Database
    def __init__(self, base_path: str, seed_sql: Optional[str] = None, seed_file: Optional[str] = None):
        self.base_path = base_path; self.seed_sql = seed_sql; self.seed_file = seed_file
        self._engine = None; self._connection = None # For SQLAlchemy sync connection

    async def setup(self):
        # Ensure base_path directory exists
        db_dir = os.path.dirname(self.base_path)
        if db_dir: # Check if db_dir is not empty (i.e., not just a filename in cwd)
            os.makedirs(db_dir, exist_ok=True)
            
        async with aiosqlite.connect(self.base_path) as db_conn:
            if self.seed_sql: await db_conn.executescript(self.seed_sql)
            elif self.seed_file:
                if not os.path.exists(self.seed_file):
                    raise FileNotFoundError(f"Seed file not found: {self.seed_file}")
                with open(self.seed_file, "r") as f_seed: seed_content = f_seed.read()
                await db_conn.executescript(seed_content)
            await db_conn.commit()

    async def get_connection(self) -> Any: # Returns aiosqlite.Connection
        if not os.path.exists(self.base_path): await self.setup()
        db_conn = await asyncio.wait_for(aiosqlite.connect(self.base_path), timeout=10.0) # Increased timeout
        await db_conn.execute("PRAGMA foreign_keys = ON"); await db_conn.execute("PRAGMA busy_timeout = 5000")
        
        async def _fetch_all(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            try:
                cursor = await db_conn.execute(query, params) if params else await db_conn.execute(query)
                if cursor.description is None: return [] # Handle queries that don't return rows (e.g. INSERT without RETURNING)
                cols = [c[0] for c in cursor.description]; rows = await cursor.fetchall()
                return [dict(zip(cols, r)) for r in rows]
            except asyncio.TimeoutError: print(f"Query timed out: {query}"); raise
        db_conn.fetch_all = _fetch_all # type: ignore
        
        async def _fetch_one(query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
            cursor = await db_conn.execute(query, params) if params else await db_conn.execute(query)
            row_data = await cursor.fetchone()
            if not row_data: return None
            if cursor.description is None: return None # Should not happen for fetch_one if row_data exists
            cols = [c[0] for c in cursor.description]; return dict(zip(cols, row_data))
        db_conn.fetch_one = _fetch_one # type: ignore

        async def _fetch_val(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
            cursor = await db_conn.execute(query, params) if params else await db_conn.execute(query)
            row_data = await cursor.fetchone(); return row_data[0] if row_data else None
        db_conn.fetch_val = _fetch_val # type: ignore
        return db_conn

    def get_sync_engine(self): # For V1 AgentEvaluator's reward function
        if not self._engine:
            if not os.path.exists(self.base_path):
                # This case is tricky for sync if setup is async.
                # For V1, assume setup was called if sync access is needed.
                raise FileNotFoundError(f"Database file {self.base_path} not found for sync engine. Ensure async setup() was called.")
            self._engine = create_engine(f"sqlite:///{self.base_path}")
        return self._engine

    def get_sync_connection(self): # For V1 AgentEvaluator's reward function
        if not self._connection:
            self._connection = self.get_sync_engine().connect()
        return self._connection

    async def create_snapshot(self, snapshot_path: str): 
        # Ensure snapshot directory exists
        snap_dir = os.path.dirname(snapshot_path)
        if snap_dir: os.makedirs(snap_dir, exist_ok=True)
        shutil.copy2(self.base_path, snapshot_path)

    async def close(self): 
        if self._connection: self._connection.close(); self._connection = None
        if self._engine: self._engine.dispose(); self._engine = None
        # aiosqlite connections are managed with 'async with' per method, so no specific close here.

class AgentEvaluator: # V1 AgentEvaluator
    def __init__(self, task_id: str, toolset_path: str, reward_path: str, base_dir: str = "./runs", seed_sql: Optional[str] = None, seed_file: Optional[str] = None):
        self.task_id = task_id; self.toolset_path = toolset_path; self.reward_path = reward_path
        self.base_dir = base_dir; self.seed_sql = seed_sql; self.seed_file = seed_file
        task_dir = os.path.join(base_dir, task_id); self.db_path = os.path.join(task_dir, "base.db")
        self.tool_registry: Optional[ToolRegistry] = None; self.reward_function: Optional[Callable[..., Any]] = None
        self.db: Optional[Database] = None

    async def setup(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            tool_mod = importlib.import_module(self.toolset_path)
            self.tool_registry = getattr(tool_mod, "R", None) 
            if not isinstance(self.tool_registry, ToolRegistry): raise AttributeError("ToolRegistry 'R' not found or not correct type in module.")
        except (ImportError, AttributeError) as e: raise ValueError(f"Failed to import tool registry from {self.toolset_path}: {e}")
        try:
            reward_mod = importlib.import_module(self.reward_path)
            self.reward_function = getattr(reward_mod, "evaluate", None)
            if not callable(self.reward_function): raise AttributeError("Reward function 'evaluate' not found or not callable in module.")
        except (ImportError, AttributeError) as e: raise ValueError(f"Failed to import reward function from {self.reward_path}: {e}")
        self.db = Database(self.db_path, self.seed_sql, self.seed_file); await self.db.setup()

    async def create_run(self, run_id: str) -> str:
        if not self.db: raise RuntimeError("Database not setup in AgentEvaluator. Call setup() first.")
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        await self.db.create_snapshot(run_db_path); return run_db_path

    async def execute_tool(self, run_id: str, tool_name: str, params: Dict[str, Any]) -> Any:
        if not self.tool_registry: raise RuntimeError("Tool registry not loaded. Call setup() first.")
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        if not os.path.exists(run_db_path): raise ValueError(f"Run database for {run_id} does not exist at {run_db_path}")
        
        run_db_instance = Database(run_db_path); db_conn_obj = await run_db_instance.get_connection()
        try: return await self.tool_registry.execute_tool(tool_name, params, db_conn_obj)
        finally: await db_conn_obj.close()

    async def evaluate(self, run_id: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if not self.reward_function: raise RuntimeError("Reward function not loaded. Call setup() first.")
        task_dir = os.path.join(self.base_dir, self.task_id)
        run_db_path = os.path.join(task_dir, f"roll_{run_id}.db")
        if not os.path.exists(run_db_path): raise ValueError(f"Run database for {run_id} does not exist at {run_db_path}")
        
        run_db_instance = Database(run_db_path)
        # V1 reward functions often expect a synchronous SQLAlchemy connection.
        # This is a known limitation when mixing with async.
        sync_db_conn = run_db_instance.get_sync_connection()
        try:
            eval_kwargs = kwargs.copy(); eval_kwargs["db"] = sync_db_conn
            result = self.reward_function(messages=messages, **eval_kwargs)
            # Ensure result is a dictionary
            if isinstance(result, BaseModel): return result.model_dump() # Pydantic v2
            if hasattr(result, "dict"): return result.dict() # Pydantic v1
            if hasattr(result, "__dict__"): return result.__dict__
            if isinstance(result, dict): return result
            # If it's a simple score (e.g. float), wrap it.
            if isinstance(result, (float, int)): return {"score": float(result), "reason": "Direct score returned."}
            raise TypeError(f"Reward function returned an unexpected type: {type(result)}. Expected dict or Pydantic model.")
        finally: sync_db_conn.close()


def load_task_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads task definitions from a JSONL file (V1 format)."""
    tasks = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line in {file_path}: {line} - Error: {e}")
    return tasks

def load_sql_from_file(file_path: str) -> str:
    """Loads SQL content from a file."""
    with open(file_path, "r") as f: return f.read()

# V2 components (Orchestrator, PythonStateResource, SQLResource, FileSystemResource, DockerResource)
# and the ForkableResource ABC definition have been moved to reward_kit.agent_v2 package.
# The Docker import logic has also been moved with DockerResource.
