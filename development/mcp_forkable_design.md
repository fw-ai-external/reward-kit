# Design: MCP Integration for Forkable and Proxied Backend Resources

## 1. Introduction

This document outlines the design for integrating diverse backend MCP server resources with the Model Context Protocol (MCP), primarily for use in Reinforcement Learning (RL) rollouts. The challenge is to provide isolated, "forkable" environments for stateful servers (like databases, filesystem views, or memory stores) while also offering efficient access to stateless or "embarrassingly parallel" utility servers (like web fetch or time services).

The solution involves an intermediary MCP server, to be developed as part of `reward-kit` (potentially as an optional package `reward-kit-mcp-agent`), named the `RewardKit-Intermediary-MCP-Server` (RK Intermediary Server). This server will abstract the complexities of managing these backend resources. It is designed to be flexible, supporting two primary modes for provisioning and interacting with backend instances:

1.  **Local Orchestration Mode:** The RK Intermediary Server directly manages the lifecycle of backend MCP server instances using **Docker** locally. This provides robust isolation and state forking capabilities (via Docker images and containers).
2.  **Remote Orchestration Proxy Mode:** The RK Intermediary Server acts as a client or proxy to a **customer-provided remote orchestration service** (e.g., a Kubernetes-based system, a custom API, or another higher-level MCP server). This remote service handles the actual provisioning and lifecycle of backend instances.

This dual-mode design ensures `reward-kit` can adapt to various customer environments and operational capabilities, whether they prefer local Docker-based management by the intermediary or integration with their existing remote resource provisioning systems.

This design covers various backend types, including:
*   **Stateful/Forkable (requiring isolated instances per session):** `filesystem`, `duckdb` (e.g., via `mcp-server-motherduck`), `memory` (knowledge graph), `everything` (comprehensive test server).
*   **Stateless/"Embarrassingly Parallel" (can use shared instances):** `fetch`, `time`.

The RK Intermediary Server will expose a unified set of MCP tools to the `RewardKit-Client`, allowing RL agents to interact with these diverse backend resources in a consistent manner, regardless of the underlying orchestration mode.

## 2. Core Architecture

The system involves three main participants:

1.  **`RewardKit-Client`**:
    *   An **Orchestrator**: Requests setup of backend environments (sessions with one or more backend instances) from the RK Intermediary Server.
    *   **`RLRolloutWorker` instances**: Perform RL rollouts, interacting with dedicated or shared backend instances via tools exposed by the RK Intermediary Server, using a session context.

2.  **`RewardKit-Intermediary-MCP-Server` (RK Intermediary Server)**:
    *   Determines orchestration mode (local Docker or remote proxy) based on configuration for each backend type.
    *   **In Local Orchestration Mode:** Acts as a Docker orchestrator. Manages the lifecycle of backend server Docker containers (run, commit for stateful templates, stop, remove).
    *   **In Remote Orchestration Proxy Mode:** Translates `RewardKit-Client` requests into API calls to a customer's remote orchestration service.
    *   Handles session management, mapping `RewardKit-Client` sessions to specific backend instances (local containers or remote service handles).
    *   Exposes MCP tools that abstract the underlying orchestration operations and tool forwarding.

3.  **`Resource Provisioning Environment`**: This can be:
    *   **Local Docker Environment**: Managed directly by the RK Intermediary Server. Requires Docker images for backend MCPs (e.g., from `/Users/bennychen/Documents/references/servers/`) and a Docker daemon accessible to the intermediary. Template data for stateful backends can be baked into images or mounted for `docker commit` workflows.
    *   **Remote Orchestration Service**: An API provided by the customer (e.g., Kubernetes-based, custom HTTP API, or a dedicated MCP orchestration server). The customer provides an API endpoint and authentication details for the RK Intermediary Server to request and manage remote backend instances.

### Overall Flow Diagram (Illustrating Dual Orchestration Modes)

```mermaid
sequenceDiagram
    participant RKC_Orchestrator as RewardKit Client (Orchestrator)
    participant RLRolloutWorker as RewardKit Client (RLRolloutWorker)
    participant RK_Intermediary_MCP as RewardKit Intermediary MCP Server
    participant Orchestration_Client as Intermediary's Orchestration Client <br/> (LocalDockerOrchestrator OR RemoteAPIClient)
    participant Target_System as Target System <br/> (Local Docker Daemon OR Remote Orchestration API)
    participant Backend_Instance_X as Backend MCP Instance X <br/> (Local Container: HTTP or Stdio OR Remote Service Endpoint)

    RKC_Orchestrator->>RK_Intermediary_MCP: MCP: tool_name='initialize_session', args={backends: [{backend_name_ref: "fs_env", ...}]}
    activate RK_Intermediary_MCP
    Note over RK_Intermediary_MCP, Orchestration_Client: Intermediary selects OrchestrationClient based on config for "fs_env".
    RK_Intermediary_MCP->>Orchestration_Client: request_instances(backend_config, num_instances, template_info)
    activate Orchestration_Client
    Orchestration_Client->>Target_System: Provision N instances (e.g., docker run with port mapping for HTTP, or just run for Stdio; OR POST /create_instance for remote)
    activate Target_System
    Target_System-->>Orchestration_Client: Instance_X_Details (e.g., ContainerID+HostPort for HTTP, ContainerID for Stdio OR RemoteEndpointURL+RemoteInstanceID)
    deactivate Target_System
    Orchestration_Client-->>RK_Intermediary_MCP: List of ManagedInstanceInfo (with Instance_X_Details including transport type)
    deactivate Orchestration_Client
    Note over RK_Intermediary_MCP: Creates rkSessionId. <br/> Maps rkSessionId to backend instances. <br/> Stores in session state.
    RK_Intermediary_MCP-->>RKC_Orchestrator: Returns {session_id: rkSessionId, backend_details: [...]}
    deactivate RK_Intermediary_MCP

    RKC_Orchestrator-->>RLRolloutWorker: Assigns rkSessionId and instance_X client-facing ID

    RLRolloutWorker->>RK_Intermediary_MCP: MCP: tool_name='call_backend_tool', args={backend_name_ref:"fs_env", tool_name:"read_file", ...} <br/> (HTTP Header: mcp-session-id: rkSessionId)
    activate RK_Intermediary_MCP
    Note over RK_Intermediary_MCP: Extracts rkSessionId. <br/> Gets Backend_Instance_X info (local host_port for HTTP, container_id for Stdio; OR remote_url).
    RK_Intermediary_MCP->>Orchestration_Client: forward_tool_call(instance_X_info, "read_file", args)
    activate Orchestration_Client
    Orchestration_Client->>Backend_Instance_X: HTTP MCP call to target endpoint (tool: "read_file") OR Stdio MCP interaction (e.g. via docker attach)
    activate Backend_Instance_X
    Backend_Instance_X-->>Orchestration_Client: Tool result
    deactivate Backend_Instance_X
    Orchestration_Client-->>RK_Intermediary_MCP: Tool result
    deactivate Orchestration_Client
    RK_Intermediary_MCP-->>RLRolloutWorker: Returns tool result
    deactivate RK_Intermediary_MCP

    opt Cleanup
        RKC_Orchestrator->>RK_Intermediary_MCP: MCP: tool_name='cleanup_session' (HTTP Header: mcp-session-id: rkSessionId)
        activate RK_Intermediary_MCP
        Note over RK_Intermediary_MCP: Retrieves session. For each managed instance:
        RK_Intermediary_MCP->>Orchestration_Client: request_cleanup(instance_X_info)
        activate Orchestration_Client
        Orchestration_Client->>Target_System: Request cleanup (e.g., docker stop/rm OR POST /delete_instance)
        activate Target_System
        Target_System-->>Orchestration_Client: Cleanup status
        deactivate Target_System
        Orchestration_Client-->>RK_Intermediary_MCP: Cleanup status for instance X
        deactivate Orchestration_Client
        Note over RK_Intermediary_MCP: (If local Docker & temp image) docker rmi <temporary_template_image_id>
        Note over RK_Intermediary_MCP: Clear internal session state.
        RK_Intermediary_MCP-->>RKC_Orchestrator: Returns {status: "cleaned"}
        deactivate RK_Intermediary_MCP
    end
```

## 3. Key Interaction Flows (Generalized for Orchestration Modes)

### 3.1. Session Initialization with Backend Instances
1.  The `RewardKit-Client` (Orchestrator) calls `initialize_session` on the `RK Intermediary Server`, specifying desired backends.
2.  The `RK Intermediary Server`, for each requested backend:
    *   Consults `AppConfig` to get the `BackendServerConfig` (including `orchestration_mode`).
    *   Selects the appropriate `OrchestrationClient` (e.g., `LocalDockerOrchestrationClient` or `RemoteHttpOrchestrationClient`).
    *   Calls the `OrchestrationClient` to provision `num_instances`.
    *   **Local Docker Mode (Stateful, HTTP Transport):** The client may trigger `docker commit` for template state, then `docker run` for instances, returning container info and mapped host ports.
    *   **Local Docker Mode (Stateful, Stdio Transport):** The client may trigger `docker commit` for template state, then `docker run` (without port mapping, but with `stdin_open=True`), returning container info. Interaction occurs via `docker attach` or similar.
    *   **Remote API Mode (Stateful):** The client makes API calls to the customer's service to create instances, returning remote endpoint URLs and instance IDs.
    *   **Stateless/Shared Mode (Local or Remote):** The client ensures a shared instance is running and returns its access details (URL for HTTP, container ID for local stdio).
3.  The `RK Intermediary Server` creates an `IntermediarySession`, stores `ManagedInstanceInfo` (which now accommodates local container details for HTTP/stdio and remote instance details) for all provisioned backends, and associates this with a new `rkSessionId`.
4.  Returns `rkSessionId` and instance details to the Orchestrator.

### 3.2. Rollout Interaction
1.  Orchestrator assigns `rkSessionId` and client-facing `instance_id`(s) to an `RLRolloutWorker`.
2.  `RLRolloutWorker` calls `call_backend_tool` on the `RK Intermediary Server` (with `rkSessionId` in header).
3.  `RK Intermediary Server` retrieves the `IntermediarySession`, identifies the target `ManagedInstanceInfo`.
4.  It uses the `OrchestrationClient` associated with that instance to forward the tool call:
    *   **Local Docker (HTTP Transport):** Makes an HTTP MCP call to `http://localhost:<host_port>/mcp`.
    *   **Local Docker (Stdio Transport):** Interacts with the container's stdio (e.g., via `docker attach` to send/receive MCP messages).
    *   **Remote API:** Makes an HTTP MCP call to the `remote_endpoint_url`.
5.  Result is returned to the `RLRolloutWorker`.

### 3.3. Session Cleanup
1.  Orchestrator calls `cleanup_session` (with `rkSessionId` in header).
2.  `RK Intermediary Server` retrieves session. For each managed instance, it calls the appropriate `OrchestrationClient` to terminate/delete the backend instance (local container stop/rm or remote API delete call).
3.  Temporary local Docker images (if any) are removed. Session object is deleted.

## 4. `RewardKit-Intermediary-MCP-Server` Design Details
(This section is now part of Section 8: Detailed Implementation Plan)

## 5. Resource Provisioning Environment Expectations

1.  **Common Requirement**: Access to backend MCP server definitions (e.g., executables, Docker images from sources like `/Users/bennychen/Documents/references/servers/`).
2.  **For Local Docker Orchestration Mode**:
    *   A Docker environment where the `RK Intermediary Server` can run and has permissions to interact with the Docker daemon.
    *   Backend MCP server Docker images can expose either an **HTTP MCP transport** (listening on a port) or an **stdio MCP transport** (interacting via stdin/stdout). The orchestrator must handle both.
    *   For HTTP transport, images should expose a port. For stdio, they should be configured to read MCP messages from stdin and write responses to stdout.
    *   Template data (optional) for stateful backends, either baked into images or mountable for `docker commit` workflows.
    *   Startup checks for stdio servers might be simpler (e.g., container is running) compared to HTTP pings, unless a specific stdio health check tool is available.
3.  **For Remote Orchestration Proxy Mode**:
    *   A customer-provided remote orchestration API (e.g., Kubernetes-based, custom HTTP API, or another MCP server for orchestration).
    *   A clear API contract (endpoints, request/response schemas, authentication methods) for this remote service.
    *   The remote service is responsible for the actual lifecycle and state isolation of the backend instances it provisions.

## 6. Comparison with Original Examples

Original MCP servers (`filesystem`, `mcp-server-motherduck`, etc.) are typically standalone.
*   **Our Intermediary's Role**: It adds a layer of orchestration and session management.
    *   **Local Docker Mode**: It automates what one would do manually with Docker (commit, run, manage ports) to get isolated stateful instances.
    *   **Remote Proxy Mode**: It acts as a standardized client to potentially diverse customer-specific remote provisioning systems, presenting a consistent MCP interface to `RewardKit`.

## 7. Benefits
*   **Flexible Orchestration**: Supports both local Docker-based management and proxying to remote customer systems.
*   **Robust Isolation (Local Docker Mode)**: Strong isolation via Docker containers.
*   **Standardized Deployment (Local Docker Mode)**: Backend MCPs managed as containers.
*   **Simplified State Forking (Local Docker Mode)**: `docker commit` is a powerful primitive.
*   **Adaptability**: Can integrate with various customer infrastructure capabilities.

This design provides a scalable and maintainable solution for integrating forkable SQL database resources into the `reward-kit`'s RL evaluation workflows using MCP. The following section details a plan for implementing this solution.

## 8. Detailed Implementation Plan

This section outlines a phased approach to implement the `RewardKit-Intermediary-MCP-Server` within a new `reward_kit.mcp_agent` module. The implementation will leverage the MCP Python SDK found at `/Users/bennychen/Documents/references/python-sdk/src/mcp` for core server functionalities.

### 8.1. Module Structure (`reward_kit.mcp_agent`)

```
reward_kit/
└── mcp_agent/
    ├── __init__.py
    ├── intermediary_server.py  # Main RewardKitIntermediaryServer logic
    ├── orchestration/
    │   ├── __init__.py
    │   ├── base_client.py      # AbstractOrchestrationClient, ManagedInstanceInfo
    │   ├── local_docker_client.py # LocalDockerOrchestrationClient
    │   └── remote_http_client.py  # RemoteHttpOrchestrationClient (example for K8s/Custom API)
    ├── config.py               # Pydantic models for configuration
    ├── session.py              # Custom session class (IntermediarySession)
    ├── backends/
    │   ├── __init__.py
    |   ├── generic backends that can handle different types of servers
    └── main.py                 # CLI entry point for running the server (optional)

tests/
└── mcp_agent/
    ├── __init__.py
    ├── test_intermediary_server.py
    ├── orchestration/
    │   ├── test_local_docker_client.py
    │   └── test_remote_http_client.py
    ├── backends/
    │   ├── test_filesystem_backend.py
    │   # ... other backend handler tests
    └── mock_mcp_server_image/
        ├── Dockerfile
        └── mock_server.py # Generic mock MCP server for containerized testing
    └── mock_remote_orchestrator.py # Mock HTTP server for remote orchestration API
```

Most servers are already docker based, and have persistence through file.

for mcp main repo examples:
- src/filesystem
- src/memory

they are already docker based.

- https://github.com/motherduckdb/mcp-server-motherduck/tree/main
this one is not docker based, so we will need to think about replicating the databse file to make sure we can actually fork the resources and have multiple rollouts

### 8.2. Core Components and Classes

1.  **`AppConfig` (in `config.py`)**:
    *   `backends: List[BackendServerConfig]`: Configurations for all backend types the intermediary can manage/proxy.
    *   `log_level: str`.
    *   `global_docker_options: Optional[Dict[str, Any]]` (e.g., default network).
    *   `global_remote_api_defaults: Optional[Dict[str, Any]]` (e.g., default timeouts for remote calls).

2.  **`RemoteApiConfig` (in `config.py`)**:
    *   Pydantic model for remote orchestration API details.
    *   `base_url: str`.
    *   `create_instance_endpoint: str` (e.g., `/instances`).
    *   `delete_instance_endpoint_template: str` (e.g., `/instances/{remote_instance_id}`).
    *   `call_tool_endpoint_template: Optional[str]` (If tool calls are also proxied via orchestrator, else direct to instance).
    *   `auth_type: Literal["none", "bearer_token", "custom_header"]`.
    *   `auth_details: Optional[Dict[str, str]]`.

3.  **`BackendServerConfig` (in `config.py`)**:
    *   `backend_name_ref: str` (Unique name, e.g., "workspace_fs", "shared_fetch_service").
    *   `backend_type: Literal["filesystem", "duckdb", "memory", "everything", "fetch", "time"]`.
    *   `orchestration_mode: Literal["local_docker", "remote_http_api"]`.
    *   `instance_scoping: Literal["session", "shared_global"]`.
    *   `mcp_transport: Literal["http", "stdio"] = "http"` (New field, defaults to "http". Determines interaction protocol).
    *   **Local Docker Specific (`if orchestration_mode == 'local_docker'`):**
        *   `docker_image: str`.
        *   `container_port: Optional[int]` (Internal port of MCP app in container. Required if `mcp_transport` is "http").
        *   `template_data_path_host: Optional[str]` (For pre-seeding state).
        *   `container_template_data_path: Optional[str]` (Mount path for template data in setup container).
        *   `docker_run_args: Optional[List[str]]`.
        *   `startup_check_mcp_tool: Optional[Dict[str, Any]]`.
    *   **Remote API Specific (`if orchestration_mode == 'remote_http_api'`):**
        *   `remote_api_config_ref: Optional[str]` (Reference to a global `RemoteApiConfig` if shared, or inline).
        *   `remote_resource_type_identifier: str` (Type identifier for the remote API, e.g., "duckdb_v1", "filesystem_large").

4.  **`AbstractOrchestrationClient` (in `orchestration/base_client.py`)**:
    *   Interface:
        *   `async def provision_instances(backend_config: BackendServerConfig, num_instances: int, session_id: str, template_details: Optional[Any]) -> List[ManagedInstanceInfo]`.
        *   `async def deprovision_instances(instances: List[ManagedInstanceInfo])`.
        *   `async def call_tool_on_instance(instance: ManagedInstanceInfo, tool_name: str, tool_args: dict) -> dict`. (Implementation will differ based on `instance.mcp_transport`).

5.  **`LocalDockerOrchestrationClient(AbstractOrchestrationClient)` (in `orchestration/local_docker_client.py`)**:
    *   Uses Docker SDK for Python.
    *   For HTTP transport: Implements methods for `docker run` with port mapping, `commit`, `stop`, `rm`, `rmi`, HTTP calls to container. Manages host port allocation.
    *   For Stdio transport: Implements `docker run` (no port mapping, `stdin_open=True`), `commit`, `stop`, `rm`, `rmi`. Tool calls involve interacting with container's stdio (e.g., via `container.attach_socket()`).
    *   Startup checks are transport-dependent.

6.  **`RemoteHttpOrchestrationClient(AbstractOrchestrationClient)` (in `orchestration/remote_http_client.py`)**:
    *   Uses `httpx` or `aiohttp`. Implements methods by calling configured remote API endpoints.

7.  **`ManagedInstanceInfo` (in `orchestration/base_client.py`)**:
    *   Stores all necessary details to interact with a provisioned backend instance.
    *   `instance_id: str` (Client-facing ID within a session).
    *   `backend_name_ref: str`.
    *   `orchestration_mode: Literal["local_docker", "remote_http_api"]`.
    *   `mcp_transport: Literal["http", "stdio"]` (New field, indicates how to talk to the instance).
    *   `mcp_endpoint_url: Optional[str]` (e.g., `http://localhost:<host_port>/mcp` for HTTP, `None` for stdio).
    *   `internal_instance_details: Dict[str, Any]` (e.g., `{"container_id": "...", "host_port": ...}` for Docker/HTTP, or `{"container_id": "..."}` for Docker/stdio, or `{"remote_instance_id": "...", "access_token": "..."}`).
    *   `committed_image_tag: Optional[str]` (If local Docker and committed).

8.  **`IntermediarySession(mcp.server.BaseSession)` (in `session.py`)**:
    *   `session_id: str` (This is the transport-level session ID).
    *   `managed_backends: Dict[str, List[ManagedInstanceInfo]]` (Keyed by `backend_name_ref`).
    *   `temporary_docker_images: List[str]` (For cleanup in local Docker mode).
    *   **Note**: This class was changed to `IntermediarySessionData` (a Pydantic BaseModel/dataclass) as `FastMCP` does not allow specifying a custom session class for its internal `MCPServer`. The `RewardKitIntermediaryServer` now manages a dictionary `self.intermediary_session_data: Dict[str, IntermediarySessionData]`.

9.  **`AbstractBackendHandler` (in `backends/base.py`)**:
    *   `async def initialize_session_instances(session_data: IntermediarySessionData, req: BackendInitRequest, server_cfg: BackendServerConfig, orch_client: AbstractOrchestrationClient) -> List[ManagedInstanceInfo]`. (Changed `session` to `session_data`)
    *   `async def cleanup_session_instances(session_data: IntermediarySessionData, backend_name_ref: str, orch_client: AbstractOrchestrationClient)`. (Changed `session` to `session_data`)

10. **Concrete `BackendHandler`s** (in `backends/*.py`):
    *   (Largely as before, but interact with `IntermediarySessionData`).

11. **`RewardKitIntermediaryServer(mcp.server.fastmcp.server.FastMCP)` (in `intermediary_server.py`)**:
    *   **Base Class Rationale**: Changed to `mcp.server.fastmcp.server.FastMCP`. While `FastMCP` does not allow specifying a custom session class for its internal `MCPServer`, it provides a more modern interface and handles its own `ToolManager`. Custom session state is managed externally in `self.intermediary_session_data`.
    *   **Initialization**: The `__init__` method calls `super().__init__(name=..., instructions=...)`.
    *   **Tool Registration**: Uses `FastMCP`'s `self.add_tool(handler_method, name="tool_name")`. A single proxy tool (`execute_proxied_tool`) is used to dispatch to internal handlers.
    *   **Context in Tool Handlers**: The proxy tool handler receives `mcp_ctx: mcp.server.fastmcp.server.Context`.
        *   `mcp_ctx.request_context` provides the `mcp.shared.context.RequestContext`.
        *   `mcp_ctx.session` provides the `mcp.server.session.ServerSession`.
    *   **Session ID Challenge**: Accessing the transport-level session ID from within the tool handler (via `mcp_ctx`) has proven problematic. Neither `mcp_ctx.session.session_id` (public inherited) nor `mcp_ctx.session._session_id` (internal) are reliably accessible. `mcp_ctx.request_context.request.path_params` has also been empty. This is a critical unresolved issue.
    *   Initializes `OrchestrationClient`s (one `LocalDockerOrchestrationClient`, and one `RemoteHttpOrchestrationClient` per unique remote API config).
    *   Initializes `BackendHandler`s.
    *   Manages global shared instances for stateless backends (using the appropriate `OrchestrationClient`).
    *   Tool handlers select the correct `OrchestrationClient` and `BackendHandler` based on `BackendServerConfig.orchestration_mode` and `backend_type`.

### 8.3. Tool Definitions for Intermediary Server (Largely Unchanged from Client Perspective)

1.  **`initialize_session`**:
    *   Input: `backends: List[BackendInitRequest]` (as before, but `BackendInitRequest` might include `orchestration_preference: Optional[Literal["local_docker", "remote_http_api"]]` if a backend supports multiple modes).
    *   Output: `{ session_id: "<rk_session_id>", initialized_backends: List[BackendInitResult] }`.
    *   Action: Creates `IntermediarySession`. For each request, determines `OrchestrationClient` and calls `BackendHandler.initialize_session_instances`. The `ManagedInstanceInfo` returned will include the `mcp_transport`.

2.  **`call_backend_tool`**:
    *   Input: (as before).
    *   Action: Retrieves `IntermediarySession`. Finds `ManagedInstanceInfo`. Calls `OrchestrationClient.call_tool_on_instance()`, which will use the appropriate transport (HTTP or stdio) based on `instance.mcp_transport`.

3.  **`cleanup_session`**:
    *   Input: (as before).
    *   Action: Retrieves `IntermediarySession`. Calls `BackendHandler.cleanup_session_instances` for each managed backend, which in turn uses the `OrchestrationClient`.

### 8.4. Leveraging the MCP Python SDK & Dependencies
*   MCP Python SDK from `/Users/bennychen/Documents/references/python-sdk/src/mcp/` as a source dependency. `reward_kit.mcp_agent` as an optional extra.
*   Dependencies: `docker` (for local Docker), `httpx` or `aiohttp` (for remote API client).

### 8.5. Configuration (`reward_kit/mcp_agent/config.py`)
*   `BackendServerConfig` updated with `orchestration_mode` and `remote_api_config_ref`.
*   `AppConfig` includes list of `BackendServerConfig` and potentially list of `RemoteApiConfig` definitions.

### 8.6. Testing Strategy
*   **`OrchestrationClient` Unit Tests**:
    *   `LocalDockerOrchestrationClient`: Mock Docker SDK.
    *   `RemoteHttpOrchestrationClient`: Mock `httpx` calls.
*   **`BackendHandler` Unit Tests**: Mock `OrchestrationClient`.
*   **`RewardKitIntermediaryServer` Integration Tests**:
    *   Test with `LocalDockerOrchestrationClient` using `mock_mcp_server_image`.
    *   Test with `RemoteHttpOrchestrationClient` using `mock_remote_orchestrator.py` (a mock HTTP server simulating the customer's API).

### 8.7. Further Considerations
*   (As before: Resource Limits, Error Handling, Timeouts, Idempotency, Docker Security).
*   **Remote Orchestration API Contract**: If customers provide a remote orchestration API, a clear specification (or options for adapting to common ones like a K8s operator API) will be needed for the `RemoteHttpOrchestrationClient`.
*   **Authentication with Remote Orchestrators**: `RemoteApiConfig` needs robust auth handling.

This revised plan incorporates the flexibility to use either local Docker orchestration or proxy to remote orchestration services, making the `RewardKit-Intermediary-MCP-Server` more adaptable.

## 9. Current Status and Next Steps

**Current Status (as of 2025-05-31):**

The `RewardKit-Intermediary-MCP-Server` and its `LocalDockerOrchestrationClient` have been significantly refactored to introduce support for stdio-based MCP transports, alongside the existing HTTP transport. Key changes implemented include:

1.  **Configuration Updates:**
    *   `BackendServerConfig` (in `reward_kit/mcp_agent/config.py`) now includes an `mcp_transport: Literal["http", "stdio"]` field (defaulting to "http").
    *   `container_port` and `startup_check_mcp_tool` (if HTTP-based) are now understood to be relevant only for `http` transport.
    *   `mcp_agent_config.yaml` has been updated for `filesystem_test` and `memory_test` backends to specify `mcp_transport: "stdio"` and remove port/HTTP health check configurations.

2.  **Orchestration Logic (`LocalDockerOrchestrationClient`):**
    *   `provision_instances`:
        *   Correctly launches stdio containers without port mappings and with `stdin_open=True`.
        *   Skips HTTP-based startup checks for stdio containers, relying on the container remaining in a "running" state.
    *   `call_tool_on_instance`:
        *   Differentiates logic based on `instance.mcp_transport`.
        *   For stdio, it now attempts to use `container.attach_socket(params={'stdin': True, 'stdout': True, 'stderr': False, 'stream': True, 'logs': False})` to get a raw socket to the container's stdio.
        *   It sends a fully structured MCP `CallToolRequest` (including `protocol_version` and `message_type`), JSON-encoded and newline-terminated, to the container's stdin via the raw socket's `sendall()` method (accessed via `attached_socket._sock.sendall()`).
        *   It then calls `raw_stdio_socket.shutdown(socket.SHUT_WR)` to signal EOF on the container's stdin.
        *   A revised read loop attempts to read the response from the container's stdout, demultiplexing the Docker stream format (handling the 8-byte header for stream type and payload size) using `select.select` for non-blocking reads with a timeout.

3.  **Testing (`test_docker_run.py` and `run_mcp_test.sh`):**
    *   `test_docker_run.py` has been updated to test stdio servers (`mcp/filesystem` by default) by launching them with correct arguments/volumes and attempting an stdio MCP call (`list_tools`) using a similar `attach_socket` and stream demultiplexing logic.
    *   Running `run_mcp_test.sh` (which executes `test_mcp_client.py` against the intermediary server) shows:
        *   Stdio containers (`filesystem_test`, `memory_test`) are provisioned successfully by the intermediary.
        *   The `initialize_session` tool call to the intermediary is successful.
        *   However, subsequent `call_backend_tool` calls (e.g., `list_files` on `filesystem_test`) to the intermediary, which are then proxied to the stdio container, are still failing with: `RuntimeError: MCP stdio tool call failed: Unexpected error. Details: No response from stdio container ...`. This is preceded by a log: `WARNING:reward_kit.mcp_agent.orchestration.local_docker_client:Stdio socket header read returned no data (closed by container ...).`
    *   This indicates that the Node.js MCP server process in the container is closing its stdout stream or exiting before sending any data back, even after its stdin has been written to and the write-half of the socket has been shut down.

**Detailed Next Steps for Further Development:**

The primary challenge remains establishing successful two-way communication with the stdio-based Node.js MCP servers (`mcp/filesystem`, `mcp/memory`). The current approach of directly managing the attached socket and MCP message framing in `LocalDockerOrchestrationClient` has not yet yielded a response from these servers.

The next phase should focus on a more isolated and foundational approach to stdio MCP communication, leveraging the official MCP Python SDK more directly, before integrating this proven method back into the `LocalDockerOrchestrationClient`.

0.  **Read all the files in python `mcp` library in .venv. We will not be able to make any good calls unless we read all the code, and luckily there isn't a lot of code for the whole mcp library.
1.  **Develop a Standalone Stdio MCP Client Test Script (`standalone_stdio_mcp_client.py`):**
    *   **Objective:** Create a new, simple Python script (e.g., `standalone_stdio_test.py`) that uses the official MCP Python SDK (`mcp.client.stdio.stdio_client` and `mcp.ClientSession`) to interact with a *single*, locally running Dockerized stdio MCP server (e.g., `mcp/filesystem`).
    *   **Details:**
        *   This script will *not* use the `RewardKit-Intermediary-MCP-Server` or `LocalDockerOrchestrationClient`.
        *   It will manually run the target Docker container (e.g., `mcp/filesystem`) using `docker.from_env().containers.run(...)` with `stdin_open=True`, `detach=True`, and correct command/volumes.
        *   **Crucial Part - Interfacing with `ClientSession`:**
            *   After the container is running, obtain its raw stdio socket using `container.attach_socket(params={'stdin': True, 'stdout': True, 'stderr': False, 'stream': True, 'logs': False})._sock`.
            *   The `mcp.client.ClientSession` expects `AsyncMessageReadStream` and `AsyncMessageWriteStream` objects (from `mcp.protocol.streams`). These normally wrap asyncio `StreamReader` and `StreamWriter`.
            *   **Investigate and Implement Stream Adaptation:**
                *   **Option A (Preferred if feasible):** Determine if a raw socket (like the one from `attach_socket()._sock`) can be converted or wrapped into asyncio `StreamReader` and `StreamWriter` objects. Python's `asyncio` library might offer utilities for this (e.g., `loop.connect_accepted_socket()` or `loop.create_connection()` with a pre-existing socket, though these are typically for server-side or client-side connection establishment, not for wrapping an already connected Docker-provided socket). This needs careful research within the `asyncio` and `docker-py` contexts.
                *   **Option B (Custom Wrappers):** If Option A is not straightforward, create custom Python classes that:
                    *   Implement the `AsyncMessageReadStream` protocol (primarily `async def read_message(self) -> Optional[Message]`) and `AsyncMessageWriteStream` protocol (primarily `async def write_message(self, message: Message)`).
                    *   These wrappers will take the single duplex raw socket from `attach_socket()._sock`.
                    *   `write_message`: Will serialize the MCP `Message` object to JSON, append a newline, encode to UTF-8, and send it over the raw socket using `socket.sendall()`. It must also handle `socket.shutdown(socket.SHUT_WR)` appropriately if the server requires it after each message or at the end of a logical request.
                    *   `read_message`: Will read from the raw socket, handle Docker's 8-byte stream demultiplexing header (to isolate stdout), buffer incoming data, parse newline-terminated JSON from the stdout stream, and deserialize it into an MCP `Message` object. This needs to be robust against partial reads and handle potential timeouts.
        *   **Perform Full MCP Lifecycle with `ClientSession`:**
            1.  Instantiate `ClientSession` with the adapted read/write streams.
            2.  `await client_session.initialize(capabilities=..., client_information=...)`: Send the `InitializeRequest` and process the `InitializeResult`. The exact `capabilities` and `client_information` to send should be minimal and compliant with what a basic client would offer. Refer to MCP Python SDK examples or the specification. This step is critical and currently missing from `LocalDockerOrchestrationClient`.
            3.  `tools = await client_session.list_tools()`: Attempt to call the `list_tools` MCP tool.
            4.  Log the `tools` response.
            5.  If `list_tools` is successful, attempt `response = await client_session.call_tool("ping", {})` and log the response.
            6.  (Optional, if `list_tools` reveals them) Attempt a filesystem-specific tool like `read_file` or `write_file` with appropriate arguments, ensuring the target path exists within the container's mounted volume.
        *   **Logging:** Implement verbose logging of all raw data sent to and received from the socket, as well as the structured MCP messages, to aid debugging.
    *   **Expected Outcome:** A clear, working example of how to use `ClientSession` with an existing Docker container's stdio streams. This will serve as the reference implementation.

2.  **Analyze MCP SDK for Stdio Framing and `ClientSession` Internals:**
    *   **Objective:** Fully understand the MCP message structures (especially `InitializeRequest`, `InitializeResult`, `CallToolRequest`, `CallToolResult`), expected stdio framing (newline-delimited JSON is assumed but verify), and any session management nuances handled by `ClientSession` and `AsyncMessageStream`.
    *   **Action:** Thoroughly review the source code of `mcp.client.session.ClientSession`, `mcp.protocol.streams.AsyncMessageStream`, `mcp.protocol.messages`, and any relevant stdio transport examples within the MCP Python SDK provided at `/home/bchen/references/python-sdk/src/mcp/`.

3.  **Bridge Standalone Test Learnings to `LocalDockerOrchestrationClient`:**
    *   **Objective:** Apply the successful communication patterns and `ClientSession` usage (or its replicated logic) from `standalone_stdio_mcp_client.py` to the `call_tool_on_instance` method in `LocalDockerOrchestrationClient`.
    *   **Details:**
        *   If custom stream wrappers were needed for `ClientSession` in the standalone test, integrate these into `LocalDockerOrchestrationClient`.
        *   Ensure that for each tool call to an stdio container, the equivalent of `session.initialize()` is performed if the MCP server expects it per "connection" (each `attach_socket` might be a new connection from the server's perspective). This might mean caching `ClientSession` objects per container if they can be reused, or performing a quick initialize + call_tool + close sequence.
        *   Refine error handling and timeouts based on insights from the standalone test.

4.  **Incremental Testing within `reward-kit`:**
    *   Once `LocalDockerOrchestrationClient` is updated, re-run `test_docker_run.py` (which should ideally be simplified to use the now-working client if possible, or kept as a very low-level socket test).
    *   Systematically run `bash run_mcp_test.sh` and debug any remaining issues in the context of the intermediary server.
    *   Focus on getting `list_tools` working first, then `ping`, then other tools.

5.  **Future Enhancements (Post-Core Functionality):**
    *   Re-evaluate `startup_check_mcp_tool` for stdio: Can a simple `list_tools` call via the established stdio `ClientSession` serve as a health check?
    *   Address the FastMCP session ID propagation issue within the intermediary server.
    *   Handle template data paths and `docker commit` workflows for stateful stdio servers if their state management requires more than just volume mounts.

This structured approach, starting with a focused standalone test using the MCP SDK's `ClientSession`, should provide a more reliable foundation for fixing the stdio communication.
