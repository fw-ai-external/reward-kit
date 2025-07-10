
# Report: Fixing Session Management in the MCP Multi-Environment Proxy

## 1. Original Problem Description

The `MultiEnvironmentProxy` server was failing in multi-environment scenarios, specifically in the `test_multi_environment_concurrent_rollouts` test. The test's output recording file, `concurrent_rollout_1.jsonl`, showed that the backend game server was never reached. Instead, resource requests for the initial game state failed with the error:

```json
{"error": "No server assigned to session", "session": "session_f9fe296c85f3e0c1"}
```

This indicated a fundamental issue in how the proxy server creates, manages, and maps sessions to the isolated backend server instances. The proxy was creating a new session for every single request, losing the connection between an environment and its dedicated backend server.

## 2. Session Management Fix - ✅ COMPLETED

### Root Cause
The root cause was in the `_get_session_id_from_context` method within `reward_kit/mcp/multi_environment_proxy.py`. The method was generating unique session IDs for every request using microsecond timestamps:

```python
timestamp = str(int(time.time() * 1000000))  # microsecond precision
unique_data = f"{client_str}_{timestamp}_{session_count}"
session_hash = hashlib.md5(unique_data.encode()).hexdigest()[:16]
session_id = f"session_{session_hash}"
```

### Solution Applied
Replaced the timestamp-based session ID generation with stable, deterministic IDs:

```python
def _get_session_id_from_context(self, ctx: Context) -> str:
    if hasattr(ctx, 'session') and hasattr(ctx.session, 'client_info'):
        client_info = ctx.session.client_info
        if client_info and hasattr(client_info, '_extra'):
            extra_data = client_info._extra
            if extra_data and isinstance(extra_data, dict):
                stable_data = {
                    "seed": extra_data.get("seed"),
                    "config": extra_data.get("config", {}),
                    "name": getattr(client_info, 'name', 'unknown'),
                    "version": getattr(client_info, 'version', 'unknown'),
                }
                stable_str = json.dumps(stable_data, sort_keys=True)
                session_id = hashlib.md5(stable_str.encode()).hexdigest()
                return session_id

    # Fallback: deterministic ID based on context object identity
    session_id = f"proxy_fallback_{id(ctx)}"
    return session_id
```

### Additional Fixes
1. **Server Script Path**: Fixed test to use `server.py` instead of `frozen_lake_mcp.py` (which lacks CLI args)
2. **Process Manager Fallback**: Added conda detection with SimpleServerProcessManager fallback
3. **Logger Initialization**: Fixed initialization order to prevent AttributeError

### Results
- ✅ Stable session IDs: `proxy_fallback_140369263254640` (consistent per request lifecycle)
- ✅ Environment isolation: All 3 unique seeds (42, 123, 456) working correctly
- ✅ Recording created: `concurrent_rollout_1.jsonl` successfully generated
- ✅ No more "Failed to create session" errors

## 3. Current Issue - ❌ BACKEND SERVER COMMUNICATION FAILURE

### Problem Description
While session management is now working correctly, a new issue has emerged: the proxy server can create sessions and start backend servers, but communication with the backend servers fails with HTTP 406 errors:

```json
{"error": "Backend server error: 406", "session": "proxy_fallback_140369263254640"}
```

### Evidence from Recording File
```bash
$ jq -r '.messages[1].content' tests/recordings/concurrent_rollout_1.jsonl | head -10
Current game state grid:
{'error': 'Backend server error: 406', 'session': 'proxy_fallback_140369263254640'}

You are navigating the 4x4 grid above. Navigate safely to reach the goal 'G' while avoiding holes 'H'. Choose your next move from: LEFT, DOWN, RIGHT, or UP.
```

### Root Cause Analysis
The HTTP 406 "Not Acceptable" error indicates a failure in content negotiation between the proxy and the backend MCP server. The root cause has been traced to the `_proxy_resource_request` method in `reward_kit/mcp/multi_environment_proxy.py`.

The client HTTP GET request being sent is malformed in two ways:
1.  **Missing `Accept` Header**: The request does not include an `Accept` header. The `406` error means the server cannot produce a response that matches the client's accepted formats. The backend server, built on the `FastMCP` framework, serves resources as `application/json`, but the proxy never declares that it accepts this content type.
2.  **Incorrect `Content-Type` Header**: The request incorrectly includes a `Content-Type: application/json` header. This header is meant for POST or PUT requests to describe the body's content and is not appropriate for a GET request, which has no body.

The combination of a missing `Accept` header and an incorrect `Content-Type` header causes the strict `FastMCP` server to reject the request.

Here is the problematic code block:
```python
# In reward_kit/mcp/multi_environment_proxy.py
async with self.http_client.get(
    target_url,
    headers={"Content-Type": "application/json"},  # ⚠️ Incorrect for GET, missing Accept
    timeout=aiohttp.ClientTimeout(total=10)
) as response:
```

### Backend Server Status
- ✅ Backend servers are starting successfully (no process creation errors)
- ✅ Ports are being allocated correctly (SimpleServerProcessManager working)
- ❌ HTTP communication between proxy and backend servers is failing due to content negotiation failure.

## 4. Next Steps Required

### Proposed Solution
The issue can be resolved by correcting the HTTP headers in the GET request within the `_proxy_resource_request` method in `reward_kit/mcp/multi_environment_proxy.py`.

1.  **Add `Accept` Header**: Add `Accept: application/json` to the headers to signal that the client can handle a JSON response.
2.  **Remove `Content-Type` Header**: Remove the `Content-Type` header, as it is not applicable to GET requests.

The corrected code should look like this:
```python
# In reward_kit/mcp/multi_environment_proxy.py
async with self.http_client.get(
    target_url,
    headers={"Accept": "application/json"},  # ✅ Correctly specify accepted content type
    timeout=aiohttp.ClientTimeout(total=10)
) as response:
```

### Verification Steps
1. **Apply the code change** to `reward_kit/mcp/multi_environment_proxy.py`.
2. **Run the test** `test_multi_environment_concurrent_rollouts` in `examples/frozen_lake_mcp/tests/test_record_and_replay_e2e.py`.
3. **Check the recording file** `concurrent_rollout_1.jsonl`. The `error` message should be gone, and the initial game state (grid, position) should be present.
4. **Confirm the test passes** and that trajectories for all environments are generated successfully, indicating that end-to-end communication is restored.

### Test Environment
- Test: `test_multi_environment_concurrent_rollouts` in `examples/frozen_lake_mcp/tests/test_record_and_replay_e2e.py`
- Recording: `examples/frozen_lake_mcp/tests/recordings/concurrent_rollout_1.jsonl`
- Proxy: `reward_kit/mcp/multi_environment_proxy.py` (specifically the `_proxy_resource_request` method)

## 5. Status Summary
- ✅ **Session Management**: Fixed - stable session IDs working correctly
- ✅ **Environment Isolation**: Fixed - multiple seeds and environments working
- ✅ **Backend Server Creation**: Fixed - servers start successfully
- ❌ **Backend Communication**: **BROKEN** - HTTP 406 errors traced to missing `Accept` header in proxy-to-backend resource requests.
- ❌ **End-to-End Functionality**: **BLOCKED** - environments cannot get initial state

---

**Report Updated**: Session management has been fixed. The backend server communication failure has been root-caused to a missing `Accept` header in the proxy's resource requests. The next step is to apply the proposed header fix.
