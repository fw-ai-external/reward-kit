# FrozenLake MCP Example - Three-Layer Architecture

This directory demonstrates the clean three-layer architecture for MCP-based environments, with proper separation of concerns between production servers, simulation capabilities, and clients.

## 🏗️ **Architecture Overview**

```
Client → Simulation Server → Production Server
```

**Three Layers:**
1. **Production Server** - Standalone, deployable game server
2. **Simulation Server** - Wraps production server, adds research capabilities
3. **Client** - Performs rollouts using simulation server

## 📁 **File Organization**

### **Layer 1: Production Server**

**`frozen_lake_server.py`** - 🏭 **PRODUCTION MCP SERVER**
- Standalone FrozenLake game server
- Only tool: `lake_move(action: str)`
- No session management tools
- Deployable anywhere (Docker, Cloud Run, etc.)
- Zero dependencies on simulation frameworks

### **Layer 2: Simulation Server**

**`simulation_server.py`** - 🧪 **SIMULATION MCP SERVER**
- Wraps production server via HTTP
- Adds simulation tools: `initialize_session`, `get_initial_observation`
- Manages seeds, configuration, session state
- Proxies game actions to production server
- Enables research and batch rollouts

### **Layer 3: Client**

**`rollout_client.py`** - 🎮 **ROLLOUT CLIENT**
- Only talks to simulation server
- Performs single episodes and batch rollouts
- Handles seeding and configuration
- Clean interface for reward-kit integration

### **Framework Alternative**

**`mcp_server_new.py`** + **`frozen_lake_adapter.py`** - Framework-based approach
- Uses reward-kit MCP utilities
- Alternative to pure production server

### **Legacy**

**`simulation_wrapper.py`** - Legacy testing utilities (kept for reference)

## 🚀 **Quick Start**

### **Three-Layer Setup**

```bash
# 1. Start production server
python frozen_lake_server.py --transport streamable-http --port 8001

# 2. Start simulation server (wraps production)
python simulation_server.py --transport streamable-http --port 8000 --prod-url http://localhost:8001

# 3. Run rollouts via simulation server
python rollout_client.py --test single --seed 42
python rollout_client.py --test batch --count 10
```

### **Framework Alternative**

```bash
# Start framework-based server
python mcp_server_new.py --transport streamable-http --port 8000

# Test with legacy tools
python simulation_wrapper.py --test basic
```

## 🎯 **Key Benefits**

| Layer | Purpose | Dependencies | Deployment |
|-------|---------|--------------|------------|
| **Production** | Pure game logic | FastMCP + Gym only | Any platform |
| **Simulation** | Research tools | Production server + httpx | Testing environments |
| **Client** | Rollout execution | Simulation server only | Local/CI |

## 🏭 **Production Server Details**

**`frozen_lake_server.py` characteristics:**
- ✅ **Single tool**: `lake_move(action: str)` only
- ✅ **Auto-start**: Game begins at position 0 automatically
- ✅ **Self-contained**: No external dependencies
- ✅ **Deployable**: Docker, Kubernetes, Cloud Run ready
- ✅ **Domain-specific**: Pure FrozenLake logic

**Example deployment:**
```dockerfile
FROM python:3.11-slim
COPY frozen_lake_server.py .
RUN pip install mcp[server] gymnasium
EXPOSE 8000
CMD ["python", "frozen_lake_server.py", "--transport", "streamable-http", "--port", "8000"]
```

## 🧪 **Simulation Server Details**

**`simulation_server.py` capabilities:**
- ✅ **Session management**: `initialize_session(seed)`
- ✅ **State tracking**: Episode metadata and statistics
- ✅ **Production proxy**: Forwards `lake_move` to production server
- ✅ **Configuration**: Per-episode seeding and config
- ✅ **Research tools**: Designed for reward-kit integration

**Tools provided:**
```python
initialize_session(seed: Optional[int]) -> Dict
get_initial_observation() -> Dict
lake_move(action: str) -> Dict  # Proxied to production
get_session_info() -> Dict
```

## 🎮 **Client Details**

**`rollout_client.py` features:**
- ✅ **Single episodes**: With optional seeding
- ✅ **Batch rollouts**: Multiple episodes with statistics
- ✅ **Clean interface**: Only talks to simulation server
- ✅ **Error handling**: Comprehensive error reporting
- ✅ **Metrics**: Success rate, goal achievement, performance

**Usage patterns:**
```python
client = SimulationRolloutClient("http://localhost:8000")
result = await client.run_episode(seed=42)
batch_results = await client.run_batch(count=100)
```

## 🎯 **Architecture Principles**

### **1. Clean Separation**
- Production server has NO knowledge of simulation
- Simulation server has NO shared code with production
- Client only knows about simulation interface

### **2. Independent Deployment**
- Production server deploys independently (like Google Docs MCP or Shopify MCP)
- Simulation server is research-specific infrastructure
- No code sharing between layers

### **3. Domain Specificity**
- Production server is FrozenLake-specific
- Would be completely different for Google Docs or Shopify
- No generic abstractions in production layer

### **4. Zero Tool Pollution**
- Production server: Only game actions
- Simulation server: Adds research tools without contaminating production
- Clear interface boundaries

## 🧪 **Testing Examples**

### **Single Episode with Seed**
```bash
python rollout_client.py --test single --seed 42
```

### **Batch Performance Testing**
```bash
python rollout_client.py --test batch --count 50
```

### **Production Server Validation**
```bash
# Test production server directly (advanced)
curl -X POST http://localhost:8001/mcp/tools/lake_move \
  -H "Content-Type: application/json" \
  -d '{"action": "DOWN"}'
```

## 📋 **Development Workflow**

1. **Build production server** - Domain-specific, deployable
2. **Add simulation wrapper** - Research capabilities around production
3. **Create rollout clients** - Use simulation server for experiments
4. **Deploy independently** - Production and simulation have different lifecycles

## 🎯 **Recommendations**

- **For production deployment**: Use `frozen_lake_server.py`
- **For research/rollouts**: Use `simulation_server.py` + `rollout_client.py`
- **For framework exploration**: Try `mcp_server_new.py`
- **For new games**: Copy production server pattern (no code sharing)

## 🔗 **Integration with Reward-Kit**

The simulation server provides the interface for reward-kit's north star API:

```python
# Future reward-kit integration
envs = rk.make("http://localhost:8000/mcp", n=100, seeds=[1,2,3,...])
trajectories = rk.rollout(envs, policy, steps=512)
```

---

**Status**: ✅ **Production Ready** - Three-layer architecture with clean separation.
**Pattern**: Client → Simulation Server → Production Server
