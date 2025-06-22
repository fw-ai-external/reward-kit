# Next Steps - Corrected MCP Architecture

This document outlines the next development steps after achieving the corrected MCP architecture. The fundamental issues from the original design have been resolved.

## ✅ **What's Been Completed (M0)**

### **Architecture Corrections**
- ✅ **Production Server**: Stateless shim with global state (`frozen_lake_server.py`)
- ✅ **Simulation Server**: Independent implementation with framework (`simulation_server.py`)
- ✅ **Framework Enforcement**: `SimulationServerBase` prevents session tool pollution
- ✅ **Tool Signature Matching**: Import + assertion prevents tool drift
- ✅ **Clean Separation**: No code sharing, independent deployments

### **Key Learnings Applied**
1. **Session management tools violate MCP spec** - Now internal only
2. **Production servers are stateless shims** - Like Google Docs MCP pattern
3. **Simulation servers are research-specific** - Completely separate implementations
4. **Framework enforcement is critical** - Prevents accidental violations
5. **Tool signature stability is essential** - Enforced via imports + assertions

## 🎯 **Priority 1: Client Integration (M1)**

### **Immediate Tasks**

#### **1.1 Implement `rk.make()` Client**
```python
# Target API from north star
envs = rk.make(
    "http://localhost:8000/mcp",  # Simulation server URL
    n=100,
    seeds=[1, 2, 3, ...],
    model_id="qwen3-235b-a22b"
)
```

**Implementation Location**: `reward_kit/mcp/client.py`

**Requirements**:
- Connect to simulation servers only (not production)
- Handle multiple concurrent sessions
- Seed management per session
- Error handling and retries
- Compatible with existing reward-kit patterns

#### **1.2 Implement `rk.rollout()` Integration**
```python
trajectories = rk.rollout(envs, policy=policy, steps=512)
```

**Integration Points**:
- Extend existing `reward_kit/evaluation.py`
- Support MCP-based environments
- Maintain compatibility with non-MCP environments
- Batch rollout optimization

### **1.3 Validation Tests**
- [ ] Test `rk.make()` with FrozenLake simulation server
- [ ] Validate concurrent session handling (100+ parallel)
- [ ] Test rollout performance vs direct Gymnasium
- [ ] Integration test with actual policy training

## 🚀 **Priority 2: Production Deployment (M2)**

### **2.1 Production Server Deployment**

#### **Docker Containers**
```dockerfile
# Dockerfile.production
FROM python:3.11-slim
COPY frozen_lake_server.py .
RUN pip install mcp[server] gymnasium
EXPOSE 8000
CMD ["python", "frozen_lake_server.py", "--transport", "streamable-http"]
```

#### **Cloud Run Deployment**
- [ ] Create Cloud Run configuration
- [ ] Auto-scaling based on connection count
- [ ] Health checks and monitoring
- [ ] Production logging and metrics

### **2.2 Simulation Server Deployment**

#### **Research Environment Setup**
```yaml
# docker-compose.yml
services:
  production:
    build:
      dockerfile: Dockerfile.production
    ports: ["8001:8000"]

  simulation:
    build:
      dockerfile: Dockerfile.simulation
    ports: ["8000:8000"]
    depends_on: [production]
```

## 🧪 **Priority 3: Framework Expansion (M3)**

### **3.1 Environment Templates**

#### **CartPole Template**
```python
class CartPoleSimulation(SimulationServerBase):
    def get_domain_tools(self):
        # Import cartpole_server to enforce matching
        import cartpole_server
        production_tools = set(cartpole_server.app._tool_manager._tools.keys())
        # ... rest of implementation
```

#### **Template Generation**
- [ ] Create cookiecutter template for new environments
- [ ] Automated production + simulation server generation
- [ ] Tool signature validation in CI/CD

### **3.2 Advanced Features**

#### **Multi-Environment Support**
```python
# Support multiple games in one simulation server
class MultiGameSimulation(SimulationServerBase):
    def get_domain_tools(self):
        return {
            "frozen_lake_move": self._frozen_lake_move,
            "cartpole_move": self._cartpole_move,
            # Each matches corresponding production server
        }
```

#### **Configuration Management**
- [ ] Environment parameter validation
- [ ] Configuration templates and presets
- [ ] Runtime configuration updates

## 🛠️ **Implementation Guidelines**

### **Framework Rules (CRITICAL)**

1. **Never expose session management tools**:
   ```python
   # ❌ FORBIDDEN - This violates MCP spec
   @app.tool()
   def initialize_session(): pass

   # ✅ CORRECT - Internal framework management
   def _get_or_create_session(ctx): pass
   ```

2. **Always enforce tool signature matching**:
   ```python
   # ✅ REQUIRED in all simulation servers
   import production_server
   production_tools = set(production_server.app._tool_manager._tools.keys())
   assert simulation_tools == production_tools
   ```

3. **Maintain independent implementations**:
   ```python
   # ❌ FORBIDDEN - No proxying
   await production_client.post(...)

   # ✅ CORRECT - Separate implementation
   result = env.step(action)
   ```

### **Testing Requirements**

#### **Tool Signature Tests**
```python
def test_tool_signature_matching():
    """Ensure simulation tools exactly match production."""
    from production_server import app as prod_app
    from simulation_server import SimulationClass

    prod_tools = set(prod_app._tool_manager._tools.keys())
    sim_server = SimulationClass("test")
    sim_tools = set(sim_server.mcp._tool_manager._tools.keys())

    assert prod_tools == sim_tools
```

#### **Framework Enforcement Tests**
```python
def test_no_session_tools_exposed():
    """Verify framework prevents session tool pollution."""
    sim_tools = get_simulation_tools()
    forbidden_tools = ['initialize_session', 'get_session_info', 'create_session']

    for tool in forbidden_tools:
        assert tool not in sim_tools
```

## 🚨 **Critical Warnings**

### **Common Mistakes to Avoid**

1. **Session Tool Pollution**: Framework prevents this, but be vigilant
2. **Tool Signature Drift**: Always import production server for validation
3. **Proxying Temptation**: Simulation servers must be independent implementations
4. **Framework Bypass**: Never use `@app.tool()` directly in simulation servers

### **Architecture Principles**

1. **Production First**: Production servers define the interface
2. **Simulation Follows**: Simulation servers match production exactly
3. **Framework Enforces**: Use `SimulationServerBase` to prevent violations
4. **Independent Deployment**: No shared code between production and simulation

## 📋 **Handoff Checklist**

- [ ] Review corrected architecture in `examples/frozen_lake_mcp/`
- [ ] Understand tool signature enforcement mechanism
- [ ] Validate framework prevents session tool pollution
- [ ] Test production server deployment independently
- [ ] Confirm simulation server matches production tools exactly

## 🎯 **Success Metrics**

- [ ] `rk.make()` creates 1000+ concurrent sessions successfully
- [ ] `rk.rollout()` performance within 10% of direct Gymnasium
- [ ] Production servers deployable to Cloud Run/Docker
- [ ] Framework prevents all session management tool violations
- [ ] Tool signature matching enforced in CI/CD

---

**Current Status**: M0 complete, ready for M1 implementation
**Next Developer**: Focus on `rk.make()` and `rk.rollout()` integration
**Architecture**: Proven, tested, and ready for scaling
