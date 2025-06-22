## 🏅 Reward-kit × MCP — **North-Star design & first-blocker plan**

---

### 1 · Why this matters

Reward-kit’s mission is *“push a billion roll-outs per day through an interface a new grad can grok.”*
The Model Context Protocol (MCP) already gives us:

* **tool calls** → perfect match for environment **step()**
* **session header** (`Mcp-Session-Id`) for *stateful* episodes ([modelcontextprotocol.io][1])
* **initializationOptions** to pass configuration such as **seed** and **model-id** at session start ([modelcontextprotocol.io][2])

Marry these pieces once and every RL or agent team in the company—Fireworks, Reward-kit, product pods—speaks the *same* language.

---

### 2 · North-star developer experience (DX)

```python
import reward_kit as rk
from fireworks import FireworksPolicy              # thin wrapper around LLM API

policy = FireworksPolicy(
           model_id="accounts/fireworks/models/qwen3-235b-a22b",
           temperature=0.2)

seeds  = [row.seed for row in load_jsonl("rollouts.jsonl")]   # one seed per episode

envs   = rk.make(                                   # 1️⃣ create vector of MCP sessions
           "http://localhost:8000/mcp/lake@mcp",
           n=len(seeds),
           seeds=seeds,                             # passed via MCP initialize()
           model_id=policy.model_id)                # <- NEW: model baked into session

trajectories = rk.rollout(                          # 2️⃣ parallel roll-out
           envs,
           policy=policy,
           steps=512)
```

*Looks like Gymnasium, scales to 10 000 × roll-outs, talks to the same Fireworks model we use in prod.*

---

### 3 · Architecture in one picture (text)

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Training / evaluation script                                           │
 │  reward_kit.VectorEnv (N)                                               │
 │     └─ httpx.AsyncClient ───────────────┐  initialize (seed, model_id)  │
 │                                         │  lake.move(action) …         │
 └──────────────────────────────────────────┘                              │
         ▲                                     Mcp-Session-Id: ABC…        │
         │                                                                 │
         │ HTTP (Streamable)                                               │
         ▼                                                                 │
 ┌────────────────── Docker container ────────────────────────────────────┐
 │ FastMCP server (uvicorn)                                              │
 │   Session map: { SID → FrozenLakeEnv, model_id, last_used }           │
 │                                                                       │
 │   initialize_handler(seed, model_id)                                  │
 │        → env.reset(seed) ; return first obs ; set header SID          │
 │                                                                       │
 │   tool lake.move(action)                                              │
 │        → sessions[SID].step(...)                                      │
 │                                                                       │
 │   DELETE /mcp  (client-driven)  or idle-TTL sweeper (server-driven)   │
 └────────────────────────────────────────────────────────────────────────┘
```

*One container, many seeds, many concurrent sessions; scalable later via Docker-Compose, Kubernetes or Cloud Run but **not required** for local validation.*

---

### 4 · Detailed protocol flow (per episode)

| #   | Client → Server   | Header | Body                                 | Server action                                                                                |
| --- | ----------------- | ------ | ------------------------------------ | -------------------------------------------------------------------------------------------- |
| 1   | `initialize`      | —      | `{"seed": 123, "model_id": "…a22b"}` | create env, reset, **return** `{initialObservation: 5}` **and** header `Mcp-Session-Id: SID` |
| 2-N | `lake.move`       | `SID`  | `{"action":"LEFT"}`                  | env.step; return obs/reward/...                                                              |
| End | **DELETE** `/mcp` | `SID`  | —                                    | pop session dict                                                                             |

Spec references: session header rules and delete semantics ([modelcontextprotocol.io][1], [modelcontextprotocol.io][2]).

---

### 5 · Corrected Architecture (Lessons Learned)

**Critical Correction**: The original north star design had a fundamental flaw. It mixed production and simulation concerns. Here's the corrected architecture:

#### Production Server (Stateless Shim)
```python
# frozen_lake_server.py - Production deployment
from mcp.server.fastmcp import FastMCP, Context
from gymnasium.envs.toy_text import FrozenLakeEnv

# Global state - single session (like Google Docs MCP)
GAME_ENV: FrozenLakeEnv = None
CURRENT_POSITION: int = 0

app = FastMCP("FrozenLake-v1")

@app.tool(name="lake_move")
def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
    global GAME_ENV, CURRENT_POSITION
    # Stateless shim - operates on global state
    # Auto-resets when game ends
    return {"position": CURRENT_POSITION, "reward": reward, ...}
```

#### Simulation Server (Research Wrapper)
```python
# simulation_server.py - Uses framework, enforces tool matching
from reward_kit.mcp.simulation_server import SimulationServerBase
import frozen_lake_server  # Import to enforce signature matching

class FrozenLakeSimulation(SimulationServerBase):
    def get_domain_tools(self):
        # CRITICAL: Must match production exactly
        production_tools = set(frozen_lake_server.app._tool_manager._tools.keys())
        simulation_tools = {"lake_move"}
        assert simulation_tools == production_tools

        return {"lake_move": self._lake_move}
```

**Key Corrections**:
1. **No session management tools exposed** (violates MCP spec)
2. **Production servers are stateless shims** (like real-world MCPs)
3. **Simulation servers are independent implementations** (not proxies)
4. **Framework enforces tool signature matching** (prevents drift)
5. **Session management happens internally** (MCP initializationOptions)

---

### 6 · Revised validation criteria (Corrected Architecture)

| Validation Item | Success Criteria |
|----------------|-------------------|
| **Production Server Validation** | Stateless server with single global session; auto-resets on completion |
| **Simulation Framework** | `SimulationServerBase` prevents session tool pollution; enforces tool signature matching |
| **Tool Signature Parity** | Simulation tools exactly match production tools (enforced via import + assertion) |
| **Framework Enforcement** | Impossible to accidentally expose `initialize_session` or similar session tools |
| **Independent Deployment** | Production server deployable without simulation dependencies |
| **Concurrent Session Management** | Simulation server handles multiple independent sessions internally |

**Validation status**: ✅ **COMPLETED** in `examples/frozen_lake_mcp/`

**Key Learnings**:
1. **Session tools violate MCP spec** - must be internal
2. **Production ≠ Simulation** - completely different architectures
3. **Framework enforcement required** - prevents accidental mistakes
4. **Tool signature matching critical** - for interoperability

---

### 7 · Implementation Roadmap (Updated)

| Milestone | Outcome | Status |
|-----------|---------|--------|
| **M0** | ✅ **COMPLETE**: Corrected architecture implemented in `examples/frozen_lake_mcp/` | **DONE** |
| **M1** | `rk.make()` & `rk.rollout()` client integration with simulation servers | **NEXT** |
| **M2** | Production deployment patterns (Docker, Cloud Run) for both server types | **NEXT** |
| **M3** | Framework templates for other environments (CartPole, Atari, etc.) | **FUTURE** |

**M0 Deliverables (Completed)**:
- ✅ Production server (`frozen_lake_server.py`) - stateless shim
- ✅ Simulation server (`simulation_server.py`) - framework-based
- ✅ Framework (`SimulationServerBase`) - prevents tool pollution
- ✅ Tool signature enforcement - import + assertion validation
- ✅ Client (`rollout_client.py`) - batch rollout capabilities

---

## 8 · Why this scales company-wide

* **One spec, many surfaces** — product teams expose *their* environments as MPC tools; agents and evaluation harnesses reuse reward-kit with zero coupling.
* **Model routing baked in** — `model_id` travels with session; same code runs on Fireworks, Anthropic, or internal HF checkpoints.
* **Local-first** — everybody can reproduce a 10 k-episode experiment on a laptop **today**; cloud scaling is an additive knob, not a dependency.

---

### 📌 Take-away

> **North star**: *“One line to publish any env as an MCP server, one line to roll-out thousands of seeded sessions against a chosen model.”*
> **First blocker**: prove the MCP Python SDK handles `Mcp-Session-Id` + `initializationOptions` exactly as the spec says. Once that test turns green, the rest of the roadmap is execution, not research.

[1]: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports?utm_source=chatgpt.com "Transports - Model Context Protocol"
[2]: https://modelcontextprotocol.io/docs/concepts/transports?utm_source=chatgpt.com "Transports - Model Context Protocol"
