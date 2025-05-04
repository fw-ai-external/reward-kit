# Issues and Tasks

## RewardKit Agent Evaluation Framework — **"Task Bundle"** Design

This extension to Reward Kit enables evaluation of agentic models through a modular, self-contained task architecture.

*(reward functions + tools live side-by-side in the **same folder**; the core library stays lean and generic)*

---

### 0. Guiding principles

1. **Self-contained task bundles**:
   Each task folder contains everything needed for evaluation:

   ```
   my_task/
   ├─ reward.py           # Reward function with @reward_function decorator
   ├─ tools.py            # Tool registry for this specific task
   ├─ seed.sql            # Initial DB state (optional)
   └─ task.jsonl          # Dataset rows with task specifications
   ```
2. **Core framework agnostic to tools**:
   The framework imports tools based on dataset specifications rather than shipping them.
3. **One import path = one tool registry**:
   Separate tasks have separate tool registries:

   ```
   flight_task/  (reward.py + tools.py)
   hotel_task/   (reward.py + tools.py)
   ```

---

## 1. Task Bundle Structure

| File         | Required | Purpose                                                                                              |
| ------------ | -------- | ---------------------------------------------------------------------------------------------------- |
| `reward.py`  | **yes**  | Defines a single `@reward_function` function, compatible with current Reward Kit                      |
| `tools.py`   | **yes**  | Defines one `ToolRegistry` and all tool functions                                                    |
| `seed.sql`   | no       | Initial DB fixture (can also be embedded in dataset row)                                             |
| `task.jsonl` | **yes**  | Dataset rows, each with `toolset: "my_task.tools"` for proper import                                 |

> **Note**: For complex tools, use a `tools/` directory with `__init__.py` that instantiates one registry.

---

## 2. Implementation Examples

### 2.1 reward.py

```python
from reward_kit import reward_function, RewardOutput

@reward_function
def evaluate(messages, *, db, end_goal_sql, **kwargs):
    ok = db.execute(end_goal_sql).scalar_one()
    return RewardOutput(
        score=1.0 if ok else 0.0,
        reason="Task completed successfully" if ok else "Task incomplete",
        metrics={"task_complete": {"score": 1.0 if ok else 0.0, "reason": "Goal achieved" if ok else "Goal not met"}}
    )
```

### 2.2 tools.py

```python
from reward_kit.agent import ToolRegistry

# Create tool registry
R = ToolRegistry("flight_tools")

@R.tool(description="List flights", parameters={"origin": str, "dest": str, "date": str})
async def search_flights(origin, dest, date, db):
    return await db.fetch_all("""
        SELECT id, depart, arrive, seats_available
        FROM flights
        WHERE origin=:o AND dest=:d AND date(depart)=:date AND seats_available>0
    """, {"o": origin, "d": dest, "date": date})

@R.tool(description="Reserve seat", parameters={"flight_id": int, "passenger": str})
async def create_booking(flight_id, passenger, db):
    bid = await db.fetch_val("SELECT hex(randomblob(4))")
    await db.execute("INSERT INTO bookings(id, flight_id, passenger, status)"
                     "VALUES(:bid,:fid,:pass,'reserved')",
                     {"bid": bid, "fid": flight_id, "pass": passenger})
    return {"booking_id": bid}

@R.tool(description="Pay for booking", parameters={"booking_id": str})
async def pay_booking(booking_id, db):
    await db.execute("UPDATE bookings SET status='paid' WHERE id=:bid",
                     {"bid": booking_id})
    return {"ok": True}

# Create FastAPI app for debugging
app = R.create_fastapi_app()  # Use: uvicorn my_task.tools:app --reload
```

---

## 3. Task Dataset Format

```json
{
  "id": "flight.booking.001",
  "seed_sql": "file:seed.sql",
  "end_goal_sql": "SELECT COUNT(*)>0 FROM bookings WHERE passenger='Alice' AND status='paid';",
  "initial_messages": [
    {"role":"user","content":"Book me a flight from SFO to JFK for tomorrow morning"}
  ],
  "sim_user_prompt": "You are Alice, a traveller.",
  "toolset": "my_task.tools",
  "n_rollouts": 4
}
```

Multiple tasks can share the same tools by referencing the same `toolset` path. For different toolsets, create separate task directories.

---

## 4. Framework Implementation

### 4.1 Dynamic Import System

```python
# row["toolset"] is "my_task.tools"
tool_module = importlib.import_module(row["toolset"])
tools_spec = tool_module.R.get_openai_tools()   # Format for LLM
tool_app = tool_module.R.create_fastapi_app()   # For in-process tool calls
reward_mod = importlib.import_module("my_task.reward")
evaluate_fn = reward_mod.evaluate              # Already decorated with @reward_function
```

The framework only references what's defined in the task bundle, maintaining clean separation.

### 4.2 Evaluation Storage

```
runs/
└─ <row_id>/                    # One directory per task
   └─ base.db                   # Initial seeded database
      roll_<uuid>.db            # Copy-on-write for each evaluation run
```

Evaluation artifacts are stored outside task directories to avoid Git bloat.

---

## 5. CLI Usage

```bash
cd my_task
export MODEL_AGENT=openai/gpt-4o-mini
export MODEL_SIM=openai/gpt-3.5-turbo
reward-kit agent-eval --dataset task.jsonl
```

CLI commands will be integrated with the existing Reward Kit CLI, maintaining consistency with current patterns.

---

## 6. Developer Experience

| Goal                         | Method                                                                      |
| ---------------------------- | --------------------------------------------------------------------------- |
| Debug tools                  | `uvicorn my_task.tools:app --reload` and test via API requests             |
| Test reward function         | `python -c "from my_task.reward import evaluate; print(evaluate([...]))"` |
| Share tasks                  | Package task directory; recipients run with `reward-kit agent-eval`         |
| Use custom models            | `export MODEL_AGENT=/path/to/model; reward-kit agent-eval ...`             |
| Add helper utilities         | Create module in task directory and import with `from .utils import ...`    |

---

## Integration Plan

1. **Create agent module**: Add `reward_kit.agent` with `ToolRegistry` class
2. **Extend CLI**: Add `agent-eval` command to `reward-kit` CLI  
3. **Implement database system**: Add SQLite database management for task state
4. **Add orchestrator**: Create component to manage agent-environment interactions
5. **Create documentation**: Add tutorials and examples for agent evaluation
6. **Build examples**: Develop sample task bundles demonstrating the framework

This design leverages existing Reward Kit patterns for reward functions while adding the infrastructure needed for agent evaluation with tools.