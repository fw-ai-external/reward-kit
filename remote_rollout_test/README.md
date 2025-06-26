# Remote Rollout Test

This directory contains a **purely remote rollout setup** that tests the deployed MCP server on Google Cloud Run.

## 🌐 Remote Setup

- **Server**: https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app
- **Purpose**: Test end-to-end remote deployment without local server dependencies
- **Architecture**: Client → Google Cloud Run → FastMCP Server

## 📁 Files

- `test_remote_north_star.py` - Remote version of the north star interface test
- `rollouts.jsonl` - Dataset with environment configuration and prompts
- `README.md` - This file

## 🚀 Running the Tests

### Basic Remote Connection Test
Tests MCP protocol connectivity, tool discovery, and basic operations:
```bash
cd remote_rollout_test
python test_remote_north_star.py
```

### What the Test Does

1. **Connection Test**: Verifies MCP protocol connectivity to Cloud Run
2. **Tool Discovery**: Lists available tools from remote server
3. **Resource Access**: Reads initial state from remote MCP resources
4. **Tool Execution**: Makes moves using remote MCP tools
5. **North Star Interface**: Tests the full `rk.make()` and `rk.rollout()` API

## ✅ Expected Output

```
🌐 REMOTE ROLLOUT TEST SUITE
============================================================
🎯 Purpose: Test purely remote MCP deployment
📡 Server: Google Cloud Run (no local dependencies)
🧪 Tests: Connection + North Star Interface

🔌 Testing basic remote MCP connection...
📡 Connecting to: https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp
✅ Remote MCP session initialized successfully
✅ Found 1 tools on remote server:
   - lake_move: Move in the FrozenLake game...
✅ Found 2 resources on remote server:
   - game://frozen_lake/initial_state: MCP Resource: Provides initial game state
   - game://frozen_lake/config: MCP Resource: Provides game configuration
✅ Read initial state from remote server
✅ Made move on remote server
🎉 Remote MCP connection test passed!

🌟 Testing Remote North Star Interface
==================================================
🌐 REMOTE MODE: Connecting to Google Cloud Run
📡 Server URL: https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app
🚀 This tests pure remote deployment (no local server)
📊 Loaded dataset with 2 rows
✅ Policy created successfully
✅ Remote MCP environments created successfully
✅ Generated 2 trajectories from REMOTE server
🏆 Remote north star interface test completed successfully!
```

## 🎯 Benefits of Remote Setup

- **No Local Dependencies**: No need to run local MCP server
- **Production Testing**: Tests actual deployed infrastructure
- **Scalability Validation**: Verifies Cloud Run deployment scales
- **CI/CD Ready**: Can be run in automated environments
- **True Remote Rollouts**: Demonstrates real remote agent execution

## 🔧 Troubleshooting

If tests fail:

1. **Check server status**: `curl https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp`
2. **Check Cloud Run logs**: `gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=frozen-lake-mcp' --project nomadic-bison-363821 --limit 10`
3. **Verify MCP protocol**: Ensure URL ends with `/mcp` (no trailing slash)

## 🌟 North Star Validation

This setup validates the exact north star interface from the MCP design:

```python
import reward_kit as rk

# Load dataset with environment configuration and prompts
dataset = load_jsonl("rollouts.jsonl")

# Create general policy (environment-agnostic via tool calling)
policy = rk.FireworksPolicy(
    model_id="accounts/fireworks/models/qwen3-235b-a22b",
    temperature=0.2
)

# 1️⃣ create vector of MCP sessions - REMOTE SERVER
envs = rk.make(
    "https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp",  # 🌐 REMOTE URL
    dataset=dataset,
    model_id=policy.model_id
)

# 2️⃣ parallel tool-calling rollouts
trajectories = await rk.rollout(envs, policy=policy, steps=8)
```

This demonstrates that the reward-kit north star works seamlessly with remote MCP deployments! 🎉
