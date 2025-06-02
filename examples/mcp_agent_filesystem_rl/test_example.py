#!/usr/bin/env python3
"""
Test script for MCP Agent Filesystem RL Example

This script verifies that the example setup is working correctly by:
1. Testing the template directory structure
2. Testing the reward function with mock data
3. Testing MCP server connectivity (if running)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the reward-kit package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reward_kit.models import Message


def test_template_structure():
    """Test that the template directory has the correct structure."""
    print("Testing template directory structure...")

    template_path = Path("../../mcp_agent_test_templates/fs_rl_example_scenario")

    # Check directories exist
    assert template_path.exists(), f"Template directory not found: {template_path}"
    assert (template_path / "source_files").exists(), "source_files directory missing"
    assert (template_path / "archive").exists(), "archive directory missing"

    # Check important_document.txt exists
    important_doc = template_path / "source_files" / "important_document.txt"
    assert important_doc.exists(), "important_document.txt missing from source_files"

    print("✓ Template directory structure is correct")


def test_dataset_format():
    """Test that the dataset file is correctly formatted."""
    print("Testing dataset format...")

    with open("dataset.jsonl", "r") as f:
        line = f.readline().strip()
        data = json.loads(line)

    assert "prompt" in data, "Dataset missing 'prompt' field"
    assert (
        "important_document.txt" in data["prompt"]
    ), "Prompt doesn't mention target file"
    assert (
        "/data/source_files/" in data["prompt"]
    ), "Prompt doesn't mention source directory"
    assert "/data/archive/" in data["prompt"], "Prompt doesn't mention target directory"

    print("✓ Dataset format is correct")


def test_reward_function_import():
    """Test that the reward function can be imported and has correct signature."""
    print("Testing reward function import...")

    # Import the reward function
    import reward_function

    assert hasattr(
        reward_function, "mcp_filesystem_move_reward"
    ), "Reward function not found"

    # Test with mock data (without MCP server)
    messages = [
        {"role": "user", "content": "Move the file"},
        {"role": "assistant", "content": "I'll help you move the file"},
    ]

    result = reward_function.mcp_filesystem_move_reward(
        messages=messages,
        rk_session_id=None,  # This should trigger the early return
        instance_id=None,
    )

    assert result.score == 0.0, "Expected 0.0 score for missing session ID"
    assert not result.is_score_valid, "Expected invalid score for missing session ID"

    print("✓ Reward function import and basic validation works")


async def test_mcp_server_connectivity():
    """Test connectivity to MCP intermediary server if it's running."""
    print("Testing MCP server connectivity...")

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Try to connect to the MCP server
            response = await client.get("http://localhost:8001/health", timeout=5.0)
            if response.status_code == 200:
                print("✓ MCP server is running and accessible")
                return True
            else:
                print(f"⚠ MCP server responded with status {response.status_code}")
                return False

    except httpx.ConnectError:
        print("⚠ MCP server not running (this is okay for testing)")
        return False
    except Exception as e:
        print(f"⚠ Error connecting to MCP server: {e}")
        return False


def test_config_file():
    """Test that the config file is valid."""
    print("Testing config file...")

    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "model" in config, "Config missing 'model' field"
    assert "rollout_settings" in config, "Config missing 'rollout_settings' field"
    assert "dataset" in config, "Config missing 'dataset' field"
    assert "reward_function" in config, "Config missing 'reward_function' field"
    assert "agent_config" in config, "Config missing 'agent_config' field"

    # Check agent config specifics
    agent_config = config["agent_config"]
    assert agent_config["type"] == "mcp_agent", "Agent type should be 'mcp_agent'"
    assert "backend_name_ref" in agent_config, "Agent config missing 'backend_name_ref'"

    print("✓ Config file is valid")


async def main():
    """Run all tests."""
    print("Running MCP Agent Filesystem RL Example Tests")
    print("=" * 50)

    try:
        test_template_structure()
        test_dataset_format()
        test_reward_function_import()
        test_config_file()
        await test_mcp_server_connectivity()

        print("\n" + "=" * 50)
        print("✓ All tests passed! Example setup is ready.")
        print("\nTo run the full example:")
        print("1. Start the MCP intermediary server:")
        print("   python -m reward_kit.mcp_agent.main")
        print("2. Run the evaluation:")
        print("   reward-kit run --config config.yaml")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
