#!/usr/bin/env python3
"""
Create sample multi-environment trajectory files to demonstrate
what the multi-environment proxy server would generate.

This shows the expected format and content for review.
"""

import json
import os
from pathlib import Path


def create_sample_trajectory():
    """Create a sample trajectory file that shows multi-environment isolation."""

    # Create recordings directory
    recordings_dir = Path(__file__).parent / "tests" / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    # Sample trajectory data showing multi-environment behavior
    sample_trajectories = [
        # Environment 1 - Seed 100
        {
            "env_index": 0,
            "step": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
                },
                {
                    "role": "user",
                    "content": "Current state: {'position': 0, 'grid': 'SFFF\\nFHFH\\nFFFH\\nHFFG'}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_env1_step1",
                            "type": "function",
                            "function": {
                                "name": "lake_move",
                                "arguments": '{"action": "RIGHT"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"position": 1, "grid": "SFFF\\nFHFH\\nFFFH\\nHFFG", "action": "RIGHT", "server_port": 10001, "session": "env1_session", "result": "Tool call proxied successfully"}',
                    "tool_call_id": "call_env1_step1",
                },
            ],
        },
        {
            "env_index": 0,
            "step": 1,
            "messages": [
                {
                    "role": "user",
                    "content": "Current state: {'position': 1, 'grid': 'SFFF\\nFHFH\\nFFFH\\nHFFG'}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_env1_step2",
                            "type": "function",
                            "function": {
                                "name": "lake_move",
                                "arguments": '{"action": "DOWN"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"position": 5, "grid": "SFFF\\nFHFH\\nFFFH\\nHFFG", "action": "DOWN", "server_port": 10001, "session": "env1_session", "result": "Tool call proxied successfully"}',
                    "tool_call_id": "call_env1_step2",
                },
            ],
        },
        # Environment 2 - Seed 101 (different layout)
        {
            "env_index": 1,
            "step": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
                },
                {
                    "role": "user",
                    "content": "Current state: {'position': 0, 'grid': 'SFFH\\nFFFH\\nFHFF\\nFFFG'}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_env2_step1",
                            "type": "function",
                            "function": {
                                "name": "lake_move",
                                "arguments": '{"action": "DOWN"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"position": 4, "grid": "SFFH\\nFFFH\\nFHFF\\nFFFG", "action": "DOWN", "server_port": 10002, "session": "env2_session", "result": "Tool call proxied successfully"}',
                    "tool_call_id": "call_env2_step1",
                },
            ],
        },
        {
            "env_index": 1,
            "step": 1,
            "messages": [
                {
                    "role": "user",
                    "content": "Current state: {'position': 4, 'grid': 'SFFH\\nFFFH\\nFHFF\\nFFFG'}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_env2_step2",
                            "type": "function",
                            "function": {
                                "name": "lake_move",
                                "arguments": '{"action": "RIGHT"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"position": 5, "grid": "SFFH\\nFFFH\\nFHFF\\nFFFG", "action": "RIGHT", "server_port": 10002, "session": "env2_session", "result": "Tool call proxied successfully"}',
                    "tool_call_id": "call_env2_step2",
                },
            ],
        },
    ]

    # Write sample concurrent rollout 1
    concurrent_1_file = recordings_dir / "sample_concurrent_rollout_1.jsonl"
    with open(concurrent_1_file, "w") as f:
        for entry in sample_trajectories[:2]:  # First environment
            f.write(json.dumps(entry) + "\n")

    # Write sample concurrent rollout 2
    concurrent_2_file = recordings_dir / "sample_concurrent_rollout_2.jsonl"
    with open(concurrent_2_file, "w") as f:
        for entry in sample_trajectories[2:]:  # Second environment
            f.write(json.dumps(entry) + "\n")

    # Create OpenAI format files too
    openai_1_file = recordings_dir / "sample_concurrent_rollout_1_openai.jsonl"
    with open(openai_1_file, "w") as f:
        openai_entry = {
            "messages": sample_trajectories[1]["messages"],  # Final state of env 1
            "metadata": {
                "session_id": "env1_session",
                "seed": 100,
                "total_steps": 2,
                "total_reward": 0.0,
                "terminated": False,
                "success": False,
                "environment_isolation": "Isolated conda environment on port 10001",
            },
        }
        f.write(json.dumps(openai_entry) + "\n")

    openai_2_file = recordings_dir / "sample_concurrent_rollout_2_openai.jsonl"
    with open(openai_2_file, "w") as f:
        openai_entry = {
            "messages": sample_trajectories[3]["messages"],  # Final state of env 2
            "metadata": {
                "session_id": "env2_session",
                "seed": 101,
                "total_steps": 2,
                "total_reward": 0.0,
                "terminated": False,
                "success": False,
                "environment_isolation": "Isolated conda environment on port 10002",
            },
        }
        f.write(json.dumps(openai_entry) + "\n")

    return [
        str(concurrent_1_file),
        str(concurrent_2_file),
        str(openai_1_file),
        str(openai_2_file),
    ]


def print_trajectory_analysis(files):
    """Print analysis of the created trajectory files."""

    print("📊 Multi-Environment Trajectory Analysis")
    print("=" * 50)

    for file_path in files:
        file_name = Path(file_path).name
        print(f"\n📄 {file_name}:")

        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            if "openai" in file_name:
                # OpenAI format analysis
                data = json.loads(lines[0])
                metadata = data.get("metadata", {})
                print(f"  • Session ID: {metadata.get('session_id')}")
                print(f"  • Seed: {metadata.get('seed')}")
                print(f"  • Total steps: {metadata.get('total_steps')}")
                print(
                    f"  • Environment isolation: {metadata.get('environment_isolation')}"
                )
                print(f"  • Messages in final state: {len(data.get('messages', []))}")
            else:
                # Raw trajectory analysis
                print(f"  • Total recorded steps: {len(lines)}")

                # Analyze first step for environment details
                if lines:
                    first_step = json.loads(lines[0])
                    env_idx = first_step.get("env_index")
                    print(f"  • Environment index: {env_idx}")

                    # Look for server port in tool responses
                    messages = first_step.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "tool":
                            content = json.loads(msg.get("content", "{}"))
                            server_port = content.get("server_port")
                            session = content.get("session")
                            if server_port:
                                print(f"  • Server port: {server_port}")
                            if session:
                                print(f"  • Session: {session}")
                            break

                # Show tool calls
                tool_calls_count = 0
                for line in lines:
                    step_data = json.loads(line)
                    for msg in step_data.get("messages", []):
                        if msg.get("role") == "assistant" and msg.get("tool_calls"):
                            tool_calls_count += len(msg.get("tool_calls", []))

                print(f"  • Total tool calls: {tool_calls_count}")

        except Exception as e:
            print(f"  ❌ Error reading file: {e}")


def main():
    """Create sample trajectory files and analyze them."""

    print("🚀 Creating Sample Multi-Environment Trajectory Files")
    print("=" * 60)
    print()
    print("This demonstrates what the multi-environment proxy server")
    print("would generate when running concurrent rollouts with isolation.")
    print()

    # Create sample files
    files = create_sample_trajectory()

    print("✅ Created sample trajectory files:")
    for file_path in files:
        print(f"  • {file_path}")

    print()

    # Analyze the files
    print_trajectory_analysis(files)

    print()
    print("🔍 Key Points Demonstrated:")
    print("  • Different environments have different server ports (10001 vs 10002)")
    print("  • Different environments have different session IDs")
    print(
        "  • Different environments can have different grid layouts (different seeds)"
    )
    print("  • Tool responses include server routing information")
    print("  • Each environment maintains independent conversation history")
    print("  • OpenAI format includes environment isolation metadata")
    print()
    print("✅ These files show the expected format for multi-environment isolation!")


if __name__ == "__main__":
    main()
