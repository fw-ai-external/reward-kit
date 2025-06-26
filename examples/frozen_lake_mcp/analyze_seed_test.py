#!/usr/bin/env python3
"""
Analysis script to verify different seeds produce different behaviors
from the clean trajectory logs.
"""

import json
from collections import defaultdict


def analyze_trajectory_logs(log_file: str):
    """Analyze the clean trajectory logs to verify seed behavior."""
    print("🔍 Analyzing Clean Trajectory Logs")
    print("=" * 50)

    # Load all log entries
    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]

    # Group by environment/session
    env_data = defaultdict(lambda: {"seed": None, "actions": [], "observations": []})

    for log in logs:
        if log["type"] == "initial_state":
            env_idx = log["env_index"]
            env_data[env_idx]["seed"] = log["seed"]
            env_data[env_idx]["session_id"] = log["session_id"]

        elif log["type"] == "tool_call":
            env_idx = log["env_index"]
            action = log["arguments"]["action"]
            step = log["step"]
            env_data[env_idx]["actions"].append(f"Step {step}: {action}")

        elif log["type"] == "step_result":
            env_idx = log["env_index"]
            obs = log["observation"]
            if "grid_layout" in obs:
                grid = obs["grid_layout"].replace("\n", " | ")
                env_data[env_idx]["observations"].append(f"Step {log['step']}: {grid}")

    # Analyze differences
    print("📊 Seed Behavior Analysis:")
    print("-" * 30)

    for env_idx in sorted(env_data.keys()):
        data = env_data[env_idx]
        print(
            f"\n🌱 Environment {env_idx} (Seed: {data['seed']}, Session: {data['session_id']}):"
        )
        print(f"   First 3 actions: {data['actions'][:3]}")

        # Show first observation with grid layout
        if data["observations"]:
            first_obs = data["observations"][0]
            print(f"   Initial grid: {first_obs}")

    # Show action comparison
    print("\n🔄 Action Sequence Comparison:")
    print("-" * 30)

    first_actions = []
    for env_idx in sorted(env_data.keys()):
        data = env_data[env_idx]
        seed = data["seed"]
        if data["actions"]:
            first_action = data["actions"][0].split(": ")[1]  # Extract just the action
            first_actions.append((seed, first_action))
            print(f"   Seed {seed:3d}: {first_action}")

    # Verify they're different
    unique_first_actions = set(action for _, action in first_actions)
    if len(unique_first_actions) > 1:
        print(
            f"\n✅ SUCCESS: {len(unique_first_actions)} different first actions detected!"
        )
        print("   This confirms different seeds produce different behaviors.")
    else:
        print(f"\n❌ WARNING: All environments took the same first action!")
        print("   Seeds might not be producing different behaviors.")

    # Summary stats
    rollout_complete = [log for log in logs if log["type"] == "rollout_complete"]
    if rollout_complete:
        summary = rollout_complete[0]
        print(f"\n📈 Rollout Summary:")
        print(f"   Total trajectories: {summary['num_trajectories']}")
        print(f"   Successful: {summary['successful_trajectories']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Average steps: {summary['avg_steps']:.1f}")
        print(f"   Total duration: {summary['total_duration']:.2f}s")


if __name__ == "__main__":
    analyze_trajectory_logs("clean_trajectories.jsonl")
