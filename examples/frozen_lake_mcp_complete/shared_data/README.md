# Shared Test Data for FrozenLake MCP Integration

This directory contains standardized test data for consistent CI testing of the FrozenLake MCP integration.

## Files

### `rollouts.jsonl`
Standardized dataset with 3 environment configurations using seeds 42, 123, and 999. Each entry contains:
- `id`: Environment identifier
- `seed`: Random seed for deterministic behavior
- `system_prompt`: System prompt for the environment
- `user_prompt_template`: Template for formatting user prompts
- `environment_context`: Additional context (game type, grid size, etc.)

### `recorded_trajectory.jsonl`
Canonical recording of a complete rollout session using the `rollouts.jsonl` dataset. Contains 18 trajectory entries (6 steps per environment) for fast CI testing.

This file enables deterministic, offline testing that is >1000x faster than live recording.

## CI Integration

For fast CI testing, set the environment variable to use the canonical recording:

```bash
export REWARD_KIT_PLAYBACK_FILE=examples/frozen_lake_mcp_complete/shared_data/recorded_trajectory.jsonl
pytest tests/test_record_and_playback_e2e.py
```

The test will automatically:
1. Use the standardized dataset from `rollouts.jsonl`
2. Skip live recording and use the canonical recording
3. Run playback-only validation in <0.1 seconds
4. Verify that the MCP server is properly hit during playback

## Local Development

When developing locally, you can:
1. Delete the `recorded_trajectory.jsonl` file to force live recording
2. Run tests normally to generate new recordings
3. Update the canonical recording as needed for new test scenarios

## Regenerating Canonical Recording

To update the canonical recording with new test data:

```bash
# Remove existing recording to force regeneration
rm examples/frozen_lake_mcp_complete/shared_data/recorded_trajectory.jsonl

# Run the test to generate new recording
pytest tests/test_record_and_playback_e2e.py::TestRecordAndPlaybackE2E::test_basic_record_and_playback

# The test will create a new canonical recording automatically
```
