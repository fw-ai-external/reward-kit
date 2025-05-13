# Reproducing BFCL as an Evaluation with Agent v2 Stack

## Objective

Reproduce the Berkeley Function Call Leaderboard (BFCL) evaluation using the Reward Kit agent v2 stack to evaluate LLM agents on multi-turn tool-use tasks.

## Current Framework Status (Agent v2 for BFCL)

The core Agent v2 framework for BFCL evaluation is operational:
*   **`BFCLSimAPIResource`**: Wraps BFCL environments and infers tool schemas.
*   **Dataset Conversion**: `scripts/convert_bfcl_dataset.py` generates YAML task definitions from original BFCL data (currently for `multi_turn_base_*` tasks).
*   **Orchestrator**: `reward_kit/agent_v2/orchestrator.py` correctly handles multi-turn interactions, including an inner loop for multiple tool calls per user turn, and passes necessary data to the reward function.
*   **Task Definitions**: YAML files in `evaluations/bfcl/tasks/` are correctly structured.
*   **`bfcl_reward.py`**: Receives ground truth and model history; current implementation has known deviations from original BFCL checking logic (see Next Steps).

The system can run BFCL tasks end-to-end. Observed low scores are due to a combination of LLM agent reasoning errors and discrepancies between `bfcl_reward.py` and the original BFCL evaluation criteria.

## Task Verification Summary

This table tracks the status of BFCL tasks run with `gpt-4.1-2025-04-14` and the Agent v2 framework.

| Task ID             | Framework Status | Agent Performance (Current `bfcl_reward.py`) | Date Verified | Notes (Agent/Reward Discrepancies)                                                                                                |
|---------------------|------------------|----------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `multi_turn_base_0` | ✅ Verified      | ❌ Failed (Score 0.0)                        | 2025-05-13    | Func calls match (4/4) after `sort` fix. State match fails (CWD/pathing). Original BFCL: per-turn state check.                   |
| `multi_turn_base_1` | ✅ Verified      | ❌ Failed (Score 0.0)                        | 2025-05-13    | Func match 3/4 (Turn 2 pathing). State match failed. Original BFCL: per-turn state check.                                         |
| `multi_turn_base_2` | ✅ Verified      | ❌ Failed (Score 0.0)                        | 2025-05-13    | Func match 4/5 (Turn 1 `echo` content). State match failed. Original BFCL: per-turn state check, execution result check.        |
| `multi_turn_base_3` | ✅ Verified      | ❌ Failed (Score 0.0)                        | 2025-05-13    | Func match 2/2. State match failed (dir content order). Original BFCL: per-turn state check.                                    |
| `multi_turn_base_4` | ✅ Verified      | ❌ Failed (Score 0.0)                        | 2025-05-13    | Func match 2/3 (Turn 2 `post_tweet` args). State match failed. Original BFCL: per-turn state check, execution result check. |

*Framework Status*: Indicates if the Agent v2 framework executed the task.
*Agent Performance*: Current score from `bfcl_reward.py`. Failures highlight areas for reward logic refinement or agent improvement.

## Understanding Original BFCL Evaluation (`multi_turn_checker.py`)

Analysis of the original `references/verifiers/verifiers/berkeley-function-call-leaderboard/bfcl/eval_checker/multi_turn_eval/multi_turn_checker.py` reveals:
*   **Overall Evaluation:** Binary pass/fail (`valid: True/False`).
*   **Per-Turn Checks:** The checker validates state and tool call outcomes *after each turn*. A failure in any turn leads to an overall task failure.
*   **State Check (`state_checker`):** Requires a strict, attribute-by-attribute match of all public attributes of environment instances against the ground truth state for that turn.
*   **Response Check (`response_checker`):** Compares the *execution results* (string outputs) of tool calls. It verifies if the ground truth execution results for the *current turn* are an unordered subsequence of the model's *accumulated* execution results from all turns up to and including the current one.
*   **Tool Call Trajectory:** The exact sequence or definitions of `tool_name(arg=value)` calls made by the agent are *not* directly compared (the `method_invoke_order_checker` is commented out).

## Urgent: Orchestrator Issues

**1. Conversation History Truncation (Highest Priority):**
    *   **Observation:** Logs (e.g., `/tmp/bfcl_eval_multi_turn_base_0.log`) indicate that the `Orchestrator` might be sending only the most recent tool call result to the LLM, instead of the complete conversation history. Example:
        ```
        DEBUG:Orchestrator.multi_turn_base_0:Calling OpenAI: model=gpt-4.1-2025-04-14, messages=[
          {
            "tool_call_id": "call_fABkqPmylhZkgmpotY8k3xp3",
            "role": "tool",
            "name": "mv",
            "content": "{\"error\": \"mv: no path allowed in destination. Only file name and folder name is supported for this operation.\"}"
          }
        ]
        ```
    *   **Impact:** This severely limits the LLM's context, leading to poor performance and incorrect evaluation data.
    *   **Action:** Investigate and fix `reward_kit/agent_v2/orchestrator.py` to ensure the full conversation history (user prompts, assistant messages, all tool calls, and all tool results from previous turns) is maintained and passed to the LLM in subsequent turns.
    *   **Status (2025-05-13):** Confirmed via enhanced logging (`messages_FULL_HISTORY`) that the orchestrator *is* correctly accumulating and preparing the full conversation history for the API call. The initial concern about truncation was due to a summarized debug log. **This issue is resolved.**

## Investigation: State Matching Logic in `bfcl_reward.py` (2025-05-13)

**Objective:** Thoroughly review and compare the state representation and comparison logic in `reward_kit` against the original BFCL `multi_turn_checker.py` to identify and address discrepancies that might be causing unexpectedly low scores.

**Phase 1: Understanding Nuances of State Representation and Comparison**
1.  **Deep Dive into Original BFCL State Checking (`multi_turn_checker.py`):**
    *   Re-analyze `state_checker` and `_compare_instances`.
    *   Focus: Handling of attribute types, reliance on `__eq__`, representation of file system states (CWD, contents), comparison of complex objects.
2.  **Analyze Current Reward Kit State Serialization and Comparison:**
    *   Review `BFCLSimAPIResource.get_comparable_state()`: How are attributes serialized (esp. complex ones like `Directory` objects)? Is serialization deterministic?
    *   Review `bfcl_reward.compare_comparable_states()`: How does its key-checking and direct dict comparison align with original attribute-by-attribute checks?
3.  **Examine Key BFCL Environment Classes (e.g., `GorillaFileSystem`):**
    *   Read source of `references/verifiers/verifiers/envs/bfcl_envs/gorilla_file_system.py`.
    *   Focus: State-defining attributes, custom `__eq__` methods, CWD management, representation of file/directory contents.

**Phase 2: Identifying Potential Discrepancies and Forming Hypotheses**
*   **Serialization Mismatches:** Does `str(obj)` lose info or differ from original `__eq__`?
*   **Order Sensitivity:** For lists or dicts (e.g., directory contents), does our logic match original BFCL's sensitivity/insensitivity to order?
*   **Handling of Extra/Missing Items:** How do both systems handle extra/missing attributes or files within state objects?
*   **CWD Representation:** Is CWD consistently represented and compared?
*   **String Representations of Complex Objects:** Are `str()` representations canonical, or could they cause flaky comparisons (e.g., due to memory addresses)?

**Phase 3: Proposing and Implementing Refinements**
1.  **Detailed Report:** Document findings on divergences or ambiguities.
2.  **Suggestions for Refinement:**
    *   Adjust `BFCLSimAPIResource.get_comparable_state()` for more canonical/faithful state representation.
    *   Make `bfcl_reward.compare_comparable_states()` more nuanced if needed.
    *   Goal: Ensure our comparison aligns with original BFCL's equality definition for environment instances.
3.  **Implement Per-Turn Checks:** This remains a high priority for full alignment, using the refined state comparison method at each turn.

**Initial Actions for Investigation (Phase 1):**
1.  Read `reward_kit/rewards/bfcl_reward.py` (focus: `compare_comparable_states`).
2.  Read `reward_kit/agent_v2/resources/bfcl_sim_api_resource.py` (focus: `get_comparable_state`).
3.  Read `state_checker` & `_compare_instances` in original BFCL `multi_turn_checker.py`.
4.  Read `GorillaFileSystem` class source from original BFCL.

## Next Steps: Aligning `bfcl_reward.py` with Original BFCL Logic

The primary goal is to modify `bfcl_reward.py` to accurately reflect the original BFCL evaluation. This should be addressed *after* the orchestrator's conversation history management is confirmed to be correct.

1.  **Implement Per-Turn Simulation and Checking in `bfcl_reward.py` (High Priority):**
    *   The Orchestrator passes the full `model_history` and `ground_truth_function_calls` (which is a list of lists, per turn) and the *final* `ground_truth_comparable_state`.
    *   `bfcl_reward.py` needs to:
        *   **Simulate GT Turns:** Iterate through `ground_truth_function_calls` turn by turn. For each turn:
            *   Instantiate a fresh environment based on the task's `initial_config`.
            *   Execute all GT calls *up to and including the current turn*.
            *   Capture the environment state (`gt_turn_state`) and the execution results of calls made *in the current GT turn* (`gt_turn_execution_results`).
        *   **Simulate Model Turns:** Iterate through the `model_history` turn by turn (aligning with user messages). For each turn:
            *   Instantiate a fresh environment based on `initial_config`.
            *   Execute all model tool calls from `model_history` *up to and including the current turn*.
            *   Capture the environment state (`model_turn_state`) and accumulate all model execution results so far (`model_accumulated_execution_results`).
        *   **Perform Per-Turn Checks:**
            *   **State Check:** Compare `model_turn_state` with `gt_turn_state` using a strict, attribute-by-attribute comparison (mirroring original `state_checker`). If mismatch, task fails.
            *   **Response Check:** Check if `gt_turn_execution_results` is an unordered subsequence of `model_accumulated_execution_results` (mirroring original `response_checker`). If mismatch, task fails.
    *   **Note:** This per-turn simulation within the reward function is complex. An alternative is modifying the Orchestrator to provide per-turn states and results, but that's a larger change.

2.  **Refine Scoring Logic:**
    *   **Binary Outcome:** The primary score should likely be binary (1.0 for pass, 0.0 for fail) based on whether all per-turn state and response checks pass.
    *   **Metrics:** Continue to provide detailed metrics (e.g., which turn failed, state diffs, response diffs) for diagnostics, even if the main score is binary. The current `format_check` can remain as a separate, secondary metric.

3.  **Address State Comparison Nuances:**
    *   For file system state, ensure the comparison of directory contents (dictionaries) is robust to key order if the original BFCL's string-based state comparison was also order-sensitive. If order doesn't matter, make the comparison order-agnostic for directory listings.

4.  **Deprecate/Remove Stricter Function Call Definition Matching:**
    *   The current `_is_subsequence_unordered` comparing parsed call *definitions* and `_are_function_calls_equivalent` will likely be replaced or heavily modified by the new response-based check.

## Other Considerations (Agent Performance & Framework)

*   **Agent Reasoning:** Continue to monitor agent performance on CWD management, pathing, and verbosity. These are separate from reward logic but impact overall success.
*   **Expand Task Coverage:** Once the reward function is aligned, consider converting and testing other BFCL task categories.
*   **Documentation:** Update all relevant documentation post-changes.
