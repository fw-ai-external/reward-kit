## III. Function Calling Example (`examples/function_calling_example/`)
*   **Core Reward Function(s):** `composite_function_call_reward`
*   **Status:**
    *   [ ] 0. Data Preparation & Cleaning:
        *   [ ] Define process to create/curate `examples/function_calling_example/dataset.jsonl`.
    *   [ ] 1. Dataset & Local Eval:
        *   [ ] Create `examples/function_calling_example/dataset.jsonl`
        *   [ ] Create `examples/function_calling_example/local_eval.py`
    *   [ ] 2. Fireworks Preview API:
        *   [ ] Create `examples/function_calling_example/fireworks_preview.py`
    *   [ ] 3. Fireworks Regeneration (Qwen3):
        *   [ ] Create `examples/function_calling_example/fireworks_regenerate.py`
    *   [ ] 4. TRL Integration (GRPO):
        *   [ ] Create `examples/function_calling_example/trl_grpo_integration.py` (investigate linearization or prompt-based approach)
    *   [ ] 5. End-to-End Integration Test:
        *   [ ] Add end-to-end script execution tests to `tests/test_readiness.py`
