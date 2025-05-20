import json
import re
from typing import Any, Dict, List, Optional

from reward_kit.models import Message  # Import Message model
from reward_kit.models import EvaluateResult, MetricResult
from reward_kit.reward_function import reward_function


@reward_function
def lean_prover_reward(
    messages: List[Message],  # Full conversation, model's response is messages[-1]
    ground_truth: Optional[str],  # This is the expected_answer (proof string)
    # statement is still expected via kwargs as per original logic
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates a Lean proof by analyzing the response for valid syntax, proof completion,
    and correctness based on the DeepSeek-Prover-V2 benchmark approach.

    Args:
        messages: List of conversation messages. The last message is the assistant's response.
        ground_truth: The expected proof string. Corresponds to 'expected_answer' in original kwargs.
        **kwargs: Must include 'statement' (str). Optional:
                  'lean_version' (str, default "4"), 'check_partial_progress' (bool, default True),
                  'verbose' (bool, default False).

    Returns:
        EvaluateResult with score and metrics
    """
    statement: Optional[str] = kwargs.get("statement")
    # expected_answer is now the ground_truth parameter
    expected_answer: Optional[str] = ground_truth
    # lean_version: str = kwargs.get("lean_version", "4") # lean_version is not used in this function's logic
    check_partial_progress: bool = kwargs.get("check_partial_progress", True)
    verbose: bool = kwargs.get("verbose", False)

    if not statement:
        return EvaluateResult(
            score=0.0,
            reason="Statement not provided in kwargs.",
            metrics={
                "error": MetricResult(
                    score=0.0, is_score_valid=False, reason="Statement not provided."
                )
            },
        )

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    response = messages[-1].content
    if not response:  # Check if content is empty string
        return EvaluateResult(
            score=0.0,
            reason="Assistant response content is empty.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Empty assistant response content.",
                )
            },
        )

    # Define patterns for Lean syntax validation
    patterns = {
        "theorem_def": r"theorem\s+\w+(\s*\{[^}]*\})?(\s*\([^)]*\))?\s*:=?",
        "lemma_def": r"lemma\s+\w+(\s*\{[^}]*\})?(\s*\([^)]*\))?\s*:=?",
        "example_def": r"example\s*(\{[^}]*\})?(\s*\([^)]*\))?\s*:=?",
        "by_tactic": r"by\s+\w+",
        "sorry": r"sorry",
        "admitted": r"admitted",
        "end_of_proof": r"(QED|qed|∎|#check)",
        "have_statement": r"have\s+\w+(\s*:\s*[^:=]+)?\s*:=",
        "apply_tactic": r"apply\s+[\w\.]+",
        "intro_tactic": r"intro\s+\w+",
        "rw_tactic": r"rw\s+[\[\]\w\s\.\,]+",
        "simp_tactic": r"simp(\s+[\[\]\w\s\.\,]+)?",
        "exact_tactic": r"exact\s+[\w\.]+",
        "calc_block": r"calc\s+",
    }

    # Check if it's a valid Lean proof attempt
    has_theorem_def = (
        bool(re.search(patterns["theorem_def"], response))
        or bool(re.search(patterns["lemma_def"], response))
        or bool(re.search(patterns["example_def"], response))
    )

    # Check for sorry/admitted (incomplete proof)
    has_sorry = bool(re.search(patterns["sorry"], response))
    has_admitted = bool(re.search(patterns["admitted"], response))

    # Check for proof completion indicators
    has_end_marker = bool(re.search(patterns["end_of_proof"], response))
    has_by_tactic = bool(re.search(patterns["by_tactic"], response))

    # Check for common proof tactics
    tactics_present = []
    tactics_count = 0
    for tactic_name in [
        "have_statement",
        "apply_tactic",
        "intro_tactic",
        "rw_tactic",
        "simp_tactic",
        "exact_tactic",
        "calc_block",
    ]:
        if bool(re.search(patterns[tactic_name], response)):
            tactics_present.append(tactic_name)
            tactics_count += len(re.findall(patterns[tactic_name], response))

    # Calculate basic score
    score = 0.0
    reason = "No valid Lean proof attempt"

    # Score 0: No valid Lean proof attempt
    if not has_theorem_def and tactics_count == 0:
        score = 0.0
        reason = "No valid Lean proof attempt"
    # Score 0.1-0.4: Has definition but incomplete or partial proof
    elif has_theorem_def and (has_sorry or has_admitted):
        # Partial credit based on how much of the proof was completed
        if check_partial_progress:
            # Scale score based on number of tactics used (up to 0.4)
            score = min(0.4, 0.1 + (tactics_count / 10) * 0.3)
            reason = f"Incomplete proof with {tactics_count} tactics"
        else:
            score = 0.1  # Only give minimal credit for incomplete proofs
            reason = "Incomplete proof (has sorry/admitted)"
    # Score 0.5-0.9: Has complete proof
    elif has_theorem_def and not (has_sorry or has_admitted):
        # Base score for complete proof
        score = 0.5
        reason = "Complete proof"

        # Add up to 0.4 more based on tactics complexity
        if tactics_count >= 5:
            score += 0.4
            reason = f"Complete proof with good complexity ({tactics_count} tactics)"
        else:
            score += (tactics_count / 5) * 0.4
            reason = f"Complete proof with {tactics_count} tactics"
    # Score 1.0: Perfect score when we have expected_answer to compare against
    if expected_answer and expected_answer.lower() in response.lower():
        score = 1.0
        reason = "Perfect match with expected proof"

    # Prepare metrics
    metrics = {}
    if verbose:
        metrics = {
            "syntax": MetricResult(
                score=float(has_theorem_def),
                is_score_valid=has_theorem_def,
                reason=(
                    "Has valid theorem definition"
                    if has_theorem_def
                    else "Missing theorem definition"
                ),
            ),
            "completeness": MetricResult(
                score=0.0 if has_sorry or has_admitted else 1.0,
                is_score_valid=not (has_sorry or has_admitted),
                reason=(
                    "Incomplete proof (has sorry/admitted)"
                    if has_sorry or has_admitted
                    else "Complete proof"
                ),
            ),
            "tactics": MetricResult(
                score=min(1.0, tactics_count / 10),
                is_score_valid=tactics_count > 0,  # Basic success if any tactics used
                reason=f"Used {tactics_count} tactics",
            ),
        }

        if expected_answer:
            expected_match_bool = expected_answer.lower() in response.lower()
            metrics["expected_match"] = MetricResult(
                score=1.0 if expected_match_bool else 0.0,
                is_score_valid=expected_match_bool,
                reason=(
                    "Matches expected proof"
                    if expected_match_bool
                    else "Doesn't match expected proof"
                ),
            )

    # Create and return result
    return EvaluateResult(
        score=score,
        reason=reason,  # Use the existing reason variable
        metrics=metrics,
    )


@reward_function
def deepseek_prover_v2_reward(
    messages: List[Message],  # Full conversation, model's response is messages[-1]
    ground_truth: Optional[str],  # This is the expected_proof
    # statement is still expected via kwargs
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates a Lean proof based on the DeepSeek-Prover-V2 methodology that
    focuses on subgoal decomposition and formal verification.

    Args:
        messages: List of conversation messages. The last message is the assistant's response.
        ground_truth: The expected proof string. Corresponds to 'expected_proof' in original kwargs.
        **kwargs: Must include 'statement' (str). Optional:
                  'check_subgoals' (bool, default True), 'verbose' (bool, default False).
    Returns:
        EvaluateResult with score and metrics
    """
    statement: Optional[str] = kwargs.get("statement")
    # expected_proof is now the ground_truth parameter
    expected_proof: Optional[str] = ground_truth
    check_subgoals: bool = kwargs.get("check_subgoals", True)
    verbose: bool = kwargs.get("verbose", False)

    if not statement:
        return EvaluateResult(
            score=0.0,
            reason="Statement not provided in kwargs for deepseek_prover_v2_reward.",
            metrics={
                "error": MetricResult(
                    score=0.0, is_score_valid=False, reason="Statement not provided."
                )
            },
        )

    # The model's response is in messages[-1].content.
    # lean_prover_reward will handle checking messages[-1].
    # No need for a separate check here if lean_prover_reward does it.

    # Prepare kwargs for lean_prover_reward.
    # The `ground_truth` for lean_prover_reward is `expected_proof`.
    lean_prover_kwargs_for_call = {
        "statement": statement,
        # "expected_answer" for lean_prover_reward is our expected_proof (now ground_truth for this func)
        # This will be passed as the ground_truth argument to lean_prover_reward directly.
        "check_partial_progress": True,  # Default from original call structure
        "verbose": verbose,
    }

    # Call the refactored lean_prover_reward.
    # messages (full convo) is passed as messages.
    # expected_proof (this function's ground_truth) is passed as ground_truth to lean_prover_reward.
    base_evaluate_result: EvaluateResult = lean_prover_reward(
        messages=messages, ground_truth=expected_proof, **lean_prover_kwargs_for_call
    )

    base_score = base_evaluate_result.score
    base_reason = base_evaluate_result.reason or "Formal proof evaluation"
    base_metrics = base_evaluate_result.metrics or {}
    top_level_reason = base_reason

    metrics = base_metrics.copy()

    # Specific patterns for DeepSeek-Prover-V2 subgoal approach
    subgoal_patterns = {
        "have_statement": r"have\s+(\w+)(\s*:\s*[^:=]+)?\s*:=",
        "suffices": r"suffices\s+(\w+)(\s*:\s*[^,]+)?\s*,",
        "let": r"let\s+(\w+)(\s*:\s*[^:=]+)?\s*:=",
        "decomposition_comment": r"(\/\*|\/\/)\s*(decomposing|breaking down|subgoal|step \d+)",
        "recursion": r"(recursion|induction|structural|recursive)",
    }

    # Analyze subgoal decomposition if requested
    # Need `response_content` from messages[-1].content for this part.
    # Ensure messages[-1] is valid before accessing content (already done by lean_prover_reward if it was called)
    # If lean_prover_reward returned due to invalid messages, base_score would be 0 and this part might not run or matter.
    response_content = ""
    if (
        messages
        and isinstance(messages[-1], Message)
        and messages[-1].role == "assistant"
        and messages[-1].content is not None
    ):
        response_content = messages[-1].content
    # If response_content is empty here, subgoal checks will yield 0, which is fine.

    final_score = base_score
    subgoal_count = 0
    hierarchy_depth = 0
    subgoal_score = 0
    hierarchy_score = 0

    if check_subgoals and response_content:  # Check response_content
        # Count subgoal patterns
        subgoal_count = 0
        for pattern_name, pattern in subgoal_patterns.items():
            subgoal_count += len(re.findall(pattern, response_content))

        # Detect hierarchical structure using indentation analysis
        lines = response_content.split("\n")
        max_indent = 0
        for line in lines:
            spaces = len(line) - len(line.lstrip(" "))
            if spaces > max_indent:
                max_indent = spaces

        # Calculate hierarchical depth (normalized to 0-1)
        hierarchy_depth = min(1.0, max_indent / 40) if max_indent > 0 else 0

        # Adjust score based on subgoal decomposition quality
        subgoal_score = min(0.3, (subgoal_count / 10) * 0.3)
        hierarchy_score = hierarchy_depth * 0.2

        # The DeepSeek-Prover-V2 approach should have good subgoal decomposition
        # Only apply this bonus if the base score is already decent
        if base_score >= 0.5:
            final_score = min(1.0, base_score + subgoal_score + hierarchy_score)
            # Update top_level_reason, as 'reason' might not be defined in this scope
            # if base_reason was used for top_level_reason.
            top_level_reason = f"{top_level_reason} with good subgoal decomposition"
        else:
            final_score = base_score

        # Add subgoal metrics
        subgoal_decomposition_score_normalized = (
            subgoal_score / 0.3 if subgoal_score > 0 else 0.0
        )
        metrics["subgoal_decomposition"] = MetricResult(
            score=min(
                1.0, subgoal_decomposition_score_normalized
            ),  # Ensure score is <= 1.0
            is_score_valid=subgoal_decomposition_score_normalized > 0.5,
            reason=f"Found {subgoal_count} subgoal patterns",
        )

        metrics["hierarchical_structure"] = MetricResult(
            score=hierarchy_depth,
            is_score_valid=hierarchy_depth
            > 0.5,  # Mark success if structure is reasonably deep
            reason=f"Hierarchical depth: {hierarchy_depth:.2f}",
        )
        # top_level_reason is already updated if base_score >= 0.5

    # Create and return result
    return EvaluateResult(
        score=final_score,
        reason=top_level_reason,
        metrics=metrics,
    )


@reward_function
def deepseek_huggingface_prover_benchmark(
    messages: List[Message],  # Full conversation, model's response is messages[-1]
    ground_truth: Dict[
        str, Any
    ],  # Expected to contain 'statement', and optionally 'dataset_item' or its components
    # Other specific args like dataset_name, check_for_answer, verbose can remain in kwargs
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates a Lean proof against the DeepSeek ProverBench dataset from Hugging Face.
    This reward function is specifically designed to work with the
    deepseek-ai/DeepSeek-ProverBench dataset.

    Args:
        messages: List of conversation messages. The last message is the assistant's response.
        ground_truth: A dictionary containing ground truth information. Expected keys:
                      'statement' (str): The theorem statement.
                      Optionally 'dataset_item' (dict): Pre-loaded dataset item.
                      Optionally 'expected_proof' (str): The reference proof.
                      Optionally 'answer' (str): A short answer if applicable.
        **kwargs: Optional: 'dataset_name' (str), 'check_for_answer' (bool), 'verbose' (bool).

    Returns:
        EvaluateResult with score and metrics
    """
    statement: Optional[str] = ground_truth.get("statement")
    dataset_item: Optional[Dict[str, Any]] = ground_truth.get("dataset_item")
    # Allow expected_proof and answer to be directly in ground_truth if dataset_item is not.
    expected_proof_from_gt: Optional[str] = ground_truth.get("expected_proof")
    answer_from_gt: Optional[str] = ground_truth.get("answer")

    dataset_name: str = kwargs.get("dataset_name", "deepseek-ai/DeepSeek-ProverBench")
    check_for_answer: bool = kwargs.get("check_for_answer", True)
    verbose: bool = kwargs.get("verbose", False)

    if not statement:
        return EvaluateResult(
            score=0.0,
            reason="Statement not found in ground_truth dict for HuggingFace benchmark.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Statement not provided in ground_truth.",
                )
            },
        )

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    response = messages[-1].content  # This is the model's proof attempt
    if not response:
        return EvaluateResult(
            score=0.0,
            reason="Assistant response content is empty for HuggingFace benchmark.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Empty assistant response content.",
                )
            },
        )

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to use this reward function. "
            "Please install it with 'pip install datasets'."
        )

    # Initial metrics
    metrics = {}

    # Load dataset item if not provided
    if dataset_item is None:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)

        # Find matching problem by statement (if exact match not found, we'll use fuzzy matching)
        matched_item = None
        for split in dataset.keys():
            for item in dataset[split]:
                if statement.strip() in item.get("statement", ""):
                    matched_item = item
                    break
            if matched_item:
                break

        if not matched_item:
            # Try fuzzy matching if exact match not found
            from difflib import SequenceMatcher

            best_ratio = 0
            matched_ratio = 0.0  # Ensure float

            for split in dataset.keys():
                for item in dataset[split]:
                    ratio = SequenceMatcher(
                        None, statement.strip(), item.get("statement", "")
                    ).ratio()
                    if ratio > best_ratio and ratio > 0.7:  # 70% similarity threshold
                        best_ratio = ratio
                        matched_item = item
                        matched_ratio = ratio

            if not matched_item:
                return EvaluateResult(
                    score=0.0,
                    reason="No matching problem found in the dataset",
                    metrics={
                        "dataset_match": MetricResult(
                            score=0.0,
                            is_score_valid=False,
                            reason="No matching problem found in the dataset",
                        )
                    },
                )

            # Add fuzzy match info to metrics
            metrics["dataset_match"] = MetricResult(
                score=matched_ratio,
                is_score_valid=matched_ratio
                > 0.7,  # Success if similarity is above threshold
                reason=f"Found similar problem with {matched_ratio:.2f} similarity",
            )
        else:
            # Add exact match info to metrics
            metrics["dataset_match"] = MetricResult(
                score=1.0, is_score_valid=True, reason="Found exact match in dataset"
            )

        dataset_item = matched_item

    # Extract expected proof if available from dataset_item or directly from ground_truth
    expected_proof = expected_proof_from_gt  # Prioritize direct key from ground_truth
    reference_solution = None
    if dataset_item:
        if not expected_proof:  # If not in ground_truth directly, try from dataset_item
            expected_proof = dataset_item.get("expected_proof", None)
        reference_solution = dataset_item.get("reference_solution", None)

    # Use the expected proof or reference solution if available
    proof_reference = expected_proof or reference_solution

    # Check for the answer/solution if required
    current_top_level_reason = "Evaluation against DeepSeek ProverBench dataset."
    # Use answer_from_gt if available, otherwise try from dataset_item
    answer_to_check = answer_from_gt
    if not answer_to_check and dataset_item:
        answer_to_check = dataset_item.get("answer")

    if check_for_answer and answer_to_check:
        expected_answer_str = str(answer_to_check)
        answer_found = expected_answer_str in response

        if not answer_found:
            metrics["answer_match"] = MetricResult(
                score=0.0,
                is_score_valid=False,
                reason=f"Expected answer '{expected_answer_str}' not found in response",
            )
            return EvaluateResult(
                score=0.2,
                reason=f"Expected answer '{expected_answer_str}' not found.",
                metrics=metrics,
            )
        else:
            metrics["answer_match"] = MetricResult(
                score=1.0,
                is_score_valid=True,
                reason="Expected answer found in response",
            )
            current_top_level_reason += " Expected answer found."

    # Use the deepseek_prover_v2_reward function for evaluation
    # messages (full convo) is passed as messages.
    # proof_reference (derived) is passed as ground_truth to deepseek_prover_v2_reward.
    deepseek_kwargs_for_call = {
        "statement": statement,
        # "expected_proof" for deepseek_prover_v2_reward is our proof_reference
        # This will be passed as the ground_truth argument to deepseek_prover_v2_reward.
        "check_subgoals": True,  # Default from original call structure
        "verbose": verbose,
    }
    eval_result_from_deepseek: EvaluateResult = deepseek_prover_v2_reward(
        messages=messages, ground_truth=proof_reference, **deepseek_kwargs_for_call
    )

    result_score = eval_result_from_deepseek.score
    result_reason = eval_result_from_deepseek.reason
    result_metrics = eval_result_from_deepseek.metrics or {}

    # Combine metrics
    combined_metrics = {**metrics, **result_metrics}

    if result_reason and result_reason not in current_top_level_reason:
        current_top_level_reason += f" Sub-evaluation: {result_reason}"

    # Add dataset information as additional metrics
    if verbose:
        combined_metrics["dataset_info"] = MetricResult(
            score=1.0,  # Not an evaluative score
            is_score_valid=True,  # Informational metric
            reason=json.dumps(
                {
                    "id": dataset_item.get("id", ""),
                    "has_expected_proof": expected_proof is not None,
                    "has_reference_solution": reference_solution is not None,
                    "has_answer": "answer" in dataset_item,
                }
            ),
        )

    # Create and return final result
    return EvaluateResult(
        score=result_score, reason=current_top_level_reason, metrics=combined_metrics
    )
