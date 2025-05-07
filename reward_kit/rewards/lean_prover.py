import re
import json
from typing import Dict, Optional, Any

from reward_kit.reward_function import reward_function
from reward_kit.models import RewardOutput, MetricRewardOutput


@reward_function
def lean_prover_reward(
    response: str,
    statement: str,
    expected_answer: Optional[str] = None,
    lean_version: str = "4",
    check_partial_progress: bool = True,
    verbose: bool = False,
) -> RewardOutput:
    """
    Evaluates a Lean proof by analyzing the response for valid syntax, proof completion,
    and correctness based on the DeepSeek-Prover-V2 benchmark approach.

    Args:
        response: The model response containing a Lean proof
        statement: The mathematical statement or theorem to be proven
        expected_answer: The expected answer/proof structure (optional)
        lean_version: Lean version ("4" for Lean 4, "3" for Lean 3)
        check_partial_progress: Whether to give partial credit for partial proofs
        verbose: Whether to include detailed scoring breakdown

    Returns:
        RewardOutput with score and metrics
    """
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
            reason = (
                f"Complete proof with good complexity ({tactics_count} tactics)"
            )
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
            "syntax": MetricRewardOutput(
                score=float(has_theorem_def),
                reason=(
                    "Has valid theorem definition"
                    if has_theorem_def
                    else "Missing theorem definition"
                ),
            ),
            "completeness": MetricRewardOutput(
                score=0.0 if has_sorry or has_admitted else 1.0,
                reason=(
                    "Incomplete proof (has sorry/admitted)"
                    if has_sorry or has_admitted
                    else "Complete proof"
                ),
            ),
            "tactics": MetricRewardOutput(
                score=min(1.0, tactics_count / 10),
                reason=f"Used {tactics_count} tactics",
            ),
        }

        if expected_answer:
            metrics["expected_match"] = MetricRewardOutput(
                score=(
                    1.0 if expected_answer.lower() in response.lower() else 0.0
                ),
                reason=(
                    "Matches expected proof"
                    if expected_answer.lower() in response.lower()
                    else "Doesn't match expected proof"
                ),
            )

    # Create and return result
    return RewardOutput(
        score=score,
        metrics=metrics,
    )


@reward_function
def deepseek_prover_v2_reward(
    response: str,
    statement: str,
    expected_proof: Optional[str] = None,
    check_subgoals: bool = True,
    verbose: bool = False,
) -> RewardOutput:
    """
    Evaluates a Lean proof based on the DeepSeek-Prover-V2 methodology that
    focuses on subgoal decomposition and formal verification.

    Args:
        response: The model response containing a Lean proof
        statement: The mathematical statement or theorem to be proven
        expected_proof: The expected proof (used for exact matching if provided)
        check_subgoals: Whether to evaluate the quality of subgoal decomposition
        verbose: Whether to include detailed scoring breakdown

    Returns:
        RewardOutput with score and metrics
    """
    # We'll use the base lean_prover_reward for initial evaluation
    base_result = lean_prover_reward(
        response=response,
        statement=statement,
        expected_answer=expected_proof,
        lean_version="4",  # DeepSeek-Prover-V2 uses Lean 4
        check_partial_progress=True,
        verbose=verbose,
    )

    base_score = base_result.score
    # Get reason from metrics if available
    reason = "Formal proof evaluation"

    # Initialize metrics from base result
    metrics = base_result.metrics.copy() if base_result.metrics else {}

    # Specific patterns for DeepSeek-Prover-V2 subgoal approach
    subgoal_patterns = {
        "have_statement": r"have\s+(\w+)(\s*:\s*[^:=]+)?\s*:=",
        "suffices": r"suffices\s+(\w+)(\s*:\s*[^,]+)?\s*,",
        "let": r"let\s+(\w+)(\s*:\s*[^:=]+)?\s*:=",
        "decomposition_comment": r"(\/\*|\/\/)\s*(decomposing|breaking down|subgoal|step \d+)",
        "recursion": r"(recursion|induction|structural|recursive)",
    }

    # Analyze subgoal decomposition if requested
    final_score = base_score
    subgoal_count = 0
    hierarchy_depth = 0
    subgoal_score = 0
    hierarchy_score = 0

    if check_subgoals:
        # Count subgoal patterns
        subgoal_count = 0
        for pattern_name, pattern in subgoal_patterns.items():
            subgoal_count += len(re.findall(pattern, response))

        # Detect hierarchical structure using indentation analysis
        lines = response.split("\n")
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
            reason = f"{reason} with good subgoal decomposition"
        else:
            final_score = base_score

        # Add subgoal metrics
        metrics["subgoal_decomposition"] = MetricRewardOutput(
            score=subgoal_score / 0.3,  # Normalize to 0-1 range
            reason=f"Found {subgoal_count} subgoal patterns",
        )

        metrics["hierarchical_structure"] = MetricRewardOutput(
            score=hierarchy_depth,
            reason=f"Hierarchical depth: {hierarchy_depth:.2f}",
        )

    # Create and return result
    return RewardOutput(
        score=final_score,
        metrics=metrics,
    )


@reward_function
def deepseek_huggingface_prover_benchmark(
    response: str,
    statement: str,
    dataset_item: Optional[Dict[str, Any]] = None,
    dataset_name: str = "deepseek-ai/DeepSeek-ProverBench",
    check_for_answer: bool = True,
    verbose: bool = False,
) -> RewardOutput:
    """
    Evaluates a Lean proof against the DeepSeek ProverBench dataset from Hugging Face.
    This reward function is specifically designed to work with the
    deepseek-ai/DeepSeek-ProverBench dataset.

    Args:
        response: The model response containing a Lean proof
        statement: The mathematical statement or theorem to be proven
        dataset_item: The dataset item from the HuggingFace dataset, if already loaded
        dataset_name: The name of the HuggingFace dataset (default: deepseek-ai/DeepSeek-ProverBench)
        check_for_answer: Whether to check for the answer in the response
        verbose: Whether to include detailed scoring breakdown

    Returns:
        RewardOutput with score and metrics
    """
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
            matched_ratio = 0

            for split in dataset.keys():
                for item in dataset[split]:
                    ratio = SequenceMatcher(
                        None, statement.strip(), item.get("statement", "")
                    ).ratio()
                    if (
                        ratio > best_ratio and ratio > 0.7
                    ):  # 70% similarity threshold
                        best_ratio = ratio
                        matched_item = item
                        matched_ratio = ratio

            if not matched_item:
                return RewardOutput(
                    score=0.0,
                    metrics={
                        "dataset_match": MetricRewardOutput(
                            score=0.0,
                            reason="No matching problem found in the dataset",
                        )
                    },
                )

            # Add fuzzy match info to metrics
            metrics["dataset_match"] = MetricRewardOutput(
                score=matched_ratio,
                reason=f"Found similar problem with {matched_ratio:.2f} similarity",
            )
        else:
            # Add exact match info to metrics
            metrics["dataset_match"] = MetricRewardOutput(
                score=1.0, reason="Found exact match in dataset"
            )

        dataset_item = matched_item

    # Extract expected proof if available
    expected_proof = dataset_item.get("expected_proof", None)
    reference_solution = dataset_item.get("reference_solution", None)

    # Use the expected proof or reference solution if available
    proof_reference = expected_proof or reference_solution

    # Check for the answer/solution if required
    if check_for_answer and dataset_item.get("answer", None):
        expected_answer = str(dataset_item["answer"])
        # Look for the answer in the response
        answer_found = expected_answer in response

        # If answer is provided but not found in the response, penalize score
        if not answer_found:
            metrics["answer_match"] = MetricRewardOutput(
                score=0.0,
                reason=f"Expected answer '{expected_answer}' not found in response",
            )
            return RewardOutput(score=0.2, metrics=metrics)
        else:
            metrics["answer_match"] = MetricRewardOutput(
                score=1.0, reason="Expected answer found in response"
            )

    # Use the deepseek_prover_v2_reward function for evaluation
    result = deepseek_prover_v2_reward(
        response=response,
        statement=statement,
        expected_proof=proof_reference,
        check_subgoals=True,
        verbose=verbose,
    )

    # Combine metrics
    combined_metrics = (
        {**metrics, **result.metrics} if result.metrics else metrics
    )

    # Add dataset information as additional metrics
    if verbose:
        combined_metrics["dataset_info"] = MetricRewardOutput(
            score=1.0,  # Not an evaluative score
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
    return RewardOutput(score=result.score, metrics=combined_metrics)
