"""
Advanced math reward function for evaluating mathematical answer correctness.

This module provides functions to evaluate the correctness of mathematical
answers by extracting numerical values from text using regex patterns and
comparing them with expected answers, with more detailed analysis.
"""

from typing import Dict, List, Any, Union

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult
from .math import (
    extract_numbers,
    compare_numbers,
)  # Assuming extract_numbers and compare_numbers remain in math.py


@reward_function
def advanced_math_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    original_messages: Union[List[Dict[str, Any]], List[Message]],
    relative_tolerance: float = 0.001,
    absolute_tolerance: float = 1e-8,
    match_all_answers: bool = False,
    require_units: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Advanced math reward function with more detailed analysis.

    This function extends the basic math_reward with more detailed analysis,
    including comparing all answers and reporting detailed metrics.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation messages (containing ground truth)
        relative_tolerance: Relative tolerance for numerical comparison
        absolute_tolerance: Absolute tolerance for numerical comparison
        match_all_answers: Whether all expected answers must be matched
        require_units: Whether to require matching units
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No generated messages provided",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="No generated messages provided",
                )
            },
        )

    gen_response_message = messages[-1]
    if isinstance(gen_response_message, Message):
        if (
            gen_response_message.role != "assistant"
            or not gen_response_message.content
        ):
            return EvaluateResult(
                score=0.0,
                reason="Last generated message not from assistant or has no content",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Last generated message not from assistant or has no content",
                    )
                },
            )
        gen_content = gen_response_message.content
    elif isinstance(gen_response_message, dict):
        if gen_response_message.get(
            "role"
        ) != "assistant" or not gen_response_message.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="Last generated message not from assistant or has no content",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Last generated message not from assistant or has no content",
                    )
                },
            )
        gen_content = gen_response_message.get("content", "")
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last generated message is of unknown type",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="Last generated message is of unknown type",
                )
            },
        )

    if not original_messages or len(original_messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No original messages provided",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="No original messages provided",
                )
            },
        )

    orig_response_message = original_messages[-1]
    if isinstance(orig_response_message, Message):
        if (
            orig_response_message.role != "assistant"
            or not orig_response_message.content
        ):
            return EvaluateResult(
                score=0.0,
                reason="Last original message not from assistant or has no content",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Last original message not from assistant or has no content",
                    )
                },
            )
        orig_content = orig_response_message.content
    elif isinstance(orig_response_message, dict):
        if orig_response_message.get(
            "role"
        ) != "assistant" or not orig_response_message.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="Last original message not from assistant or has no content",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Last original message not from assistant or has no content",
                    )
                },
            )
        orig_content = orig_response_message.get("content", "")
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last original message is of unknown type",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="Last original message is of unknown type",
                )
            },
        )

    if not gen_content or not orig_content:
        return EvaluateResult(
            score=0.0,
            reason="Empty message content in generated or original message",
            metrics={
                "error": MetricResult(
                    score=0.0, success=False, reason="Empty message content"
                )
            },
        )

    gen_answers = extract_numbers(gen_content)
    orig_answers = extract_numbers(orig_content)

    metrics: Dict[str, MetricResult] = {}
    num_orig_answers_extracted = len(orig_answers)
    orig_answer_texts = ', '.join([a[0] for a in orig_answers]) if orig_answers else 'None'
    reason_for_orig_extraction = f"Extracted {num_orig_answers_extracted} answers from original message: {orig_answer_texts}"
    metrics["extracted_original_answers"] = MetricResult(
        score=0.0, # Score for extraction itself is not the focus, success flag is.
        success=True if orig_answers else False,
        reason=reason_for_orig_extraction,
    )

    num_gen_answers_extracted = len(gen_answers)
    gen_answer_texts = ', '.join([a[0] for a in gen_answers]) if gen_answers else 'None'
    reason_for_gen_extraction = f"Extracted {num_gen_answers_extracted} answers from generated message: {gen_answer_texts}"
    metrics["extracted_generated_answers"] = MetricResult(
        score=0.0, # Score for extraction itself is not the focus, success flag is.
        success=True if gen_answers else False,
        reason=reason_for_gen_extraction,
    )

    if not gen_answers or not orig_answers:
        no_answer_reason = f"Could not extract answers from {'generated' if not gen_answers else 'original'} message"
        if not gen_answers and not orig_answers:
            no_answer_reason = (
                "Could not extract answers from generated or original message"
            )
        return EvaluateResult(
            score=0.0,
            reason=no_answer_reason,
            metrics={
                **metrics,
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason=no_answer_reason,
                ),
            },
        )

    match_details_list = []
    num_orig_answers = len(orig_answers)

    if num_orig_answers == 0:
        return EvaluateResult(
            score=0.0,
            reason="No original answers to extract for comparison.",
            metrics=metrics,
        )

    if match_all_answers:
        num_correctly_matched_orig = 0
        sum_correct_match_similarities = 0.0

        for i, (orig_text, orig_value) in enumerate(orig_answers):
            best_match_for_this_orig_is_correct = False
            best_sim_for_this_orig = -1.0

            for j, (gen_text, gen_value) in enumerate(gen_answers):
                if require_units:
                    orig_parts = orig_text.split()
                    gen_parts = gen_text.split()
                    orig_unit = (
                        orig_parts[-1]
                        if len(orig_parts) > 1
                        and not orig_parts[-1].replace(".", "", 1).isdigit()
                        else ""
                    )
                    gen_unit = (
                        gen_parts[-1]
                        if len(gen_parts) > 1
                        and not gen_parts[-1].replace(".", "", 1).isdigit()
                        else ""
                    )
                    if orig_unit != gen_unit:
                        continue

                is_match, similarity = compare_numbers(
                    orig_value,
                    gen_value,
                    relative_tolerance,
                    absolute_tolerance,
                )
                if is_match:  # Found a correct match for this original answer
                    best_match_for_this_orig_is_correct = True
                    sum_correct_match_similarities += (
                        similarity  # Should be 1.0 if is_match is True
                    )
                    match_details_list.append(
                        f"Correct match: Original '{orig_text}' ({orig_value}) vs Generated '{gen_text}' ({gen_value}), Sim: {similarity:.3f}"
                    )
                    break  # Move to next original answer

                if (
                    similarity > best_sim_for_this_orig
                ):  # Track best similarity even if not a "correct" match for details
                    best_sim_for_this_orig = similarity

            if best_match_for_this_orig_is_correct:
                num_correctly_matched_orig += 1
            else:
                match_details_list.append(
                    f"No correct match for Original '{orig_text}' ({orig_value}). Best sim found: {best_sim_for_this_orig:.3f}"
                )

        if num_correctly_matched_orig == num_orig_answers:
            final_score = (
                sum_correct_match_similarities / num_orig_answers
                if num_orig_answers > 0
                else 1.0
            )
            reason = f"All {num_orig_answers} original answers were correctly matched. Average similarity: {final_score:.3f}"
        else:
            final_score = 0.0
            reason = f"Not all original answers were correctly matched. Required: {num_orig_answers}, Correctly matched: {num_correctly_matched_orig}."

        metrics["match_summary"] = MetricResult(
            score=final_score, success=final_score == 1.0, reason=reason
        )

    else:  # Not match_all_answers (original best_match_any logic)
        best_overall_similarity = 0.0
        best_match_reason_overall = "No matching answer found"

        for i, (orig_text, orig_value) in enumerate(orig_answers):
            current_best_sim_for_orig = -1.0
            current_best_reason_for_orig = ""
            found_any_match_for_orig = False

            for j, (gen_text, gen_value) in enumerate(gen_answers):
                if require_units:
                    orig_parts = orig_text.split()
                    gen_parts = gen_text.split()
                    orig_unit = (
                        orig_parts[-1]
                        if len(orig_parts) > 1
                        and not orig_parts[-1].replace(".", "", 1).isdigit()
                        else ""
                    )
                    gen_unit = (
                        gen_parts[-1]
                        if len(gen_parts) > 1
                        and not gen_parts[-1].replace(".", "", 1).isdigit()
                        else ""
                    )
                    if orig_unit != gen_unit:
                        continue

                is_match, similarity = compare_numbers(
                    orig_value,
                    gen_value,
                    relative_tolerance,
                    absolute_tolerance,
                )
                if similarity > current_best_sim_for_orig:
                    current_best_sim_for_orig = similarity
                    current_best_reason_for_orig = f"Original '{orig_text}' ({orig_value}) vs Generated '{gen_text}' ({gen_value}), Sim: {similarity:.3f}"
                    found_any_match_for_orig = True

            if found_any_match_for_orig:
                match_details_list.append(current_best_reason_for_orig)
                if current_best_sim_for_orig > best_overall_similarity:
                    best_overall_similarity = current_best_sim_for_orig
                    best_match_reason_overall = current_best_reason_for_orig
            else:
                match_details_list.append(
                    f"No match found for Original '{orig_text}' ({orig_value})"
                )

        final_score = best_overall_similarity
        reason = best_match_reason_overall
        metrics["match_summary"] = MetricResult(
            score=final_score, success=final_score > 0.5, reason=reason
        )

    metrics["match_details"] = MetricResult(
        score=metrics["match_summary"].score,
        success=metrics["match_summary"].success,
        reason=(
            "\n".join(match_details_list)
            if match_details_list
            else "No match details available."
        ),
    )

    return EvaluateResult(score=final_score, reason=reason, metrics=metrics)
