"""
Multiple Choice Question (MCQ) reward function.

This module provides a reward function specifically for evaluating
answers to multiple-choice questions, where the answer is typically
a single letter (e.g., A, B, C, D, E).
"""
from typing import Dict, List, Any, Union, Optional, Tuple

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult
import re

def extract_mcq_option(text: str) -> List[Tuple[str, str]]:
    """
    Extracts MCQ options (A-E) from text.
    Prioritizes options in parentheses or brackets, or standalone letters.

    Args:
        text: The text to extract MCQ options from.

    Returns:
        A list of tuples, where each tuple contains the original matched
        string and the uppercase letter of the MCQ option.
        Returns an empty list if no MCQ option is confidently extracted.
    """
    mcq_answers: List[Tuple[str, str]] = []
    found_mcq_letters = set()

    # Pattern looks for A-E, in various formats: (A), [A], {A}, A. or standalone A.
    # Each alternative captures the full option string and the letter.
    # Order of alternatives matters for re.finditer if there could be overlaps.
    # (?i) for case-insensitivity.
    # Groupings:
    # 1: Full match for (A), 2: A
    # 3: Full match for [A], 4: A
    # 5: Full match for {A}, 6: A
    # 7: Full match for A.,  group 8: A
    # 9: Full match for standalone A, group 10: A
    # The outer group (1) captures the full matched alternative.
    regex_pattern = r"(?i)((\(([A-E])\))|(\[([A-E])\])|(\{([A-E])\})|(([A-E])\.(?!\w))|((?<![a-zA-Z0-9_])([A-E])(?![a-zA-Z0-9_])))"
    # For "A.": added (?!\w) to ensure "A.M." doesn't match "A."
    # For standalone "A": used (?<![a-zA-Z0-9_]) and (?![a-zA-Z0-9_]) to avoid matching A in APPLE.

    final_mcq_answers: List[Tuple[str, str]] = []
    processed_spans = [] # To handle overlapping matches from different parts of the regex if they occur

    # The regex is structured as (alt1_full(alt1_letter)|alt2_full(alt2_letter)|...)
    # Group indices for letters: 3, 5, 7, 9, 12
    # Group indices for full option text: 2, 4, 6, 8, 11
    # No, this is getting too complex. Let's simplify the regex and iterate.

    patterns = [
        r"(\(([A-E])\))",       # (A) - captures ((A), A)
        r"(\[([A-E])\])",       # [A] - captures ([A], A)
        r"(\{([A-E])\})",       # {A} - captures ({A}, A)
        r"((?<![a-zA-Z0-9_])([A-E])\.(?!\w))",   # A. (not preceded by word char) - captures (A., A)
        r"((?<![a-zA-Z0-9_])([A-E])(?![a-zA-Z0-9_]))" # Standalone A (not surrounded by word chars) - captures (A, A)
    ]

    # Collect all potential matches with their spans
    all_potential_matches = []
    for p_idx, p_str in enumerate(patterns):
        for match in re.finditer(p_str, text, re.IGNORECASE):
            option_text = match.group(1) # The full matched option, e.g. (A), A.
            letter = match.group(2)      # The letter itself
            all_potential_matches.append({
                "text": option_text,
                "letter": letter.upper(),
                "span": match.span(1),
                "priority": p_idx # To prefer more specific patterns like (A) over A
            })

    # Sort matches: by start position, then by pattern priority (lower index = higher priority), then by length (longer preferred)
    all_potential_matches.sort(key=lambda m: (m["span"][0], m["priority"], -(m["span"][1] - m["span"][0])))

    # Filter out overlapping matches, keeping the highest priority/longest one
    last_covered_end = -1
    for match_info in all_potential_matches:
        start, end = match_info["span"]
        if start >= last_covered_end: # Non-overlapping or starts after last one ended
            letter_upper = match_info["letter"]
            if letter_upper not in found_mcq_letters:
                final_mcq_answers.append((match_info["text"], letter_upper))
                found_mcq_letters.add(letter_upper)
            last_covered_end = end
            
    return final_mcq_answers


@reward_function
def multiple_choice_math_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    original_messages: Union[List[Dict[str, Any]], List[Message]],
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate multiple-choice answers in messages.

    Extracts MCQ options (A-E) from the last assistant message in
    both generated and original (ground truth) messages, then compares them.

    Args:
        messages: Generated conversation messages.
        original_messages: Original conversation messages (ground truth).
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
    """
    metrics: Dict[str, MetricResult] = {}

    if not messages or not original_messages:
        return EvaluateResult(
            score=0.0,
            reason="Missing messages or original messages",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="Missing messages or original messages",
                )
            },
        )

    gen_content = ""
    if messages and len(messages) > 0:
        gen_response_message = messages[-1]
        if isinstance(gen_response_message, Message) and gen_response_message.role == "assistant":
            gen_content = gen_response_message.content or ""
        elif isinstance(gen_response_message, dict) and gen_response_message.get("role") == "assistant":
            gen_content = gen_response_message.get("content", "")
    
    if not gen_content:
        metrics["error_generated_message"] = MetricResult(score=0.0, success=False, reason="Invalid generated message: Last message not from assistant or has no content.")
        return EvaluateResult(score=0.0, reason="Last generated message not from assistant or has no content.", metrics=metrics)

    orig_content = ""
    if original_messages and len(original_messages) > 0:
        orig_response_message = original_messages[-1]
        if isinstance(orig_response_message, Message) and orig_response_message.role == "assistant":
            orig_content = orig_response_message.content or ""
        elif isinstance(orig_response_message, dict) and orig_response_message.get("role") == "assistant":
            orig_content = orig_response_message.get("content", "")

    if not orig_content:
        metrics["error_original_message"] = MetricResult(score=0.0, success=False, reason="Invalid original message: Last message not from assistant or has no content.")
        return EvaluateResult(score=0.0, reason="Last original message not from assistant or has no content.", metrics=metrics)

    gen_mcq_options = extract_mcq_option(gen_content)
    orig_mcq_options = extract_mcq_option(orig_content)

    def format_extracted_mcq(items: List[Tuple[str, str]]) -> str:
        if not items: return "None"
        return ", ".join([f"'{i[0]}' ({i[1]})" for i in items])

    metrics["extracted_original_mcq"] = MetricResult(
        score=1.0 if orig_mcq_options else 0.0, 
        success=bool(orig_mcq_options),
        reason=f"Extracted from original: {format_extracted_mcq(orig_mcq_options)}"
    )
    metrics["extracted_generated_mcq"] = MetricResult(
        score=1.0 if gen_mcq_options else 0.0,
        success=bool(gen_mcq_options),
        reason=f"Extracted from generated: {format_extracted_mcq(gen_mcq_options)}"
    )

    if not orig_mcq_options:
        return EvaluateResult(score=0.0, reason="Could not extract MCQ option from original message (ground truth). Assumed not an MCQ.", metrics=metrics)
    
    if not gen_mcq_options:
        return EvaluateResult(score=0.0, reason="Could not extract MCQ option from generated message, but original message has an MCQ option.", metrics=metrics)

    # For simplicity, we'll compare the first extracted option if multiple are found.
    # Ideally, MCQs should have one clear answer.
    # If ground truth has multiple options extracted, it's ambiguous.
    if len(orig_mcq_options) > 1:
        metrics["ambiguous_original_mcq"] = MetricResult(score=0.0, success=False, reason=f"Original message has multiple MCQ options extracted: {format_extracted_mcq(orig_mcq_options)}")
        # We could penalize here, or pick the first one. For now, let's pick the first.
        # return EvaluateResult(score=0.0, reason="Original message (ground truth) has ambiguous MCQ options.", metrics=metrics)
    
    # If generated has multiple options extracted, it's ambiguous.
    if len(gen_mcq_options) > 1:
        metrics["ambiguous_generated_mcq"] = MetricResult(score=0.0, success=False, reason=f"Generated message has multiple MCQ options extracted: {format_extracted_mcq(gen_mcq_options)}")
        # Penalize for ambiguity if GT is specific
        if len(orig_mcq_options) == 1:
             return EvaluateResult(score=0.0, reason="Generated answer is ambiguous (multiple MCQ options) while ground truth is specific.", metrics=metrics)
        # If both are ambiguous, we could try to find a common element, but for now, let's simplify and compare first vs first.

    orig_answer_letter = orig_mcq_options[0][1] # ("(A)", "A") -> "A"
    gen_answer_letter = gen_mcq_options[0][1]

    is_match = orig_answer_letter == gen_answer_letter
    score = 1.0 if is_match else 0.0
    reason = f"Match: {is_match}. Gen: '{gen_mcq_options[0][0]}' ({gen_answer_letter}) vs Orig: '{orig_mcq_options[0][0]}' ({orig_answer_letter})"

    metrics["mcq_comparison"] = MetricResult(
        score=score,
        success=is_match,
        reason=reason
    )

    return EvaluateResult(score=score, reason=reason, metrics=metrics)
