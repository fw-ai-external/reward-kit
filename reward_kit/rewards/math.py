"""
Math reward function for evaluating mathematical answer correctness.

This module provides functions to evaluate the correctness of mathematical
answers by extracting numerical values from text using regex patterns and
comparing them with expected answers.
"""

from typing import Dict, List, Tuple, Any, Union
import re
import math
from ..typed_interface import reward_function

# RewardOutput and MetricRewardOutput will be replaced by EvaluateResult and MetricResult
from ..models import Message, EvaluateResult, MetricResult


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    """
    Extract numbers from text using various formats, including LaTeX notation.

    This function extracts numbers in different formats:
    - Plain numbers (integer and float): "42", "3.14"
    - Scientific notation: "1.2e-5", "6.022e23"
    - Fractions: "1/2", "3/4"
    - With units: "42 km", "3.14 meters"
    - LaTeX notation:
      - Boxed values: "$\\boxed{42}$"
      - Fractions: "$\\frac{3}{4}$"
      - Scientific notation: "$3 \\times 10^8$"

    Args:
        text: The text to extract numbers from

    Returns:
        List of tuples containing (original_text, normalized_value)
    """

    # Order of patterns is important to avoid mis-extractions.
    # More specific patterns should come before more general ones.
    # We will collect all potential matches with their start and end positions,
    # then filter out overlapping matches, prioritizing longer/more specific ones.

    potential_matches = []

    # 0. LaTeX expressions (processed recursively to handle nested structures)
    # We'll find all $...$ and $$...$$ blocks and process their contents.
    # This needs to be done carefully to avoid double-counting if a non-LaTeX regex
    # also matches something inside a LaTeX block that we've already parsed.
    # For simplicity in this pass, we'll extract from LaTeX first, then run other regexes
    # on the *original* text, and then de-duplicate based on span.

    for latex_match in re.finditer(r"\$\$(.*?)\$\$|\$(.*?)\$", text, re.DOTALL):
        content = (
            latex_match.group(1)
            if latex_match.group(1) is not None
            else latex_match.group(2)
        )
        offset = (
            latex_match.start(1)
            if latex_match.group(1) is not None
            else latex_match.start(2)
        )

        # a. Boxed LaTeX values (e.g., \boxed{123}, \boxed{1.23}, \boxed{1/2}, \boxed{1.2 \times 10^3})
        # This regex tries to capture the content of \boxed{}
        # We need to handle different number types within \boxed{}
        for m in re.finditer(r"\\boxed\{([^}]*)\}", content):
            boxed_content = m.group(1)
            # Try to parse common forms within boxed content:
            # Simple number: \boxed{123}, \boxed{-1.23}
            simple_num_match = re.fullmatch(
                r"\s*(-?\d+(?:\.\d+)?)\s*", boxed_content
            )
            if simple_num_match:
                val_str = simple_num_match.group(1)
                potential_matches.append(
                    {
                        "text": val_str,
                        "value": float(val_str),
                        "span": (m.start(1) + offset, m.end(1) + offset),
                        "type": "latex_boxed_simple",
                    }
                )
                continue  # Matched as simple boxed number

            # Fraction: \boxed{\frac{1}{2}}
            frac_match = re.fullmatch(
                r"\s*\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}\s*",
                boxed_content,
            )
            if frac_match:
                num_str, den_str = frac_match.group(1), frac_match.group(2)
                try:
                    val = float(num_str) / float(den_str)
                    potential_matches.append(
                        {
                            "text": f"{num_str}/{den_str}",
                            "value": val,
                            "span": (m.start(1) + offset, m.end(1) + offset),
                            "type": "latex_boxed_frac",
                        }
                    )
                    continue
                except (ValueError, ZeroDivisionError):
                    pass

            # Scientific: \boxed{1.2 \times 10^3}
            sci_match = re.fullmatch(
                r"\s*(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{(?:-?\d+)\}\s*",
                boxed_content,
            )  # Simplified exponent for now
            if sci_match:  # Placeholder, needs full sci parsing
                # This part needs robust parsing similar to non-LaTeX scientific notation
                # For now, let's assume it's caught by general LaTeX number extraction if not here
                pass

        # b. LaTeX fractions (e.g., \frac{3}{4})
        for m in re.finditer(
            r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", content
        ):
            num_str, den_str = m.group(1), m.group(2)
            try:
                val = float(num_str) / float(den_str)
                potential_matches.append(
                    {
                        "text": f"{num_str}/{den_str}",
                        "value": val,
                        "span": (m.start(0) + offset, m.end(0) + offset),
                        "type": "latex_frac",
                    }
                )
            except (ValueError, ZeroDivisionError):
                pass

        # c. LaTeX scientific notation (e.g., 3 \times 10^8)
        # Pattern for base \times 10^{exponent} with optional braces for exponent
        latex_sci_pattern = (
            r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^(?:\{)?(-?\d+(?:\.\d+)?)(?:\})?"
        )
        for m in re.finditer(latex_sci_pattern, content):
            try:
                base_str, exp_str = m.group(1), m.group(2)
                base_val = float(base_str)
                exp_val = float(exp_str)
                value = base_val * (10**exp_val)
                # Constructing original text representation
                orig_text = f"{base_str} \\times 10^{{{exp_str}}}"  # Keep consistent representation
                if (
                    "{" not in exp_str
                    and "}" not in exp_str
                    and "^" in m.group(0)
                    and "{" not in m.group(0)[m.group(0).find("^") :]
                ):  # Heuristic for unbraced exponent
                    orig_text = f"{base_str} \\times 10^{exp_str}"
                potential_matches.append(
                    {
                        "text": orig_text,
                        "value": value,
                        "span": (m.start(0) + offset, m.end(0) + offset),
                        "type": "latex_sci",
                    }
                )
            except ValueError:
                pass

        # d. General numbers within LaTeX (could be integers or decimals)
        for m in re.finditer(r"(-?\d+(?:\.\d+)?)", content):
            val_str = m.group(1)
            # Avoid re-matching parts of fractions or sci-notation already handled if possible (tricky)
            potential_matches.append(
                {
                    "text": val_str,
                    "value": float(val_str),
                    "span": (m.start(1) + offset, m.end(1) + offset),
                    "type": "latex_num",
                }
            )

    # 1. Scientific notation (e.g., 1.2e-5, 6.022E23) - with optional units
    # Ensure it's not part of a word, e.g. "evaluate"
    sci_pattern = (
        r"(?<![a-zA-Z0-9_])(-?\d+\.?\d*[eE][-+]?\d+)(?:\s*([a-zA-Z%]+))?"
    )
    for match in re.finditer(sci_pattern, text):
        value_str = match.group(1)
        unit = match.group(2) or ""
        orig_text = value_str + (f" {unit}" if unit else "")
        potential_matches.append(
            {
                "text": orig_text,
                "value": float(value_str),
                "span": match.span(),
                "type": "sci",
            }
        )

    # 2. Fractions (e.g., 1/2, 3 / 4) - with optional units
    # Ensure it's not part of a date like 01/02/2023. Unit part tries to avoid matching 'and'.
    fraction_pattern = r"(?<!\d/)(?<!\d)(?<!\.)(-?\d+)\s*/\s*(-?\d+)(?!\.\d)(?!\d*/)(?:\s+(?!and\b)([a-zA-Z%]+?)\b)?"
    for match in re.finditer(fraction_pattern, text):
        num_str, den_str = match.group(1), match.group(2)
        # Group 3 is the unit, group 4 (if it existed due to (?!and\b) ) would be the problem.
        # The unit is group 3. If it's not there, match.group(3) is None.
        unit = match.group(3) or ""
        try:
            value = float(num_str) / float(den_str)
            orig_text = f"{num_str}/{den_str}" + (f" {unit}" if unit else "")
            potential_matches.append(
                {
                    "text": orig_text,
                    "value": value,
                    "span": match.span(),
                    "type": "frac",
                }
            )
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Numbers with commas (e.g., 1,234,567.89, 384,400) - with optional units
    # Allows for optional decimal part.
    comma_num_pattern = (
        r"(?<![a-zA-Z0-9_])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*([a-zA-Z%]+))?"
    )
    for match in re.finditer(comma_num_pattern, text):
        value_str_commas = match.group(1)
        unit = match.group(2) or ""
        value_str_no_commas = value_str_commas.replace(",", "")
        try:
            value = float(value_str_no_commas)
            orig_text = value_str_commas + (f" {unit}" if unit else "")
            potential_matches.append(
                {
                    "text": orig_text,
                    "value": value,
                    "span": match.span(),
                    "type": "comma_num",
                }
            )
        except ValueError:
            pass

    # 4. Decimal numbers (e.g., 3.14, -0.5) - with optional units
    # Ensure it's not part of a version number like 1.2.3 or already matched by sci/comma
    # (?!\d*[eE]) negative lookahead for scientific notation
    # (?<!,\d{3}) negative lookbehind for comma-separated numbers
    # Add negative lookahead for "op num = num" to avoid extracting LHS of simple equations if result is extracted by eq_result
    decimal_pattern = r"(?<![a-zA-Z0-9_])(?<!,\d{3})(-?\d+\.\d+)(?!\d*[eE])(?!\s*[\+\-\*\/]\s*\d+\s*=\s*-?\d+(?:\.\d+)?)(?:\s*([a-zA-Z%]+))?"
    for match in re.finditer(decimal_pattern, text):
        value_str = match.group(1)
        unit = match.group(2) or ""
        orig_text = value_str + (f" {unit}" if unit else "")
        potential_matches.append(
            {
                "text": orig_text,
                "value": float(value_str),
                "span": match.span(),
                "type": "decimal",
            }
        )

    # 5. Integers (e.g., 42, -100) - with optional units
    # Ensure it's not part of a decimal, sci, fraction, or comma-separated number
    # (?!\.\d) not followed by decimal point and digit
    # (?![eE][-+]?\d+) not followed by scientific notation exponent
    # (?! */ *\d+) not part of a fraction (denominator)
    # (?<!\d */ *) not part of a fraction (numerator)
    # (?<!\d,) not preceded by digit and comma (part of comma num)
    # (?!,\d{3}) not followed by comma and 3 digits (part of comma num)
    # Add negative lookahead for "op num = num" to avoid extracting LHS of simple equations if result is extracted by eq_result
    integer_pattern = r"(?<![a-zA-Z0-9_])(?<!\d\.)(-?\d+)(?!\.\d)(?![eE][-+]?\d+)(?!,\d{3})(?!\s*/\s*\d+)(?!\s*[\+\-\*\/]\s*\d+\s*=\s*-?\d+(?:\.\d+)?)(?:\s*([a-zA-Z%]+))?"
    for match in re.finditer(integer_pattern, text):
        value_str = match.group(1)
        unit = match.group(2) or ""
        orig_text = value_str + (f" {unit}" if unit else "")
        potential_matches.append(
            {
                "text": orig_text,
                "value": float(value_str),
                "span": match.span(),
                "type": "int",
            }
        )

    # 6. Handle expressions like "2+2=4" by extracting the result (4)
    # This should be specific enough not to clash badly, but run it late.
    eq_result_pattern = r"=\s*(-?\d+(?:\.\d+)?)\b"
    for match in re.finditer(eq_result_pattern, text):
        value_str = match.group(1)
        # Check if this span is already covered by a more specific match
        is_covered = any(
            pm["span"][0] <= match.start(1) and pm["span"][1] >= match.end(1)
            for pm in potential_matches
            if pm["type"]
            != "eq_result"  # Avoid self-comparison if we were to add eq_results earlier
        )
        if not is_covered:
            potential_matches.append(
                {
                    "text": value_str,
                    "value": float(value_str),
                    "span": (match.start(1), match.end(1)),
                    "type": "eq_result",
                }
            )

    # Filter out overlapping matches.
    # A common strategy: sort by start position, then by length (longest first) or by a priority order.
    # If two matches overlap, take the "better" one (e.g., longer, or more specific type).

    # Sort by start position, then by length descending (longer matches first for tie-breaking)
    # and then by a predefined type priority to break ties for same-span matches.
    type_priority = {
        "latex_boxed_simple": 0,
        "latex_boxed_frac": 1,  # Highest priority for specific LaTeX structures
        "latex_frac": 2,
        "latex_sci": 3,  # Specific LaTeX constructs
        "eq_result": 4,  # Specific context like "=4"
        "sci": 5,  # Non-LaTeX scientific notation
        "comma_num": 6,  # Numbers with commas
        "frac": 7,  # Non-LaTeX fractions
        "decimal": 8,  # Decimal numbers
        "int": 9,  # Integers
        "latex_num": 10,  # General number inside LaTeX, lower priority
    }
    # Add 'type' to all matches if some LaTeX ones missed it
    for pm in potential_matches:
        if "type" not in pm:
            pm["type"] = "unknown"  # Should not happen with current code

    potential_matches.sort(
        key=lambda m: (
            m["span"][0],
            -(m["span"][1] - m["span"][0]),
            type_priority.get(m["type"], 99),
        )
    )

    final_results = []
    last_covered_end = -1

    for match_info in potential_matches:
        start, end = match_info["span"]
        # If this match starts after the last covered region, it's a new valid match.
        # Or, if it's a more specific type (e.g. latex_boxed) that covers the same span as a more general one.
        # The sorting should handle most cases of longer preferred over shorter.
        # The main issue is true overlaps (A starts before B, B ends after A, A and B overlap).

        # Simple non-overlapping filter:
        if start >= last_covered_end:
            final_results.append((match_info["text"], match_info["value"]))
            last_covered_end = end
        # Else: it overlaps or is contained; skip it due to prior sort.
        # This simple filter might discard a "better" (e.g. more specific type) match if a slightly
        # earlier, longer, but less specific match was chosen.
        # E.g. "1.23e4" (sci) vs "1.23" (decimal part of it). Sci should win.
        # The sort key (length descending for same start) should help here.

    # A more robust de-duplication for LaTeX:
    # If a number was found both inside a LaTeX block and by a general regex,
    # prefer the LaTeX version if spans are very similar.
    # The current filtering is primarily based on non-overlapping spans after sorting.

    # The test cases expect a list of (original_text_of_number, float_value)
    # The `orig_text` should be what was matched by the regex for that specific number.
    return final_results


def compare_numbers(
    expected: float,
    actual: float,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-8,
) -> Tuple[bool, float]:
    """
    Compare two numbers with configurable tolerance.

    Args:
        expected: Expected answer
        actual: Actual answer
        relative_tolerance: Maximum allowed relative difference
        absolute_tolerance: Maximum allowed absolute difference

    Returns:
        Tuple of (is_match, similarity_score)
    """
    # Check if values are close enough
    is_close = math.isclose(
        expected, actual, rel_tol=relative_tolerance, abs_tol=absolute_tolerance
    )

    if is_close:
        return True, 1.0

    # If not an exact match, calculate similarity based on relative error
    try:
        if expected == 0:
            # Avoid division by zero, use absolute error
            error = abs(actual)
            similarity = max(0.0, 1.0 - min(1.0, error / absolute_tolerance))
        else:
            rel_error = abs((expected - actual) / expected)
            similarity = max(
                0.0, 1.0 - min(1.0, rel_error / relative_tolerance)
            )
    except (ZeroDivisionError, OverflowError):
        similarity = 0.0

    return False, similarity


@reward_function
def math_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    original_messages: Union[List[Dict[str, Any]], List[Message]],
    tolerance: float = 0.001,
    absolute_tolerance: float = 1e-8,
    require_units: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate mathematical answers in messages.

    This function extracts numerical answers from both generated messages
    and original (ground truth) messages, then compares them to calculate
    a reward score.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation messages (containing ground truth)
        tolerance: Relative tolerance for numerical comparison
        absolute_tolerance: Absolute tolerance for numerical comparison
        require_units: Whether to require matching units
        **kwargs: Additional keyword arguments

    Returns:
        RewardOutput with score and metrics
    """

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

    # Extract the last assistant message from each list
    # Get last message (the model's response)
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

    if (
        not gen_content or not orig_content
    ):  # This check might be redundant now due to above checks
        return EvaluateResult(
            score=0.0,
            reason="Empty message content in generated or original message",
            metrics={
                "error": MetricResult(
                    score=0.0, success=False, reason="Empty message content"
                )
            },
        )

    # Extract numerical answers
    gen_answers = extract_numbers(gen_content)
    orig_answers = extract_numbers(orig_content)

    # Log extracted answers
    metrics: Dict[str, MetricResult] = {}
    metrics["extracted_original_answers"] = MetricResult(
        score=0.0,  # Not a real score, more like metadata
        success=True if orig_answers else False,
        reason=f"Extracted answers from original message: {
            ', '.join(
                [
                    a[0] for a in orig_answers]) if orig_answers else 'None'}",
    )

    metrics["extracted_generated_answers"] = MetricResult(
        score=0.0,  # Not a real score, more like metadata
        success=True if gen_answers else False,
        reason=f"Extracted answers from generated message: {
            ', '.join(
                [
                    a[0] for a in gen_answers]) if gen_answers else 'None'}",
    )

    # Handle the case where no answers were found
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

    # Unit handling is now done properly in the comparison loop

    # Compare answers
    best_match_score = 0.0
    best_match_reason = "No matching answer found"

    for orig_text, orig_value in orig_answers:
        for gen_text, gen_value in gen_answers:
            # Compare units if required
            if require_units:
                # Extract units more reliably by looking for non-numeric parts at the end
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

            # Compare numerical values
            is_match, similarity = compare_numbers(
                orig_value,
                gen_value,
                relative_tolerance=tolerance,
                absolute_tolerance=absolute_tolerance,
            )

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_reason = (
                    f"Best match: '{gen_text}' ({gen_value}) vs '{orig_text}' ({orig_value})\n"
                    f"Match: {'Yes' if is_match else 'No'}, Similarity: {similarity:.3f}"
                )

    metrics["answer_comparison"] = MetricResult(
        score=best_match_score,
        success=best_match_score > 0,
        reason=best_match_reason,
    )

    return EvaluateResult(
        score=best_match_score, reason=best_match_reason, metrics=metrics
    )
