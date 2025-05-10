"""
Math reward function for evaluating mathematical answer correctness.

This module provides functions to evaluate the correctness of mathematical
answers by extracting numerical values from text using regex patterns and
comparing them with expected answers.
"""

from typing import Dict, List, Tuple, Any, Union, cast, Optional
import re
import math
from ..typed_interface import reward_function

# RewardOutput and MetricRewardOutput will be replaced by EvaluateResult and MetricResult
from ..models import Message, EvaluateResult, MetricResult


def extract_numbers(text: str) -> List[Tuple[str, Union[float, str]]]:
    """
    Extracts mathematical answers from text based on a hierarchical priority:
    1. Boxed LaTeX expressions (e.g., \\boxed{answer})
    2. GSM8K-style final answer markers (e.g., #### 123)
    3. Multiple Choice Question (MCQ) options (e.g., (A), B.)
    4. General numeric or LaTeX-formatted numbers as a fallback.

    Args:
        text: The text to extract answers from.

    Returns:
        A list of tuples, where each tuple contains the original matched
        string and its normalized value (float for numbers, str for MCQs
        or specific string expressions like "A or B").
        Returns an empty list if no answer is confidently extracted.
    """
    # ALGEBRAIC_VARS_SET is used by general number extraction helpers
    # to avoid misinterpreting coefficients (e.g., "4x") as standalone numbers.
    ALGEBRAIC_VARS_SET = {'x', 'y', 'z', 'a', 'b', 'c', 'n', 't', 'q', 'p', 'r', 'u', 'v', 'w'}

    # Helper to parse a string that might be a number or fraction
    def _parse_numeric_string(s: str) -> Optional[float]:
        s = s.strip()
        try:
            # Simple float/int
            if re.fullmatch(r"-?\d+(\.\d+)?", s):
                return float(s)
            # Fraction
            m_frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
            if m_frac:
                num = float(m_frac.group(1))
                den = float(m_frac.group(2))
                return num / den
        except (ValueError, ZeroDivisionError):
            return None
        return None

    # --- Priority 1: Boxed LaTeX expressions ---
    boxed_answers: List[Tuple[str, Union[float, str]]] = []
    found_any_boxed_expr = False
    for m_boxed in re.finditer(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text):
        found_any_boxed_expr = True
        original_boxed_expr = m_boxed.group(0)
        content = m_boxed.group(1).strip()

        if not content:
            continue

        if " or " in content.lower():
            boxed_answers.append((original_boxed_expr, content))
            continue
        if re.fullmatch(r"[A-Ea-e]", content):
            boxed_answers.append((original_boxed_expr, content.upper()))
            continue
        
        m_latex_frac = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", content)
        if m_latex_frac:
            try:
                num = float(m_latex_frac.group(1))
                den = float(m_latex_frac.group(2))
                boxed_answers.append((original_boxed_expr, num / den))
                continue
            except (ValueError, ZeroDivisionError):
                pass

        numeric_val = _parse_numeric_string(content)
        if numeric_val is not None:
            boxed_answers.append((original_boxed_expr, numeric_val))
            continue
        
        # NEW: Try to parse "number unit" from boxed content if other parsers failed
        # e.g. \boxed{10 km}
        m_num_unit = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z%]+)", content)
        if m_num_unit:
            try:
                num_val = float(m_num_unit.group(1))
                # unit = m_num_unit.group(2) # Unit is captured but not stored with the float value directly
                boxed_answers.append((original_boxed_expr, num_val))
                continue
            except ValueError:
                pass # Should not happen if regex is correct

    if found_any_boxed_expr: # If \boxed was found
        return boxed_answers # Return whatever was parsed from boxes (could be empty if all complex)

    # --- Priority 2: GSM8K-style final answer marker (#### ...) ---
    final_marker_answers: List[Tuple[str, Union[float, str]]] = []
    # More robust number pattern for GSM8K, ensuring group(1) is a clean number string
    # Handles integers, decimals, and numbers with commas. Does not grab trailing periods not part of the number.
    GSM8K_NUM_CONTENT_PATTERN = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
    for m_final in re.finditer(rf"####\s*({GSM8K_NUM_CONTENT_PATTERN})", text):
        original_marker_expr = m_final.group(0) # Full "#### <number>"
        num_str_from_regex = m_final.group(1)   # The number part itself
        
        cleaned_num_str = num_str_from_regex.replace(",", "")
        try:
            final_marker_answers.append((original_marker_expr, float(cleaned_num_str)))
        except ValueError:
            pass
    if final_marker_answers:
        return final_marker_answers

    # --- Priority 3: General number/LaTeX extraction (Fallback, simplified) ---
    # MCQ extraction is now handled by multiple_choice_math_reward.py
    # This is for "final text from latex" or plain numbers when no specific markers are found.
    # This part needs to be careful not to be too greedy.
    # We'll collect all potential matches and then filter for non-overlapping, prioritizing longer/more specific ones.
    
    potential_general_matches: List[Dict[str, Any]] = [] # Store dicts: {text, value, span, type_priority}

    # General LaTeX numbers (not inside \boxed, as those are handled)
    # This includes \frac, \times 10^, and simple numbers in $...$
    for latex_block_match in re.finditer(r"\$\$(.*?)\$\$|\$(.*?)\$", text, re.DOTALL):
        content = latex_block_match.group(1) if latex_block_match.group(1) is not None else latex_block_match.group(2)
        offset = latex_block_match.start(1) if latex_block_match.group(1) is not None else latex_block_match.start(2)

        if not content: continue

        # Avoid re-processing content that IS a \boxed{} expression handled earlier
        if content.strip().startswith("\\boxed{") and content.strip().endswith("}"):
            continue # This $...$ was just a wrapper for a \boxed item.

        # a. LaTeX fractions: \frac{num}{den}
        for m in re.finditer(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", content):
            try:
                num, den = float(m.group(1)), float(m.group(2))
                potential_general_matches.append({
                    "text": m.group(0), "value": num / den, 
                    "span": (m.start(0) + offset, m.end(0) + offset), "type_priority": 1
                })
            except (ValueError, ZeroDivisionError): pass
        
        # b. LaTeX scientific: base \times 10^{exp}
        for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{(.*?)\}", content):
            try:
                base, exp = float(m.group(1)), float(m.group(2))
                potential_general_matches.append({
                    "text": m.group(0), "value": base * (10**exp),
                    "span": (m.start(0) + offset, m.end(0) + offset), "type_priority": 2
                })
            except ValueError: pass

        # c. Simple numbers within LaTeX
        for m in re.finditer(r"(?<![a-zA-Z0-9_])(-?\d+(?:\.\d+)?)(?![a-zA-Z0-9_])", content):
            val_str = m.group(1)
            idx_after_val_str_in_content = m.end(1) # Use end of number group

            is_coeff = False
            # Direct variable: "4y"
            if idx_after_val_str_in_content < len(content) and \
               content[idx_after_val_str_in_content].lower() in ALGEBRAIC_VARS_SET and \
               (idx_after_val_str_in_content + 1 == len(content) or not content[idx_after_val_str_in_content+1].isalnum()):
                is_coeff = True
            # Space then variable: "4 y"
            elif idx_after_val_str_in_content + 1 < len(content) and \
                 content[idx_after_val_str_in_content] == ' ' and \
                 content[idx_after_val_str_in_content+1].lower() in ALGEBRAIC_VARS_SET and \
                 (idx_after_val_str_in_content + 2 == len(content) or not content[idx_after_val_str_in_content+2].isalnum()):
                is_coeff = True
            
            if is_coeff:
                continue

            try:
                potential_general_matches.append({
                    "text": val_str, "value": float(val_str),
                    "span": (m.start(1) + offset, m.end(1) + offset), "type_priority": 3
                })
            except ValueError: pass
    
    # General plain numbers (non-LaTeX)
    # Using simplified versions of the original regexes, ensuring algebraic var check.
    # Order: scientific, fractions, comma_nums, decimals, integers.
    # Each pattern should ensure it's not grabbing a coefficient.
    
    # Helper for algebraic variable check for plain numbers
    def _is_coefficient(match_obj, num_group_idx=1, unit_group_idx=2):
        # num_group_idx is the group index of the number string itself
        # unit_group_idx is the group index of any captured unit
        
        # Check unit if captured
        unit_candidate = match_obj.group(unit_group_idx) if len(match_obj.groups()) >= unit_group_idx and match_obj.group(unit_group_idx) else ""
        if unit_candidate and len(unit_candidate) == 1 and unit_candidate.lower() in ALGEBRAIC_VARS_SET:
            return True # e.g. "4x" where 'x' is unit_candidate

        # Check char immediately after number string if no unit was captured by regex
        # or if unit was not an algebraic var
        if not unit_candidate or not (len(unit_candidate) == 1 and unit_candidate.lower() in ALGEBRAIC_VARS_SET) :
            idx_after_num_str = match_obj.end(num_group_idx)
            # Direct variable: "4x"
            if idx_after_num_str < len(text) and text[idx_after_num_str].lower() in ALGEBRAIC_VARS_SET and \
               (idx_after_num_str + 1 == len(text) or not text[idx_after_num_str+1].isalnum()):
                return True
            # Space then variable: "4 x"
            if idx_after_num_str + 1 < len(text) and text[idx_after_num_str] == ' ' and \
               text[idx_after_num_str+1].lower() in ALGEBRAIC_VARS_SET and \
               (idx_after_num_str + 2 == len(text) or not text[idx_after_num_str+2].isalnum()):
                return True
        return False

    # Plain scientific: 1.2e-5
    sci_pattern = r"(?<![a-zA-Z0-9_])(-?\d+\.?\d*[eE][-+]?\d+)(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(sci_pattern, text):
        if _is_coefficient(m): continue
        try: potential_general_matches.append({"text": m.group(0), "value": float(m.group(1)), "span": m.span(), "type_priority": 4})
        except ValueError: pass

    # Plain fractions: 1/2
    # Unit is group 3. Ensure text field is constructed carefully.
    frac_pattern = r"(?<!\d/)(?<!\d)(?<!\.)(-?\d+)\s*/\s*(-?\d+)(?!\.\d)(?!\d*/)(?:\s+(?!(?:and|or)\b)([a-zA-Z%]+)\b)?"
    for m in re.finditer(frac_pattern, text):
        if _is_coefficient(m, unit_group_idx=3): continue
        try:
            num, den = float(m.group(1)), float(m.group(2))
            
            num_str_clean, den_str_clean = m.group(1), m.group(2)
            unit_str_clean = m.group(3) or ""
            
            display_text = f"{num_str_clean}/{den_str_clean}"
            if unit_str_clean:
                display_text += f" {unit_str_clean}"
            
            potential_general_matches.append({"text": display_text, "value": num / den, "span": m.span(), "type_priority": 5})
        except (ValueError, ZeroDivisionError): pass

    # Plain numbers with commas: 1,234.56
    comma_num_pattern = r"(?<![a-zA-Z0-9_])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(comma_num_pattern, text):
        if _is_coefficient(m): continue
        try: potential_general_matches.append({"text": m.group(0), "value": float(m.group(1).replace(",", "")), "span": m.span(), "type_priority": 6})
        except ValueError: pass
        
    # Plain decimals: 3.14 (ensure not part of sci or comma_num already)
    # Negative lookaheads for e/E and leading comma are important.
    decimal_pattern = r"(?<![a-zA-Z0-9_])(?<!,\d{3})(-?\d+\.\d+)(?!\d*[eE])(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(decimal_pattern, text):
        if _is_coefficient(m): continue
        try: potential_general_matches.append({"text": m.group(0), "value": float(m.group(1)), "span": m.span(), "type_priority": 7})
        except ValueError: pass

    # Plain integers: 42 (ensure not part of other patterns)
    # Negative lookaheads for decimal, e/E, comma, fraction are important.
    integer_pattern = r"(?<![a-zA-Z0-9_])(?<!\d\.)(-?\d+)(?!\.\d)(?![eE][-+]?\d+)(?!,\d{3})(?!\s*/\s*\d+)(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(integer_pattern, text):
        if _is_coefficient(m): continue
        try: potential_general_matches.append({"text": m.group(0), "value": float(m.group(1)), "span": m.span(), "type_priority": 8})
        except ValueError: pass

    # Filter overlapping general matches: sort by start, then length (desc), then priority (asc)
    potential_general_matches.sort(key=lambda x: (x["span"][0], -(x["span"][1] - x["span"][0]), x["type_priority"]))
    
    filtered_general_answers: List[Tuple[str, Union[float, str]]] = []
    last_covered_end = -1
    for item in potential_general_matches:
        start, end = item["span"]
        if start >= last_covered_end:
            # Ensure value is float for numbers, str for others (though general is numeric here)
            value_to_append = item["value"]
            if isinstance(value_to_append, (int, float)): # Should always be for general
                 filtered_general_answers.append((item["text"], float(value_to_append)))
            # else: string values are not expected from general numeric extractors
            last_covered_end = end
            
    if filtered_general_answers:
        return filtered_general_answers

    return [] # Fallback if nothing found


def compare_numbers( # This function remains for float comparisons
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
    tolerance: float = 0.001, # For float comparisons
    absolute_tolerance: float = 1e-8, # For float comparisons
    require_units: bool = False, # Currently for numbers with units
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate mathematical answers in messages.

    Extracts numerical or specific string (MCQ, "or" expressions) answers
    from generated and original messages, then compares them.
    Applies strictness rules based on ISSUES.md.

    Args:
        messages: Generated conversation messages.
        original_messages: Original conversation messages (ground truth).
        tolerance: Relative tolerance for numerical comparison.
        absolute_tolerance: Absolute tolerance for numerical comparison.
        require_units: Whether to require matching units for numerical answers.
                       (Note: unit handling in new extract_numbers is simplified,
                        this flag's effect might be limited to post-extraction checks if needed)
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
    """

    if not messages or not original_messages: # Ensure correct type for EvaluateResult
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
    gen_content = ""
    if messages and len(messages) > 0:
        gen_response_message = messages[-1]
        if isinstance(gen_response_message, Message) and gen_response_message.role == "assistant":
            gen_content = gen_response_message.content or ""
        elif isinstance(gen_response_message, dict) and gen_response_message.get("role") == "assistant":
            gen_content = gen_response_message.get("content", "")
    
    if not gen_content:
        return EvaluateResult(score=0.0, reason="Last generated message not from assistant or has no content",
                              metrics={"error": MetricResult(score=0.0, success=False, reason="Invalid generated message")})

    orig_content = ""
    if original_messages and len(original_messages) > 0:
        orig_response_message = original_messages[-1]
        if isinstance(orig_response_message, Message) and orig_response_message.role == "assistant":
            orig_content = orig_response_message.content or ""
        elif isinstance(orig_response_message, dict) and orig_response_message.get("role") == "assistant":
            orig_content = orig_response_message.get("content", "")

    if not orig_content:
        return EvaluateResult(score=0.0, reason="Last original message not from assistant or has no content",
                              metrics={"error": MetricResult(score=0.0, success=False, reason="Invalid original message")})

    # Extract numerical or string answers using the new extract_numbers
    gen_answers_extracted = extract_numbers(gen_content)
    orig_answers_extracted = extract_numbers(orig_content)

    metrics: Dict[str, MetricResult] = {}
    def format_extracted(items: List[Tuple[str, Union[float, str]]]) -> str:
        if not items: return "None"
        return ", ".join([f"'{i[0]}' ({i[1]})" for i in items])

    metrics["extracted_original_answers"] = MetricResult(
        score=0.0, success=bool(orig_answers_extracted),
        reason=f"Extracted from original: {format_extracted(orig_answers_extracted)}"
    )
    metrics["extracted_generated_answers"] = MetricResult(
        score=0.0, success=bool(gen_answers_extracted),
        reason=f"Extracted from generated: {format_extracted(gen_answers_extracted)}"
    )

    if not orig_answers_extracted: # If ground truth has no extractable answer, cannot evaluate.
        return EvaluateResult(score=0.0, reason="Could not extract answers from original message (ground truth).", metrics=metrics)
    
    # If gen has no answers but orig does, it's a clear mismatch.
    if not gen_answers_extracted:
        return EvaluateResult(score=0.0, reason="Could not extract answers from generated message, but original message has answers.", metrics=metrics)

    # --- Strictness Penalties (as per ISSUES.md) ---

    # Penalty 1 (Issue #1): Generated answer uses unboxed "or" to offer multiple numeric alternatives.
    # An "unboxed or" means " or " appears in gen_content, AND multiple numeric items were extracted,
    # AND no single extracted item from gen_answers_extracted is a string like "X or Y" (which would come from \boxed{X or Y}).
    
    # Check if any extracted gen_answer is a string containing " or " (this implies it was from \boxed{... or ...})
    is_gen_single_boxed_or_expr = any(
        isinstance(val, str) and " or " in val.lower()
        for _, val in gen_answers_extracted
    )
    gen_numeric_values_count = sum(1 for _, val in gen_answers_extracted if isinstance(val, (float, int)))

    if " or " in gen_content.lower() and gen_numeric_values_count > 1 and not is_gen_single_boxed_or_expr:
        specific_reason_detail = "Generated answer offers multiple numeric alternatives with an unboxed 'or'."
        full_reason = f"Strictness fail (Issue #1): {specific_reason_detail}"
        metrics["strictness_penalty_unboxed_or"] = MetricResult(score=0.0, success=False, reason=specific_reason_detail)
        return EvaluateResult(score=0.0, reason=full_reason, metrics=metrics)

    if len(orig_answers_extracted) == 1 and len(gen_answers_extracted) > 1:
        specific_reason_detail = "Ground truth is specific (one answer), but generated answer is ambiguous (multiple answers extracted)."
        full_reason = f"Strictness fail (Issue #2): {specific_reason_detail}"
        metrics["strictness_penalty_ambiguity"] = MetricResult(score=0.0, success=False, reason=specific_reason_detail)
        return EvaluateResult(score=0.0, reason=full_reason, metrics=metrics)
        
    # --- End of new strictness penalties ---

    best_match_score = 0.0
    best_match_reason = "No matching answer found"
    match_found_flag = False
    first_comparison_details_for_no_match = ""

    for orig_text, orig_value in orig_answers_extracted:
        for gen_text, gen_value in gen_answers_extracted:
            current_match = False
            current_similarity = 0.0
            comparison_details = ""

            # MCQ string comparison is now handled by multiple_choice_math_reward
            # This function will now focus on numeric or potentially other specific string (like "X or Y") comparisons.
            if isinstance(orig_value, (float, int)) and isinstance(gen_value, (float, int)):
                if require_units:
                    def has_unit_text(full_extracted_text: str, numeric_value: float) -> bool:
                        content_to_check = full_extracted_text
                        if content_to_check.startswith("\\boxed{") and content_to_check.endswith("}"):
                            content_to_check = content_to_check[7:-1].strip()
                        
                        num_str_float = str(numeric_value) # e.g. "10.0"
                        num_str_int = str(int(numeric_value)) if numeric_value == int(numeric_value) else None # e.g. "10"
                        
                        # Find the number string (float or int form) and check suffix
                        # Try float form first, then int form if applicable
                        search_terms = [num_str_float]
                        if num_str_int and num_str_int != num_str_float: # Add int form if different and valid
                            search_terms.append(num_str_int)

                        for term in search_terms:
                            found_at = content_to_check.find(term)
                            if found_at != -1:
                                suffix_start = found_at + len(term)
                                if suffix_start < len(content_to_check):
                                    suffix = content_to_check[suffix_start:].strip().split(" ")[0] # Get first word of suffix
                                    if suffix and not suffix.replace(".","",1).isdigit() and suffix.lower() != "or":
                                        return True
                        return False

                    orig_has_unit = has_unit_text(orig_text, float(orig_value))
                    gen_has_unit = has_unit_text(gen_text, float(gen_value))

                    if orig_has_unit != gen_has_unit:
                        comparison_details = f"Unit presence mismatch (require_units=True). Orig_text: '{orig_text}', Gen_text: '{gen_text}'"
                        current_match = False
                        current_similarity = 0.0
                    else: # Units presence matches (or both no units), proceed with number comparison
                        current_match, current_similarity = compare_numbers(
                            float(orig_value), float(gen_value), tolerance, absolute_tolerance
                        )
                        comparison_details = f"Numeric match: {'Yes' if current_match else 'No'}, Similarity: {current_similarity:.3f}"
                else: # require_units is False
                    current_match, current_similarity = compare_numbers(
                        float(orig_value), float(gen_value), tolerance, absolute_tolerance
                    )
                    comparison_details = f"Numeric match: {'Yes' if current_match else 'No'}, Similarity: {current_similarity:.3f}"
            
            elif isinstance(orig_value, str) and isinstance(gen_value, str):
                # This handles cases like "\\boxed{A or B}" vs "\\boxed{A or B}"
                # or other specific string answers that are not MCQs.
                if orig_value.lower() == gen_value.lower(): # Case-insensitive for general strings
                    current_match = True
                    current_similarity = 1.0
                comparison_details = f"String match: {'Yes' if current_match else 'No'} (value: '{gen_value}' vs '{orig_value}')"
            
            else: # Type mismatch
                comparison_details = f"Type mismatch: Gen({type(gen_value).__name__}) vs Orig({type(orig_value).__name__})"
            
            if not first_comparison_details_for_no_match: # Store details of the very first comparison
                first_comparison_details_for_no_match = (
                    f"Initial comparison: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )

            if current_similarity > best_match_score:
                best_match_score = current_similarity
                match_found_flag = current_match
                best_match_reason = (
                    f"Best match: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )
            elif best_match_score == 0 and not match_found_flag and current_similarity == 0:
                # If still no match and current is also no match, update reason to current (likely type mismatch)
                 best_match_reason = (
                    f"No score match: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )


    if best_match_score == 0 and not match_found_flag and first_comparison_details_for_no_match and best_match_reason == "No matching answer found":
        # If loops completed, no match was ever found (score remained 0),
        # and we have details from the first comparison, use that as the reason.
        best_match_reason = first_comparison_details_for_no_match
    
    # --- Penalty 3: Conflicting Answers (New) ---
    # Applied if a good match was found, but gen_answers also contains other significant, different numbers.
    if match_found_flag and best_match_score > 0.75: # Threshold for "good match"
        orig_numeric_values_set = {val for _, val in orig_answers_extracted if isinstance(val, (float, int))}
        # Consider only numeric values for this penalty for now
        
        conflicting_extra_numeric_values = []
        # Iterate through all extracted generated answers to find those not matching original solution
        for gen_text, gen_val in gen_answers_extracted:
            if not isinstance(gen_val, (float, int)):
                continue # Only consider numeric generated values for conflicting check

            is_part_of_orig_solution_or_close = False
            for orig_text_cmp, orig_val_cmp in orig_answers_extracted:
                if isinstance(orig_val_cmp, (float, int)): # Compare numeric gen_val with numeric orig_val
                    is_close, _ = compare_numbers(orig_val_cmp, gen_val, tolerance, absolute_tolerance)
                    if is_close:
                        is_part_of_orig_solution_or_close = True
                        break
                # Not comparing numeric gen_val with string orig_val_cmp here
            
            if not is_part_of_orig_solution_or_close:
                conflicting_extra_numeric_values.append(gen_val)
        
        # If there are numeric values in generated output that are not part of (or close to) the original solution
        if conflicting_extra_numeric_values:
            # Check if these conflicting values are "significant"
            # For now, any numeric value not accounted for by the original solution is considered part of a conflict.
            # This could be refined, e.g. by checking if they are far from original values, or form a sizable set.
            # Example: orig={10}, gen={10, 100}. 100 is conflicting.
            # Example: orig={10, 20}, gen={10, 20, 100, 200}. 100, 200 are conflicting.
            
            # Filter out very small numbers that might be noise if GT is not zero
            # This is a heuristic and might need adjustment.
            # If orig_numeric_values_set contains 0, then small numbers are fine.
            # Otherwise, filter small conflicting values if they are much smaller than typical orig values.
            # This is complex, for now, let's assume any distinct extra number is a conflict.

            # A simple check: are there any such conflicting numbers?
            if conflicting_extra_numeric_values: 
                formatted_conflicting = ", ".join(map(str, sorted(list(set(conflicting_extra_numeric_values)))))
                specific_reason_detail = (
                    f"Generated answer, while containing a match for the original, "
                    f"also includes other distinct numerical values: [{formatted_conflicting}]"
                )
                current_best_reason = best_match_reason # Save pre-penalty reason
                
                best_match_score = 0.0
                match_found_flag = False # No longer a success
                best_match_reason = f"Strictness fail (Conflicting Answers): {specific_reason_detail}. Initial match was: {current_best_reason}"
                metrics["strictness_penalty_conflicting_answers"] = MetricResult(
                    score=0.0, success=False, reason=specific_reason_detail
                )

    metrics["answer_comparison"] = MetricResult(
        score=best_match_score,
        success=match_found_flag and best_match_score > 0, # match_found_flag might be False now
        reason=best_match_reason,
    )
    return EvaluateResult(score=best_match_score, reason=best_match_reason, metrics=metrics)
