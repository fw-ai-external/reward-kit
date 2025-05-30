import json
import uuid  # For generating tool call IDs
from typing import Any, Dict, List, Optional, Union


def _stringify_dict_values(item: Any) -> Any:
    """Recursively converts primitive values in dicts/lists to strings."""
    if isinstance(item, dict):
        return {k: _stringify_dict_values(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_stringify_dict_values(i) for i in item]
    elif isinstance(item, (int, float, bool)):  # Primitives that are not strings
        return str(item)
    return item


def parse_json_list_to_list_of_stringified_dicts(json_string: Optional[str]) -> list:
    """
    Safely deserializes a JSON string expected to be a list of objects.
    Ensures all primitive values within the dictionaries are strings.
    Returns an empty list if input is invalid or not a list of dicts.
    """
    if (
        json_string is None
        or not isinstance(json_string, str)
        or not json_string.strip()
    ):
        return []
    try:
        loaded_json = json.loads(json_string)
        if isinstance(loaded_json, list):
            processed_list = []
            for item in loaded_json:
                if isinstance(item, dict):
                    # Ensure 'name' and 'arguments' keys exist, even if arguments is empty
                    if "name" not in item or "arguments" not in item:
                        # Malformed tool call structure from source
                        # print(f"DEBUG: Malformed tool call in source, skipping: {item!r}")
                        continue  # Or handle error more strictly
                    processed_list.append(
                        {
                            "name": item["name"],
                            "arguments": _stringify_dict_values(item["arguments"]),
                        }
                    )
                else:
                    return []
            return processed_list
        else:
            return []
    except json.JSONDecodeError:
        return []


def reformat_answers_str_to_ground_truth_dict_string(
    example: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Takes an example dict, reads 'answers' (original column name), parses its JSON list content.
    Each item in the list is expected to be {"name": "func_name", "arguments": {...}}.
    It transforms each item into the OpenAI tool call format:
    {"id": "call_...", "type": "function", "function": {"name": "func_name", "arguments": "{...json_string_args...}"}}
    This list of transformed tool calls is then wrapped into the standard assistant message:
    {"role": "assistant", "tool_calls": [...]},
    and serialized to a JSON string in a new 'ground_truth' field.
    """
    answers_json_string = example.get("answers")

    # This gives a list of dicts like: [{"name": "func_name", "arguments": {"arg1": "val1_str", ...}}, ...]
    parsed_raw_tool_calls = parse_json_list_to_list_of_stringified_dicts(
        answers_json_string
    )

    formatted_tool_calls_for_openai = []
    if parsed_raw_tool_calls:
        for i, raw_call in enumerate(parsed_raw_tool_calls):
            # raw_call is {"name": "...", "arguments": {..._already_stringified_values_...}}
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}_{i}"  # Generate a unique ID

            # Arguments need to be a JSON string for the OpenAI format
            arguments_json_string = json.dumps(raw_call["arguments"])

            formatted_tool_calls_for_openai.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": raw_call["name"],
                        "arguments": arguments_json_string,
                    },
                }
            )

    ground_truth_dict_structure: Dict[str, Any]
    if not formatted_tool_calls_for_openai:
        ground_truth_dict_structure = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }
    else:
        ground_truth_dict_structure = {
            "role": "assistant",
            "content": None,
            "tool_calls": formatted_tool_calls_for_openai,
        }

    example["ground_truth"] = json.dumps(ground_truth_dict_structure)

    return example


def parse_json_list_to_list_of_json_strings(json_string: Optional[str]) -> List[str]:
    if (
        json_string is None
        or not isinstance(json_string, str)
        or not json_string.strip()
    ):
        return []
    try:
        loaded_json_list = json.loads(json_string)
        if isinstance(loaded_json_list, list):
            stringified_list = []
            for item in loaded_json_list:
                if isinstance(item, dict):
                    stringified_list.append(json.dumps(item))
                else:
                    return []
            return stringified_list
        else:
            return []
    except json.JSONDecodeError:
        return []


def parse_list_of_json_strings_to_final_dicts(
    list_of_json_strings: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not isinstance(list_of_json_strings, list):
        return []
    final_dicts = []
    for json_string_item in list_of_json_strings:
        if not isinstance(json_string_item, str):
            continue
        try:
            parsed_dict = json.loads(json_string_item)
            if isinstance(parsed_dict, dict):
                final_dicts.append(_stringify_dict_values(parsed_dict))
        except json.JSONDecodeError:
            continue
    return final_dicts


def format_messages_for_eval(query: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": query}]


def format_ground_truth_for_eval(parsed_answers: Optional[List[Dict[str, Any]]]) -> str:
    # This function is likely not directly used by load_derived_dataset if preprocessing handles ground_truth creation.
    ground_truth_dict: Dict[str, Any]
    if (
        parsed_answers is None
        or not isinstance(parsed_answers, list)
        or not parsed_answers
    ):
        ground_truth_dict = {"role": "assistant", "content": None, "tool_calls": None}
    else:
        ground_truth_dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": parsed_answers,
        }
    return json.dumps(ground_truth_dict)
