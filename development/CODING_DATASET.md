# Plan for Normalizing Sandbox Fusion Python Datasets to OpenAI Format

This document outlines the plan to extract Python coding problems from the Sandbox Fusion dataset, normalize them into the OpenAI messages format with dataset-specific prompt engineering, and consolidate them into a single JSONL file.

## 1. Objective

The goal is to create a dataset suitable for fine-tuning OpenAI models, with a focus on instruction-following for code generation. This involves:
1.  Identifying Python-specific problem sets within the Sandbox Fusion data.
2.  For each dataset, analyzing its specific prompt construction logic as used in SandboxFusion.
3.  Extracting and formatting the problem description/instruction (user prompt) and the canonical solution (assistant response) according to this logic.
4.  Formatting this data into a JSONL file where each line is a JSON object compatible with OpenAI's fine-tuning requirements.

## 2. Data Source Identification

-   The primary data is located in `.jsonl` files within the `./SandboxFusion/sandbox/tests/datasets/samples/` directory (relative to the `reward-kit` project root).
-   The Python parser scripts for these datasets are located in `./SandboxFusion/sandbox/datasets/`.
-   The following files are targeted for processing:
    -   `code_eval_shadow_humaneval_python.jsonl`
    -   `code_eval_mbpp.jsonl`
    -   `code_eval_mhpp.jsonl`
    -   `code_eval_ncb_python_en.jsonl`
    -   `code_eval_ncb_python_zh.jsonl`
    -   `code_eval_repobench_c_python_sampled.jsonl`
    -   `code_eval_repobench_p_python_sampled.jsonl`
    -   `code_eval_cruxeval.jsonl`
    -   `code_eval_cruxeval_x.jsonl`
    -   `code_eval_aider_benchmark_v1.jsonl`
    -   `code_eval_bigcodebench.jsonl`
    -   `code_eval_EvoEval.jsonl`
    -   `code_eval_mbxp_v1_en.jsonl` (multilingual, filter for Python)
    -   `code_eval_humanevalds_v1_en.jsonl` (multilingual, filter for Python)
    -   `code_eval_humanevalds_v2_en.jsonl` (multilingual, filter for Python)
    -   `code_eval_mbxp_v2_en.jsonl` (multilingual, filter for Python)

## 3. Dataset-Specific Prompt Engineering Strategy

The core idea is to replicate or adapt the prompt construction logic found in each dataset's SandboxFusion parser.

### 3.1. `code_eval_aider_benchmark_v1.jsonl`
   - **Parser**: `SandboxFusion/sandbox/datasets/aider_benchmark.py`
   - **Key JSONL Fields**:
     - `content`: Main problem description/question.
     - `labels.reference`: Used as a placeholder/example in the original prompt template.
     - `canonical_solution`: The actual code solution.
   - **User Message Construction**:
     ```python
     f"{problem_json['content']}\n\nPlease generate the code in the following format:\n```python\n# Your code response here\n```"
     ```
     (The optional `autoeval_wrap_prompt` from the parser will be omitted for general dataset generation unless found to be critical.)
   - **Assistant Message Construction**: `problem_json['canonical_solution']` (or its alternatives like `solution`, `code`).

### 3.2. `code_eval_shadow_humaneval_python.jsonl`, `code_eval_EvoEval.jsonl`, `code_eval_humanevalds_*.jsonl`
   - **Parser**: `SandboxFusion/sandbox/datasets/humaneval.py`
   - **Key JSONL Fields**:
     - `prompt`: Typically contains the function signature and docstring.
     - `canonical_solution`: The body of the function.
   - **User Message Construction**:
     - Attempt to extract the docstring from `problem_json['prompt']` using `ast.get_docstring`. This docstring becomes the user message.
     - If `problem_json['prompt']` does not yield a docstring (e.g., it's a natural language instruction), use `problem_json['prompt']` directly as the user message.
   - **Assistant Message Construction**:
     - Combine the original `problem_json['prompt']` (which acts as the function stub/signature) with `problem_json['canonical_solution']` (the function body) to create the complete function code.

### 3.3. `code_eval_mbpp.jsonl`
   - **Parser**: `SandboxFusion/sandbox/datasets/mbpp.py`
   - **Key JSONL Fields**:
     - `content`: The textual problem description.
     - `test_list`: A list of assertion statements.
     - `labels.test_setup_code`: Setup code, often imports.
     - `canonical_solution`: The code solution.
   - **User Message Construction (Zero-shot approach)**:
     ```python
     tests_string = '\n'.join(problem_json.get('test_list', []))
     f"You are an expert Python programmer, and here is your task: {problem_json['content']} Your code should pass these tests:\n\n{tests_string}"
     ```
   - **Assistant Message Construction**:
     - Prepend `problem_json['labels'].get('test_setup_code', '')` to `problem_json['canonical_solution']` if the setup code is not already part of the solution. Ensure proper newline handling.

### 3.4. Other Datasets (To Be Investigated)
The following datasets and their corresponding parsers need to be analyzed to determine their specific prompt construction logic:
    - `code_eval_mhpp.jsonl` (Parser: `mhpp.py`)
    - `code_eval_ncb_python_en.jsonl`, `code_eval_ncb_python_zh.jsonl` (Parser: `natural_code_bench.py`)
    - `code_eval_repobench_c_python_sampled.jsonl` (Parser: `repobench_c.py`)
    - `code_eval_repobench_p_python_sampled.jsonl` (Parser: `repobench_p.py`)
    - `code_eval_cruxeval.jsonl`, `code_eval_cruxeval_x.jsonl` (Parser: `cruxeval.py`)
    - `code_eval_bigcodebench.jsonl` (Parser: TBD, may require direct inspection or use a generic approach if no specific parser is found)
    - `code_eval_mbxp_*.jsonl` (Parser: `mbxp.py`)

**General Fallback Strategy**: If a specific parser's logic is not easily adaptable or if a dataset has no clear parser, the refined docstring extraction method (user = docstring, assistant = full code stub + solution body) will be used as a default.

## 4. Normalization and Collection Process (Updated Script Logic)

A Python script (`development/normalize_sandbox_fusion.py`) will be updated to perform the normalization and collection.

### Script Logic Outline:

```python
import json
import os
import ast

SANDBOX_SAMPLES_DIR = "./SandboxFusion/sandbox/tests/datasets/samples/"
# ... (ALL_SOURCE_JSONL_FILES list remains the same) ...
OUTPUT_JSONL_FILE = "./development/CODING_DATASET.jsonl"

def extract_python_docstring(code_string: str) -> str | None:
    try:
        tree = ast.parse(code_string.strip())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                return docstring.strip() if docstring else None
        return None
    except SyntaxError:
        return None

def format_aider_prompt(problem_json: dict) -> str:
    question = problem_json.get('content', '')
    # The 'reference' in Aider's original prompt is a placeholder for output format,
    # not the solution itself. We instruct the LLM to provide code there.
    return f"{question}\n\nPlease generate the code in the following format:\n```python\n# Your code response here\n```"

def format_mbpp_prompt(problem_json: dict) -> str:
    question = problem_json.get('content', '')
    test_list = problem_json.get('test_list', [])
    tests_string = '\n'.join(test_list)
    return f"You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:\n\n{tests_string}"

def normalize_problem_to_openai_format(problem_json: dict, filename: str, is_multilingual_file: bool) -> dict | None:
    problem_id_str = str(problem_json.get('id', 'N/A'))
    try:
        user_content_keys = ["content", "prompt", "problem", "text"] # Prioritize 'prompt' for humaneval, 'content' for others
        assistant_content_keys = ["canonical_solution", "solution", "code", "completion"]

        raw_user_field_content = None # Content from the field like 'prompt' or 'content'
        # ... (logic to find raw_user_field_content using user_content_keys) ...
        
        raw_assistant_content = None
        # ... (logic to find raw_assistant_content using assistant_content_keys) ...

        if raw_user_field_content is None or raw_assistant_content is None:
            # ... (logging for missing primary content) ...
            return None

        final_user_content = ""
        final_assistant_content = raw_assistant_content # Default

        # Dataset-specific formatting
        if "aider_benchmark" in filename:
            final_user_content = format_aider_prompt(problem_json)
            # Assistant content is typically just canonical_solution for Aider
            final_assistant_content = raw_assistant_content
        elif "mbpp" in filename and "mbxp" not in filename: # Distinguish from mbxp
            final_user_content = format_mbpp_prompt(problem_json)
            test_setup_code = problem_json.get('labels', {}).get('test_setup_code', '')
            if test_setup_code and test_setup_code not in raw_assistant_content:
                final_assistant_content = test_setup_code.strip() + "\n\n" + raw_assistant_content
        elif "humaneval" in filename or "evoeval" in filename: # HumanEval, Shadow HumanEval, EvoEval, HumanEvalDS
            # raw_user_field_content is from problem_json['prompt'] for these
            docstring = extract_python_docstring(raw_user_field_content)
            if docstring:
                final_user_content = docstring
                # Assistant content: full function (stub + body)
                # Ensure raw_user_field_content (the stub) is combined correctly with raw_assistant_content (the body)
                if not ("def " in raw_assistant_content.strip() or "class " in raw_assistant_content.strip()):
                    if raw_user_field_content.rstrip().endswith(":"):
                        final_assistant_content = raw_user_field_content.rstrip() + "\n" + raw_assistant_content
                    elif raw_user_field_content.endswith("\n"):
                         final_assistant_content = raw_user_field_content + raw_assistant_content
                    else:
                        final_assistant_content = raw_user_field_content + "\n" + raw_assistant_content
                # else: raw_assistant_content is already full
            else: # If no docstring, use raw_user_field_content as prompt
                final_user_content = raw_user_field_content
                # Assistant content remains raw_assistant_content
        else: # Fallback for other datasets (or those not yet specifically handled)
            docstring = extract_python_docstring(raw_user_field_content)
            if docstring:
                final_user_content = docstring
                if not ("def " in raw_assistant_content.strip() or "class " in raw_assistant_content.strip()):
                    if raw_user_field_content.rstrip().endswith(":"):
                        final_assistant_content = raw_user_field_content.rstrip() + "\n" + raw_assistant_content
                    # ... (other newline handling) ...
            else:
                final_user_content = raw_user_field_content
        
        # ... (rest of the validation, label processing, multilingual filtering) ...

        if not final_user_content.strip() or not final_assistant_content.strip():
            print(f"Warning: Skipping problem ID {problem_id_str} due to empty processed content after formatting.")
            return None

        return {
            "messages": [
                {"role": "user", "content": final_user_content.strip()},
                {"role": "assistant", "content": final_assistant_content.strip()},
            ]
        }
    except Exception as e:
        print(f"Warning: Skipping problem ID {problem_id_str} due to an unexpected error ({type(e).__name__}: {e}).")
        return None

def main():
    # ... (main function largely the same, but passes `filename` to normalize_problem_to_openai_format) ...
    # ... (Ensure user_content_keys prioritizes 'prompt' for humaneval, 'content' for others, or handle within normalize_problem_to_openai_format)
    # ... (Add counters for processed and skipped items) ...
    pass # Placeholder for full main function

if __name__ == "__main__":
    main()
```

### Script Execution Steps:
1.  The script will iterate through each file in `ALL_SOURCE_JSONL_FILES`.
2.  For each problem, `normalize_problem_to_openai_format` will be called with the `filename`.
3.  Based on `filename`, specific formatting logic (as detailed in section 3) will be applied to construct `final_user_content` and `final_assistant_content`.
4.  Standard checks (multilingual, empty content, placeholder solutions) will be performed.
5.  Valid, formatted problems are written to `development/CODING_DATASET.jsonl`.

## 5. Output Format

-   The final output will be a single file: `development/CODING_DATASET.jsonl`.
-   Each line in this file will be a JSON object, structured as:
    ```json
    {"messages": [{"role": "user", "content": "Instructional prompt..."}, {"role": "assistant", "content": "Full code solution..."}]}
    ```

## 6. Next Steps (After this MD update)
1.  Complete the analysis of the remaining dataset parsers listed in section 3.4.
2.  Update the `normalize_problem_to_openai_format` function in `development/normalize_sandbox_fusion.py` with the specific logic for all identified dataset types.
3.  Thoroughly test the script with a few sample lines from each dataset type to ensure correct prompt and solution formatting.
4.  Run the script on the full dataset.
5.  Review the generated `CODING_DATASET.jsonl` for quality and correctness.

This updated plan aims to create a more targeted and effective dataset for fine-tuning.
