# Issues and Tasks

## Crafting more custom reward functions
### Code execution and math rewards

I have the reward functions from `verl` under the folder `verl/verl/utils/reward_score`. I want to borrow their code to setup math and coding reward functions.


Basically original message should have the question and the assistant message that would contain the answer, so there is no separate concept of ground truth. And then for math, we should use regex to pull answers from both the original message's final assistant message and the messages's final assistant message which is generated, and then compare the answer

For coding I want to use E2B as the sandbox for execution as well as local execution, and there are some examples for how to run this in very. For E2B, I have the docs under `E2B/apps/web/src/app/(docs)/docs`.

For both of these we need to write very comprehensive tests, and then we need to go through a bunch of steps to make it happen, so please first dump the plan into ISSUES.md for me to review, and then we will try to do it one at a time.

## Implementation Plan for Math and Code Execution Reward Functions

### 1. Math Reward Function ✅

#### Structure: ✅
1. Create a new file `math.py` in the `reward_kit/rewards/` directory ✅
2. Implement a `math_reward` function that: ✅
   - Extracts numerical answers from both original and generated messages using regex ✅
   - Compares the answers and provides a score based on match ✅
   - Provides detailed metrics including extracted values and comparison details ✅

#### Key Steps: ✅
1. Implement regex patterns for common math answer formats (numbers, fractions, scientific notation) ✅
2. Create answer extraction function that can handle multiple formats ✅
3. Implement tolerance-based comparison for floating point numbers ✅
4. Add support for units when comparing answers ✅
5. Provide comprehensive metrics explaining the scoring process ✅

#### Tests: ✅
1. Create test cases with various math problem types: ✅
   - Simple arithmetic ✅
   - Fractions and decimals ✅
   - Scientific notation ✅
   - Multiple steps with intermediate results ✅
   - Problems with/without units ✅

#### LaTeX Support: ✅
1. Added support for LaTeX-formatted answers: ✅
   - Boxed expressions: `\boxed{42}` ✅
   - Fractions: `\frac{3}{4}` ✅
   - Scientific notation: `\times 10^8` ✅
   - Units in text: `\text{kg}` ✅

### 2. Code Execution Reward Function ✅

#### Structure: ✅
1. Create a new file `code_execution.py` in the `reward_kit/rewards/` directory ✅
2. Implement the main function: ✅
   - `local_code_execution_reward` for running code in a local environment ✅
   - (E2B integration to be implemented separately) 

#### Key Steps for Local Execution: ✅
1. Safely extract code blocks from messages ✅
2. Implement a secure code execution environment with proper isolation ✅
3. Set up output capture and timeout mechanisms ✅
4. Implement comparison between expected and actual outputs ✅
5. Handle various languages (Python, JavaScript) ✅

#### Tests: ✅
1. Create test cases for different languages and scenarios: ✅
   - Simple code execution tests ✅
   - Tests with inputs/outputs ✅
   - Tests with long-running code (timeout tests) ✅
   - Error handling tests ✅
   - Tests for different languages ✅

#### E2B Integration: ✅
1. Set up E2B client integration ✅
2. Create a standardized interface to E2B sandbox ✅
3. Implement secure code execution and timeout handling ✅
4. Set up proper error handling and reporting ✅

### 3. Common Implementation Tasks

#### Project Structure Updates: ✅
1. Update `reward_kit/rewards/__init__.py` to import the new modules ✅
2. Add proper documentation in the code ✅
3. Update docs directory and example files ✅
   - Created comprehensive documentation for all out-of-the-box reward functions ✅
   - Added overview document listing all available reward functions ✅
   - Updated main documentation index with links to all new documentation ✅

#### Dependencies: ✅
1. For local execution: Used built-in Python packages for secure code execution ✅
2. For E2B: Added `e2b_code_interpreter` as an optional dependency ✅

#### Security Considerations: ✅
1. Implement strict security measures for code execution ✅
   - Used temporary files with proper cleanup ✅
   - Set execution timeouts ✅
   - Captured and handled errors properly ✅
   - Added reliability guard to disable destructive functions ✅
   - Implemented process isolation with multiprocessing ✅
   - Set memory limits and resource constraints ✅
   - Secured JavaScript execution with Node.js sandbox ✅
2. Provide clear documentation about security implications ✅
3. Set up proper sandboxing and resource limitations ✅

### 4. Implementation Schedule

#### Phase 1: Math Reward Function ✅
1. Implement core regex-based answer extraction ✅
2. Implement comparison logic with tolerance ✅
3. Write tests for basic functionality ✅
4. Add advanced features (units, scientific notation, etc.) ✅
5. Complete comprehensive tests ✅

#### Phase 2: Local Code Execution Reward ✅
1. Implement secure code extraction ✅
2. Set up basic local execution environment ✅
3. Implement output comparison ✅
4. Add language-specific handlers ✅
5. Write tests for various scenarios ✅

#### Phase 3: E2B Code Execution Reward (Partially done but never ran end to end)
1. Set up E2B client integration ✅
2. Implement sandbox execution ✅
3. Add comprehensive error handling ✅
4. Write tests for E2B execution ✅
5. Create documentation for E2B integration ✅

#### Phase 2 Complete ✅
- All Node.js tests fully implemented and verified

### Phase 3: E2B Integration Complete ✅
- Successfully ran E2B integration end-to-end for both Python and JavaScript
- Updated code to work with latest E2B API (using e2b_code_interpreter package)
- Created comprehensive tests to verify functionality
- Updated documentation with clear instructions