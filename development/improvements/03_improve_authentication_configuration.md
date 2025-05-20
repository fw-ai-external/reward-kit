# Improvement Plan: Enhance Authentication Configuration and Clarity

**Associated `IMPROVEMENT_PLAN.md` Item:** III.12. Clarify `FIREWORKS_ACCOUNT_ID` & Centralize Auth

## 1. Objective

*   To clarify the purpose, necessity, and usage of `FIREWORKS_ACCOUNT_ID` and other authentication-related variables (e.g., `FIREWORKS_API_KEY`).
*   To centralize all Fireworks AI authentication logic into a dedicated module, making it more robust, maintainable, and easier for users to configure.
*   To provide flexible configuration options for API keys and account IDs, supporting both environment variables and a dedicated INI configuration file.

## 2. Background / Current State

*   The `FIREWORKS_ACCOUNT_ID` environment variable is mentioned in `development/CONTRIBUTING.md` with a vague comment: `# For specific operations`. Its precise role is not clearly documented.
*   Authentication logic for `FIREWORKS_API_KEY` might exist in `reward_kit/auth.py`, but its full scope, support for `FIREWORKS_ACCOUNT_ID`, and alternative configuration methods like `auth.ini` are not yet established or are incomplete.
*   Logic for `FIREWORKS_ACCOUNT_ID` and potentially some API key handling might still be in `reward_kit/evaluation.py`.
*   Configuration of these credentials is predominantly reliant on environment variables.

## 3. Proposed Solution

### A. Enhance Authentication Module: `reward_kit/auth.py`

The existing module, `reward_kit/auth.py`, will be enhanced to serve as the central point for all Fireworks AI authentication.

*   **Ensure/Implement Core Functions:**
    *   `get_fireworks_api_key() -> Optional[str]`: Retrieves the Fireworks API key.
    *   `get_fireworks_account_id() -> Optional[str]`: Retrieves the Fireworks Account ID.
*   **Implement Credential Sourcing Priority:**
    1.  **Environment Variables:** (Highest priority) Checks for `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`.
    2.  **`auth.ini` Configuration File:** (Lower priority) If environment variables are not set, attempts to read from an INI file.
*   The functions will return `None` if a credential cannot be found through any source, allowing calling code to handle missing credentials appropriately.

### B. `auth.ini` Configuration File

*   **Purpose:** To provide an alternative to environment variables for storing credentials.
*   **Locations (checked in this order):**
    *  Global: `~/.fireworks/auth.ini`
*   **Format (using `configparser` module):**
    ```ini
    [fireworks]
    api_key = YOUR_FIREWORKS_API_KEY
    account_id = YOUR_FIREWORKS_ACCOUNT_ID
    ```
*   The library will gracefully handle cases where the file or specific keys are missing.

### C. Refactor Existing Code

*   `reward_kit/evaluation.py`: Any remaining direct access to `os.environ` for `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` will be removed. It will consistently import and use the getter functions from the enhanced `reward_kit/auth.py`.
*   `reward_kit/auth.py`: Ensure it does not contain logic that should reside in `evaluation.py` or other modules (i.e., it should focus solely on credential retrieval and management).
*   Other modules (if any) using these credentials will be similarly refactored to use `reward_kit/auth.py`.

### D. Investigate and Document `FIREWORKS_ACCOUNT_ID` Usage

*   A thorough review of the codebase and any interactions with Fireworks AI services will be conducted to determine:
    *   Which specific API calls or operations require `FIREWORKS_ACCOUNT_ID`.
    *   The consequences of it being absent when required.
*   This information will be explicitly documented in `README.md` and `development/CONTRIBUTING.md`.

### E. Documentation Updates

*   **`README.md`:**
    *   The "Authentication Setup" (or similar) section will be updated to explain the new methods (env vars, `auth.ini`).
    *   Briefly explain the purpose of `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`, and when the latter is necessary.
*   **`development/CONTRIBUTING.md`:**
    *   The "Required Environment Variables" section will be revised or expanded into a more general "Authentication Setup" guide.
    *   Detailed instructions on configuring credentials via both environment variables and the `auth.ini` file (including locations, format, and an example).
    *   Clear explanation of the `FIREWORKS_ACCOUNT_ID`'s purpose, based on the investigation in step 3.D.
    *   Explicitly state the credential sourcing priority.

## 4. Detailed Tasks

1.  **Task 1: Enhance `reward_kit/auth.py`**
    *   Review the existing `reward_kit/auth.py`.
    *   Ensure `get_fireworks_api_key()` and implement/enhance `get_fireworks_account_id()` to meet the requirements.
    *   Add logic to read from environment variables first for both credentials.
    *   Add logic to parse `auth.ini` (as described in 3.B) from specified locations using the `configparser` module if environment variables are not found. This includes handling file existence and key presence gracefully.
    *   Ensure functions correctly implement the sourcing priority and return `Optional[str]`.
    *   Refactor any existing logic within `auth.py` to align with this new structure if necessary.
2.  **Task 2: Codebase Review for `FIREWORKS_ACCOUNT_ID`**
    *   Scan the codebase for any current or potential usage of `FIREWORKS_ACCOUNT_ID`.
    *   Consult Fireworks AI documentation if necessary to understand its role in different API calls.
    *   Document findings.
3.  **Task 3: Refactor `reward_kit/evaluation.py` (and other relevant modules)**
    *   Replace any direct environment variable access or disparate auth logic for Fireworks credentials with calls to the enhanced `reward_kit.auth.get_fireworks_api_key()` and `reward_kit.auth.get_fireworks_account_id()`.
    *   Ensure the application handles `None` returns from these functions gracefully (e.g., by raising an informative error if a credential is required but not found).
4.  **Task 4: Update `README.md` Documentation**
    *   Revise the authentication section as per section 3.E.
5.  **Task 5: Update `development/CONTRIBUTING.md` Documentation**
    *   Revise and expand the authentication setup guide as per section 3.E.
6.  **Task 6: Add/Update Unit Tests for `reward_kit/auth.py`**
    *   Review and update existing tests for `get_fireworks_api_key()`.
    *   Add comprehensive tests for `get_fireworks_account_id()`.
    *   Test retrieval from environment variables for both.
    *   Test retrieval from `auth.ini` (mocking file system and `configparser`) for both.
        *   Test both project-local and user-config file locations.
        *   Test correct parsing of values.
    *   Test the priority logic (env vars overriding `auth.ini`).
    *   Test behavior when credentials are not found in any source (should return `None`).
    *   Test behavior with malformed `auth.ini` files (if applicable, though `configparser` handles much of this).
7.  **Task 7: Update `.gitignore`**
    *   Add `/.reward_kit_auth.ini` (or the chosen project-local filename) to `.gitignore` to prevent accidental credential commits.

## 5. Rationale for Changes

*   **Improved Clarity:** Users and contributors will have a clear understanding of how to configure authentication and the specific purpose of each credential, especially `FIREWORKS_ACCOUNT_ID`.
*   **Enhanced Flexibility:** Supporting both environment variables and a configuration file caters to diverse user preferences, development workflows, and deployment scenarios.
*   **Better Maintainability:** Centralizing authentication logic into a single, well-defined module simplifies the codebase, making it easier to understand, modify, and test.
*   **Consistency:** Provides a standardized way of handling authentication across the library.

## 6. Open Questions / Considerations

*   **Error Handling for Missing Credentials:** Confirm whether the getter functions in `auth.py` should solely return `None` (leaving error handling to the caller) or if they should have an option to raise an error if a credential is not found but deemed essential by the caller. (Current proposal: return `None` for flexibility).
*   **Other Credentials:** Are there any other authentication-related parameters for Fireworks AI or other services that should be incorporated into this new auth module now or in the near future?
*   **Security of `auth.ini`:** While `auth.ini` offers convenience, remind users in documentation about file permissions if storing sensitive keys, especially for the user-level config file. The project-level file being in `.gitignore` is a key part of this.
