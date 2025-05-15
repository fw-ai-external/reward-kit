# Top-Level Directory Cleanup Plan

This document outlines the proposed plan for cleaning up the top-level directory of the `reward-kit` repository to prepare for open-sourcing.

## Completed Actions

- The following test files, previously at the top level, have been moved into the `tests/` directory:
    - `test_agent_direct.py`
    - `test_e2b_integration.py`
    - `test_e2b_js_integration.py`
    - `test_fireworks_api.py`

## Proposed Actions for Remaining Items

Based on feedback to keep these items public but avoid cluttering the `docs/` directory (which is primarily for user-facing documentation), the following actions are proposed. This may involve creating new top-level directories like `development/` and `api_specifications/`.

1.  **`AGENT_ISSUES.md`**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move to a new directory `development/notes/AGENT_ISSUES.md`.
    *   **Rationale**: If these are public notes or issues relevant to contributors/developers rather than end-users, this location separates them from user docs while keeping them accessible.

2.  **`CLAUDE.md`**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move to `development/notes/CLAUDE.md`.
    *   **Rationale**: Assuming this contains development-related notes (e.g., experiments, specific LLM details for contributors), placing it in `development/notes/` is appropriate.

3.  **`REPRO_CODING.md`**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move to `development/guides/REPRO_CODING.md`.
    *   **Rationale**: This file, related to reproducing coding results, serves as a guide for developers or contributors.

4.  **`README_dev.md`**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move to `development/CONTRIBUTING.md` (or `development/README_dev.md` if preferred, but `CONTRIBUTING.md` is a common standard for such content).
    *   **Rationale**: Developer setup and contribution guidelines fit well within a `development/` scope, often named `CONTRIBUTING.md`.

5.  **`fireworks.swagger.yaml`**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move to a new top-level directory `api_specifications/fireworks.swagger.yaml`.
    *   **Rationale**: API specifications are crucial public information but distinct from narrative documentation. A dedicated `api_specifications/` folder makes them easy to find.

6.  **`system_architecture/` (directory)**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Keep at the top-level OR move to `development/system_architecture/`.
    *   **Rationale**: System architecture can be fundamental public information. If it's highly technical and more for contributors, `development/system_architecture/` is an option. For this plan, let's initially propose keeping it top-level if it's broadly relevant, or moving to `development/` if it's more developer-centric. **Decision needed: For now, proposing to move to `development/system_architecture/` to group developer-focused content.**

7.  **`temp_metrics/` (directory)**:
    *   **Current Location**: Top-level
    *   **Proposed Action**: Move contents to a new directory `examples/metrics/custom_temp_metrics/`.
    *   **Rationale**: This proposal remains unchanged as it doesn't involve the `docs/` directory. It places example metrics appropriately within `examples/`.

## Next Steps

Please review this plan. Upon approval, these actions can be executed. If any proposals are not suitable, please provide alternative instructions.
