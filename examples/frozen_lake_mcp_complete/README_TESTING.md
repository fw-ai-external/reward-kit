# FrozenLake 8x8 Stack Hardening Tests

This directory contains comprehensive testing capabilities for validating that your MCP simulation server can handle complex 8x8 FrozenLake scenarios with **end-to-end integration testing**.

## Purpose

The 8x8 grid tests serve as **stack hardening** - they stress-test your system with complex, multi-step scenarios that require:
- 10-20+ moves to complete (vs 3-6 for 4x4)
- Navigation through 64 states (vs 16 for 4x4)
- Dynamic grid size handling
- Robust session management
- Complex decision making

## Files

### Core Test Files
- **`test_8x8_integration.py`** - Stack hardening test suite (adapter-level testing)
- **`test_e2e_integration.py`** - End-to-end integration test with trajectory recording
- **`test_trajectory_replay.py`** - Trajectory replay test for reliable validation
- **`shared_data/rollouts.jsonl`** - 4x4 baseline test configurations
- **`shared_data/rollouts_8x8.jsonl`** - 8x8 complex test configurations
- **`shared_data/recorded_e2e_trajectories.jsonl`** - Recorded golden trajectories for replay

### Server Files
- **`mcp_server/simulation_server.py`** - Fixed simulation server with dynamic grid sizing
- **`mcp_server/frozen_lake_adapter.py`** - Environment adapter supporting multiple grid sizes

## Running Tests

### Option 1: Using pytest (Recommended)

Run all tests with pytest for comprehensive CI/CD integration:

```bash
# Run all test functions
pytest test_*.py -v

# Run specific test categories
pytest test_8x8_integration.py -v     # Stack hardening tests
pytest test_e2e_integration.py -v     # End-to-end integration tests
pytest test_trajectory_replay.py -v   # Trajectory replay tests

# Run with coverage
pytest test_*.py --cov=mcp_server --cov-report=html

# Run in parallel (if pytest-xdist installed)
pytest test_*.py -n auto
```

**Test Results:**
```
11 tests collected: ✅ 11 passed, 0 failed
- Stack hardening: 4 tests (4x4/8x8 scenarios + grid accuracy + imports)
- End-to-end integration: 3 tests (4x4/8x8 scenarios + trajectory recording)
- Trajectory replay: 4 tests (file exists + 4x4/8x8 replay + deterministic)
```

### Option 2: Direct Python Execution

Run tests individually for development and debugging:

#### 1. Stack Hardening (Adapter Level)

Test the core adapter functionality with complex scenarios:

```bash
python test_8x8_integration.py
```

This validates:
- ✅ 4x4 and 8x8 grid creation
- ✅ Dynamic grid size handling
- ✅ Multi-step scenario execution
- ✅ Complexity metrics and hardening scores

#### 2. End-to-End Integration (Full Stack)

Run the complete MCP server stack and record golden trajectories:

```bash
python test_e2e_integration.py
```

This performs:
- 🎯 **Simulates real MCP client sessions**
- 📊 **Tests complete environment lifecycle**
- 💾 **Records golden trajectories for all scenarios**
- ✅ **Validates grid size configuration end-to-end**

#### 3. Trajectory Replay (Reliability Validation)

Use recorded trajectories for deterministic, reliable testing:

```bash
python test_trajectory_replay.py
```

This ensures:
- 🔄 **100% reproducible results**
- 🎯 **Step-by-step validation**
- 📊 **Deterministic behavior verification**
- ✅ **Integration test reliability**

#### 4. Live Server Testing (Optional)

Start the MCP server for manual testing:

```bash
python test_e2e_integration.py --server-only --port 8001
```

Connect manually or with custom clients to test live interactions.

## Test Scenarios

### 4x4 Baseline (3 tests)
- **Grid**: 4×4 (16 states)
- **Min moves to goal**: 6
- **Seeds**: 42, 123, 456
- **Purpose**: Validate basic functionality

### 8x8 Complex (3 tests)
- **Grid**: 8×8 (64 states)
- **Min moves to goal**: 14
- **Seeds**: 42, 123, 999
- **Purpose**: Stress test complex scenarios

## Success Criteria

### Stack Hardening Metrics
- **Stress Test Pass Rate**: ≥80% of tests must handle minimum required moves
- **Complexity Handling**: Must handle 30+ move scenarios
- **Grid Size Accuracy**: Dynamic grid sizing must work 100%
- **Session Robustness**: No crashes or errors during complex sessions

### End-to-End Integration
- **Grid Size Validation**: 100% accurate dynamic sizing
- **Trajectory Recording**: Complete game session capture
- **Environment Lifecycle**: Proper session creation and cleanup
- **Configuration Passing**: Client context properly transmitted

### Replay Reliability
- **Deterministic Behavior**: 100% trajectory replay accuracy
- **Step Validation**: Every action produces expected result
- **Outcome Consistency**: Final results match recorded trajectories
- **Grid Size Consistency**: Dynamic sizing produces same results

### Production Readiness
✅ **HARDENED** - Ready for production deployment
✅ **VALIDATED** - End-to-end integration confirmed
✅ **RELIABLE** - Deterministic, reproducible behavior
⚠️ **NEEDS HARDENING** - Requires optimization for complex scenarios

## Example Output

### Pytest Execution
```
pytest test_*.py -v
======================================================================
test_8x8_integration.py::test_4x4_baseline_scenarios PASSED     [  9%]
test_8x8_integration.py::test_8x8_complex_scenarios PASSED      [ 18%]
test_8x8_integration.py::test_grid_size_accuracy PASSED         [ 27%]
test_8x8_integration.py::test_adapter_imports PASSED            [ 36%]
test_e2e_integration.py::test_e2e_4x4_scenarios PASSED          [ 45%]
test_e2e_integration.py::test_e2e_8x8_scenarios PASSED          [ 54%]
test_e2e_integration.py::test_trajectory_recording PASSED       [ 63%]
test_trajectory_replay.py::test_trajectory_file_exists PASSED   [ 72%]
test_trajectory_replay.py::test_trajectory_replay_4x4 PASSED     [ 81%]
test_trajectory_replay.py::test_trajectory_replay_8x8 PASSED     [ 90%]
test_trajectory_replay.py::test_trajectory_replay_deterministic PASSED [100%]

======================================================================
✅ 11 passed, 0 failed
```

### Stack Hardening Test
```
🔧 STACK HARDENING TEST SUITE
======================================================================
📋 4x4 Baseline Tests: 3/3 passed
📋 8x8 Complex Tests: 3/3 passed

🎯 STACK HARDENING VERDICT:
✅ STACK HARDENED - System can handle complex scenarios!
✅ Hardening Score: 83.3%
```

### End-to-End Integration
```
🔗 END-TO-END INTEGRATION TEST
======================================================================
📋 4x4 baseline scenarios: 3/3 successful
📋 8x8 complex scenarios: 3/3 successful
💾 Saving 6 trajectories to shared_data/recorded_e2e_trajectories.jsonl

🎯 END-TO-END VERDICT:
✅ INTEGRATION SUCCESS - All scenarios passed!
```

### Trajectory Replay
```
🔄 TRAJECTORY REPLAY INTEGRATION TEST
======================================================================
📋 4x4 Trajectory Replays: 3/3 valid
📋 8x8 Trajectory Replays: 3/3 valid

🎯 REPLAY INTEGRATION VERDICT:
✅ TRAJECTORY REPLAY SUCCESS - All replays validated!
✅ Reliability Score: 100.0%
```

## Integration with CI/CD

### Pytest Integration (Recommended)

For reliable automated testing in CI/CD pipelines:

```bash
# Standard CI/CD workflow
pytest test_*.py --tb=short --maxfail=3

# With coverage reporting
pytest test_*.py --cov=mcp_server --cov-report=xml --cov-fail-under=80

# JUnit XML output for CI systems
pytest test_*.py --junitxml=test_results.xml
```

### Sequential Execution Workflow

For environments requiring sequential execution:

```bash
# 1. Generate golden trajectories
python test_e2e_integration.py

# 2. Validate trajectory replay reliability
python test_trajectory_replay.py

# 3. Run stack hardening validation
python test_8x8_integration.py
```

### Exit Codes
- **0**: All tests passed, system ready for production
- **1**: Tests failed, system needs fixes

## Troubleshooting

### Common Issues and Solutions

1. **File not found errors**:
   - ✅ **FIXED**: All test files now use absolute paths relative to test file location
   - **Solution**: Tests work from any directory (project, parent, or root)

2. **Import errors from old test files**:
   - ✅ **FIXED**: Old incompatible test files moved to `local_testing/archive/`
   - **Solution**: Pytest no longer collects outdated test files with broken imports

3. **Click command line errors in simulation_server.py**:
   - ✅ **FIXED**: Added proper exception handling for Click's SystemExit calls
   - **Solution**: Server can be imported and tested without Click argument issues

4. **Pytest collection errors**:
   - ✅ **FIXED**: Created `pytest.ini` configuration to exclude problematic directories
   - **Solution**: Clean test collection with proper path handling

### Working Directory Flexibility

Tests now work from **any directory**:

```bash
# From project root (/home/user/reward-kit/)
.venv/bin/pytest examples/frozen_lake_mcp_complete/test_*.py -v

# From examples directory (/home/user/reward-kit/examples/)
pytest frozen_lake_mcp_complete/test_*.py -v

# From project directory (/home/user/reward-kit/examples/frozen_lake_mcp_complete/)
pytest test_*.py -v
```

All commands produce identical results: ✅ **11 passed, 0 failed**

### Path Resolution

- **Before**: Relative paths like `"shared_data/rollouts.jsonl"` failed when working directory changed
- **After**: Absolute paths calculated from test file location: `os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_data", filename)`

### Pytest Configuration

The `pytest.ini` file provides:
- **Clean collection**: Excludes archive and problematic directories
- **Path setup**: Adds current directory to Python path
- **Warning filters**: Suppresses known deprecation warnings
- **Markers**: Organizes tests by category (integration, e2e, unit, slow)

## Why This Matters

The complete testing workflow exposes issues that simpler tests miss:

### Stack Hardening (Complexity)
1. **Memory management** - Longer sessions test resource cleanup
2. **State consistency** - Complex navigation tests state tracking
3. **Performance** - Multi-step scenarios test response times
4. **Edge cases** - Larger grids expose boundary condition bugs
5. **Scalability** - Proves system can handle production complexity

### End-to-End Integration (Full Stack)
6. **MCP protocol** - Tests real client-server communication
7. **Session management** - Validates proper connection lifecycle
8. **Configuration passing** - Ensures client context transmission
9. **Resource handling** - Tests tool/resource serving

### Trajectory Replay (Reliability)
10. **Deterministic behavior** - Catches non-deterministic bugs
11. **Regression testing** - Prevents behavior changes
12. **Integration reliability** - Ensures consistent CI/CD results
13. **Production confidence** - Validates reproducible results

### Pytest Integration (CI/CD Ready)
14. **Automated testing** - Integrates with standard CI/CD pipelines
15. **Parallel execution** - Speeds up test runs in distributed environments
16. **Coverage reporting** - Tracks test coverage for quality assurance
17. **JUnit compatibility** - Works with standard testing frameworks

Your stack is only as strong as its most comprehensive test suite! 🎯
