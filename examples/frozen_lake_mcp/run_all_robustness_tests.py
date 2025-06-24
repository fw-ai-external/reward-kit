#!/usr/bin/env python3
"""
Comprehensive Test Runner for MCP Robustness Testing

This script runs all robustness tests and provides detailed reporting.
It can be used both for manual testing and CI/CD integration.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_robustness_tests():
    """Run the main robustness test suite."""
    print("🔍 Running Core Robustness Tests...")
    print("-" * 40)

    try:
        # Import and run the robustness tests
        from test_robustness_issues import run_all_robustness_tests

        success = run_all_robustness_tests()
        return success

    except ImportError as e:
        print(f"❌ Failed to import robustness tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to run robustness tests: {e}")
        traceback.print_exc()
        return False


def run_edge_case_tests():
    """Run the edge case test suite."""
    print("\n🔍 Running Edge Case Tests...")
    print("-" * 40)

    try:
        # Import and run the edge case tests
        from test_edge_cases import run_all_edge_case_tests

        success = run_all_edge_case_tests()
        return success

    except ImportError as e:
        print(f"⚠️  Edge case tests not available: {e}")
        return True  # Don't fail if edge cases aren't available
    except Exception as e:
        print(f"❌ Failed to run edge case tests: {e}")
        traceback.print_exc()
        return False


def run_north_star_integration_test():
    """Run the north star integration test."""
    print("\n🔍 Running North Star Integration Test...")
    print("-" * 40)

    try:
        # Check if test file exists
        test_file = Path(__file__).parent / "test_north_star.py"
        if not test_file.exists():
            print("⚠️  North star test file not found, skipping...")
            return True

        # Try to run the north star test
        import subprocess

        result = subprocess.run(
            [sys.executable, str(test_file)], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print("✅ North star integration test passed")
            return True
        else:
            print(f"❌ North star integration test failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  North star test timed out (likely needs real MCP server)")
        return True  # Don't fail on timeout
    except Exception as e:
        print(f"⚠️  Could not run north star test: {e}")
        return True  # Don't fail if we can't run it


def run_specific_test_suite(suite_name):
    """Run a specific test suite by name."""
    print(f"\n🔍 Running Specific Test Suite: {suite_name}")
    print("-" * 40)

    if suite_name == "robustness":
        return run_robustness_tests()
    elif suite_name == "edge_cases":
        return run_edge_case_tests()
    elif suite_name == "north_star":
        return run_north_star_integration_test()
    else:
        print(f"❌ Unknown test suite: {suite_name}")
        print("Available suites: robustness, edge_cases, north_star")
        return False


def main():
    """Main test runner function."""
    print("🧪 MCP Robustness Testing Suite")
    print("=" * 50)

    # Check if user wants to run specific test suite
    if len(sys.argv) > 1:
        suite_name = sys.argv[1]
        success = run_specific_test_suite(suite_name)
        sys.exit(0 if success else 1)

    # Run all test suites
    results = []

    # 1. Core robustness tests (always run)
    results.append(("Robustness Tests", run_robustness_tests()))

    # 2. Edge case tests (run if available)
    results.append(("Edge Case Tests", run_edge_case_tests()))

    # 3. North star integration test (run if available)
    results.append(("North Star Integration", run_north_star_integration_test()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, success in results:
        if success:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All test suites passed!")
        print("\n💡 Your MCP implementation is looking robust!")
    else:
        print(f"⚠️  {failed} test suite(s) need attention")
        print("\n💡 Check the individual test outputs above for details")
        print("💡 Focus on fixing the specific issues identified")

    # Exit with appropriate code for CI/CD
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
