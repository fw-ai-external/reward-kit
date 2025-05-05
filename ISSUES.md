# Next set of ticket items

## cpp test is failing right now 

reward_kit/rewards/cpp_code.py:757: TypeError
===================================================================================================================== warnings summary =====================================================================================================================
.venv/lib/python3.12/site-packages/pydantic/_internal/_config.py:323
.venv/lib/python3.12/site-packages/pydantic/_internal/_config.py:323
  /home/bchen/home/reward-kit/.venv/lib/python3.12/site-packages/pydantic/_internal/_config.py:323: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================================================================= short test summary info ==================================================================================================================
FAILED tests/test_cpp_code.py::TestOutputComparison::test_whitespace_normalization - assert 1.0 < 1.0
FAILED tests/test_cpp_code.py::TestPistonClient::test_get_runtimes - TypeError: TestPistonClient.test_get_runtimes.<locals>.mock_aenter() takes 0 positional arguments but 1 was given
FAILED tests/test_cpp_code.py::TestPistonClient::test_execute_success - TypeError: TestPistonClient.test_execute_success.<locals>.mock_aenter() takes 0 positional arguments but 1 was given
FAILED tests/test_cpp_code.py::TestPistonClient::test_execute_compile_error - TypeError: TestPistonClient.test_execute_compile_error.<locals>.mock_aenter() takes 0 positional arguments but 1 was given
FAILED tests/test_cpp_code.py::TestPistonClient::test_execute_runtime_error - TypeError: TestPistonClient.test_execute_runtime_error.<locals>.mock_aenter() takes 0 positional arguments but 1 was given
FAILED tests/test_cpp_code.py::TestExecuteCppCode::test_execute_cpp_success - assert False is True
FAILED tests/test_cpp_code.py::TestExecuteCppCode::test_execute_cpp_compile_error - AssertionError: assert 'Compilation error' in 'Unknown error during execution'
FAILED tests/test_cpp_code.py::TestExecuteCppCode::test_execute_c_code - assert False is True
FAILED tests/test_cpp_code.py::TestIOICppCodeReward::test_success_match - TypeError: '_asyncio.Future' object is not subscriptable
FAILED tests/test_cpp_code.py::TestIOICppCodeReward::test_success_mismatch - TypeError: '_asyncio.Future' object is not subscriptable
FAILED tests/test_cpp_code.py::TestIOICppCodeReward::test_execution_failure - TypeError: '_asyncio.Future' object is not subscriptable
FAILED tests/test_cpp_code.py::TestIOICppCodeReward::test_multiple_test_cases - TypeError: object of type '_asyncio.Future' has no len()
================================================================================================== 12 failed, 242 passed, 2 skipped, 2 warnings in 7.82s ===================================================================================================

## Support uploading of HuggingFace datasets to Fireworks dataset (requires Fireworks dataset API integration)

People should be able to just specify a huggingface dataset for evaluation job, and we should still make everything run

## TRL adapter for reward functions

We want to make sure the reward functions can be used with TRL as well with GRPO. I downloaded the grpo trainer code into TRL cookbooks, please check it out before implementing the TRL adapter and then make sure our reward functions can be used inside TRL as well.