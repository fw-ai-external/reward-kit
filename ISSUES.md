# Reward-Kit SDK Issues

# Mid pri issues

## Replicate the result for coding 
Check references/DeepCoder_ A Fully Open-Source 14B Coder at O3-mini Level.pdf , we want to reproduce the deep coder example minimally with TRL and with Qwen3 in our example. Check REPRO_CODING.md.


# Low pri issues

## Delete all legacy_reward_function 
Should not be needed at this point

## Fix the E2B unittest
```
tests/test_deepcoder_reward.py ....s.s...                                                                  [ 55%]
tests/test_fractional_code.py .....s..                                                                     [100%]
```
some of the tests are skipped due to E2B not working for me earlier
