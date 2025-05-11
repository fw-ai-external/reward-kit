# Reward-Kit SDK Issues

# High pri issues

## Data filtering script still not filtering things properly

```
python scripts/convert_hf_math_to_openai_jsonl.py open-r1/OpenR1-Math-220k ./runs/openr1_numeric.jsonl --filter_by_match --math_type numeric
```

for example this row passed 

```
{
  "messages": [
    {
      "role": "user",
      "content": "(7) Let $z \\in \\mathbf{C}$, and satisfy $|z-\\mathrm{i}| \\leqslant 1, A=\\operatorname{Re}(z)\\left(|z-\\mathrm{i}|^{2}-1\\right)$. Find $\\max A$ and $\\min A$."
    },
    {
      "role": "assistant",
      "content": "(Answer: When $z=-\\frac{\\sqrt{3}}{3}+\\mathrm{i}$, $\\max A=\\frac{2 \\sqrt{3}}{9}$; when $z=\\frac{\\sqrt{3}}{3}+\\mathrm{i}$, $\\min A=-\\frac{2 \\sqrt{3}}{9}$ )"
    }
  ],
  "ground_truth_answer_from_column": "\\maxA=\\frac{2\\sqrt{3}}{9},\\A=-\\frac{2\\sqrt{3}}{9}",
  "match_details": {
    "filter_passed": true,
    "reward_score": 1,
    "match_comparison_reason": "Best match: Gen='2' (2.0) vs Orig='2' (2.0).\nNumeric match: Yes, Similarity: 1.000",
    "math_type_used_for_filter": "numeric",
    "extracted_from_solution_column": "Extracted from generated: '3' (3.0), '3' (3.0), '2' (2.0), '3' (3.0), '9' (9.0), '3' (3.0), '3' (3.0), '2' (2.0), '3' (3.0), '9' (9.0)",
    "extracted_from_gt_answer_column": "Extracted from original: '2' (2.0), '3' (3.0), '9' (9.0), '2' (2.0), '3' (3.0), '9' (9.0)"
  }
}
```

you can see the ground truth answer is not even a numerical number, but it is getting filtered

# Mid pri issues

## Move deepseek deps into coding
Should be part of main package

# Low pri issues

## Delete all legacy_reward_function 
Should not be needed at this point

## Fix the E2B unittest
```
tests/test_deepcoder_reward.py ....s.s...                                                                  [ 55%]
tests/test_fractional_code.py .....s..                                                                     [100%]
```
some of the tests are skipped due to E2B not working for me earlier
