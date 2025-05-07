"""
Example of using the DeepSeek-Prover-V2 reward function for evaluating Lean proofs.

This example demonstrates:
1. Using the lean_prover_reward function for basic evaluation
2. Using the deepseek_prover_v2_reward function for more advanced evaluation
3. Using the deepseek_huggingface_prover_benchmark function for evaluation against the ProverBench dataset

Requirements:
- For the deepseek_huggingface_prover_benchmark, install with: pip install "reward-kit[deepseek]"
"""

from reward_kit.rewards.lean_prover import (
    lean_prover_reward,
    deepseek_prover_v2_reward,
    deepseek_huggingface_prover_benchmark,
)


def main():
    # Example statement and response
    statement = "For all natural numbers n, the sum of the first n natural numbers is n(n+1)/2."

    # Example of a partial proof with "sorry"
    partial_response = """
    theorem sum_naturals (n : ℕ) : ∑ i in range n, i = n * (n + 1) / 2 :=
    begin
      induction n with d hd,
      { simp, },
      { sorry }
    end
    """

    # Example of a more complete proof with subgoal decomposition
    complete_response = """
    theorem sum_naturals (n : ℕ) : ∑ i in range n, i = n * (n + 1) / 2 :=
    begin
      -- We'll prove this by induction on n
      induction n with d hd,
      -- Base case: n = 0
      { simp, },
      -- Inductive step: assume true for n = d, prove for n = d + 1
      { 
        have step1 : ∑ i in range (d + 1), i = (∑ i in range d, i) + d,
          by simp [sum_range_succ],
        have step2 : (∑ i in range d, i) + d = d * (d + 1) / 2 + d,
          by rw [hd],
        have step3 : d * (d + 1) / 2 + d = (d * (d + 1) + 2 * d) / 2,
          by ring,
        have step4 : (d * (d + 1) + 2 * d) / 2 = (d * (d + 1) + 2 * d) / 2,
          by refl,
        have step5 : d * (d + 1) + 2 * d = d * (d + 3),
          by ring,
        have step6 : d * (d + 3) = (d + 1) * (d + 2) - 2,
          by ring,
        have step7 : (d + 1) * (d + 2) - 2 = (d + 1) * ((d + 1) + 1) - 2,
          by refl,
        calc
          ∑ i in range (d + 1), i = (∑ i in range d, i) + d : by simp [sum_range_succ]
          ... = d * (d + 1) / 2 + d : by rw [hd]
          ... = (d * (d + 1) + 2 * d) / 2 : by ring
          ... = (d * (d + 1 + 2)) / 2 : by ring
          ... = (d * (d + 3)) / 2 : by ring
          ... = ((d + 1) * (d + 2) - 2) / 2 : by rw [step6]
          ... = ((d + 1) * (d + 2)) / 2 - 1 : by ring
          ... = (d + 1) * ((d + 1) + 1) / 2 : by ring,
      }
    end
    """

    # Basic evaluation with lean_prover_reward
    basic_partial_result = lean_prover_reward(
        response=partial_response, statement=statement, verbose=True
    )

    basic_complete_result = lean_prover_reward(
        response=complete_response, statement=statement, verbose=True
    )

    # Advanced evaluation with deepseek_prover_v2_reward
    advanced_partial_result = deepseek_prover_v2_reward(
        response=partial_response,
        statement=statement,
        check_subgoals=True,
        verbose=True,
    )

    advanced_complete_result = deepseek_prover_v2_reward(
        response=complete_response,
        statement=statement,
        check_subgoals=True,
        verbose=True,
    )

    # Print results
    print("Basic evaluation (partial proof):")
    print(f"Score: {basic_partial_result.score}")
    if (
        hasattr(basic_partial_result, "metrics")
        and basic_partial_result.metrics
    ):
        print("Metrics:")
        for metric_name, metric in basic_partial_result.metrics.items():
            print(f"  {metric_name}: {metric.score} - {metric.reason}")

    print("\nBasic evaluation (complete proof):")
    print(f"Score: {basic_complete_result.score}")
    if (
        hasattr(basic_complete_result, "metrics")
        and basic_complete_result.metrics
    ):
        print("Metrics:")
        for metric_name, metric in basic_complete_result.metrics.items():
            print(f"  {metric_name}: {metric.score} - {metric.reason}")

    print("\nAdvanced evaluation (partial proof):")
    print(f"Score: {advanced_partial_result.score}")
    if (
        hasattr(advanced_partial_result, "metrics")
        and advanced_partial_result.metrics
    ):
        print("Metrics:")
        for metric_name, metric in advanced_partial_result.metrics.items():
            print(f"  {metric_name}: {metric.score} - {metric.reason}")

    print("\nAdvanced evaluation (complete proof):")
    print(f"Score: {advanced_complete_result.score}")
    if (
        hasattr(advanced_complete_result, "metrics")
        and advanced_complete_result.metrics
    ):
        print("Metrics:")
        for metric_name, metric in advanced_complete_result.metrics.items():
            print(f"  {metric_name}: {metric.score} - {metric.reason}")

    # Optional: Evaluate against the DeepSeek-ProverBench dataset
    # Requires the "datasets" package, see requirements at the top
    try:
        from datasets import load_dataset

        # 1. First approach: Create a mock dataset item
        mock_dataset_item = {
            "id": "sum_naturals",
            "statement": statement,
            "expected_proof": None,  # No exact match expected
        }

        benchmark_result = deepseek_huggingface_prover_benchmark(
            response=complete_response,
            statement=statement,
            dataset_item=mock_dataset_item,
            verbose=True,
        )

        # 2. Second approach: Use the HuggingFace dataset directly
        print(
            "\n\nLoading actual DeepSeek-ProverBench dataset from HuggingFace..."
        )

        try:
            # Load the actual dataset (this will download it if not already available)
            dataset = load_dataset(
                "deepseek-ai/DeepSeek-ProverBench", split="train"
            )

            if len(dataset) > 0:
                print(
                    f"Successfully loaded dataset with {len(dataset)} examples"
                )

                # Get a sample statement from the dataset
                sample_idx = 0
                sample_statement = dataset[sample_idx].get("statement", "")

                if sample_statement:
                    print(f"\nSample statement: {sample_statement[:100]}...")

                    # For demonstration, we'll use our existing proof
                    # In a real scenario, you would generate a proof for this specific statement
                    print("\nEvaluating with automatic dataset matching...")

                    hf_result = deepseek_huggingface_prover_benchmark(
                        response=complete_response,  # Using our existing proof as example
                        statement=sample_statement,  # Using statement from HF dataset
                        dataset_name="deepseek-ai/DeepSeek-ProverBench",  # Will search in this dataset
                        verbose=True,
                    )

                    print(f"Score: {hf_result.score}")
                    if "dataset_match" in hf_result.metrics:
                        match_info = hf_result.metrics["dataset_match"]
                        print(
                            f"Dataset match: {match_info.score} - {match_info.reason}"
                        )
            else:
                print("Dataset loaded but appears to be empty.")
        except Exception as e:
            print(f"Error loading or processing actual dataset: {str(e)}")

        print("\nBenchmark evaluation (against ProverBench):")
        print(f"Score: {benchmark_result.score}")
        if hasattr(benchmark_result, "metrics") and benchmark_result.metrics:
            print("Metrics:")
            for metric_name, metric in benchmark_result.metrics.items():
                print(f"  {metric_name}: {metric.score} - {metric.reason}")

    except ImportError:
        print(
            "\nSkipping benchmark evaluation - 'datasets' package not installed."
        )
        print("Install with: pip install reward-kit[deepseek]")


if __name__ == "__main__":
    main()
