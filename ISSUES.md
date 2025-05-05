# Next set of ticket items

## Support DeepSeek-V2 Prover reward function
- paper is under ./deepseek-prover-v2
  - do not proceed unless you can read the paper
- personally I am not familiar with Lean
- we also need to support testing with not just json dataset but also huggingface dataset, since the deepseek dataset is on HF `deepseek-ai/DeepSeek-ProverBench`
  - you should pull the dataset and check a few rows to get a sense on what the statement is about.


## Make sure dependencies are up to date
- check setup.py as well as all the imports we have for ./reward_kit, make sure that we have all the libraries necessary