# Cloned Repositories

## Repo 1: chain-of-thought-hub
- URL: https://github.com/FranxYao/chain-of-thought-hub
- Purpose: CoT-focused benchmark hub (GSM8K, MATH, MMLU, BBH and others)
- Location: `code/chain-of-thought-hub/`
- Key files: `code/chain-of-thought-hub/readme.md`, dataset subfolders (`gsm8k/`, `MMLU/`, `BBH/`, `MATH/`)
- Notes: Useful reference for CoT prompt templates and benchmark setup.

## Repo 2: lm-evaluation-harness
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Purpose: Standardized LLM evaluation framework across many tasks
- Location: `code/lm-evaluation-harness/`
- Key files: `code/lm-evaluation-harness/README.md`, `code/lm-evaluation-harness/lm_eval/`
- Notes: Good baseline infra for controlled CoT vs non-CoT experiments.

## Repo 3: simple-evals
- URL: https://github.com/openai/simple-evals
- Purpose: Lightweight eval implementations (MMLU, GPQA, MATH, MGSM, DROP, HumanEval, SimpleQA)
- Location: `code/simple-evals/`
- Key files: `code/simple-evals/README.md`, `code/simple-evals/simple_evals.py`
- Notes: Includes explicit zero-shot CoT style evaluation philosophy; useful for reproducible scoring pipelines.

## Requirements/Blockers
- Repos were cloned and inspected at README level.
- Full execution not run here because external model API keys and model backends are required for meaningful eval runs.
