## Resources Catalog

### Summary
This document catalogs all resources gathered for the project: papers, datasets, and code repositories for testing whether CoT increases model convergence.

### Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Chain-of-Thought Prompting Elicits Reasoning in LLMs | Wei et al. | 2022 | `papers/2201.11903_chain_of_thought_prompting_elicits_reasoning_in_large_langua.pdf` | Foundational CoT method |
| Large Language Models are Zero-Shot Reasoners | Kojima et al. | 2022 | `papers/2205.11916_large_language_models_are_zero_shot_reasoners.pdf` | Zero-shot CoT baseline |
| Self-Consistency Improves CoT Reasoning | Wang et al. | 2022 | `papers/2203.11171_self_consistency_improves_chain_of_thought_reasoning_in_lang.pdf` | Multi-trajectory convergence |
| Language Models Don't Always Say What They Think | Turpin et al. | 2023 | `papers/2305.04388_language_models_don_t_always_say_what_they_think_unfaithful_.pdf` | CoT unfaithfulness |
| Faithful Chain-of-Thought Reasoning | Xiao et al. | 2023 | `papers/2301.13379_faithful_chain_of_thought_reasoning.pdf` | Faithfulness-focused CoT |
| On the Hardness of Faithful CoT Reasoning in LLMs | Ley et al. | 2024 | `papers/2406.10625_on_the_hardness_of_faithful_chain_of_thought_reasoning_in_la.pdf` | Accuracy-faithfulness tension |
| Chain-of-Thought Hub | Fu et al. | 2023 | `papers/2305.17306_chain_of_thought_hub_a_continuous_effort_to_measure_large_la.pdf` | Benchmark suite |
| Evaluating LLM Biases in Persona-Steered Generation | Alikhani et al. | 2024 | `papers/2405.20253_evaluating_large_language_model_biases_in_persona_steered_ge.pdf` | Persona steering effects |
| Is Self-knowledge and Action Consistent or Not | Ai et al. | 2024 | `papers/2402.14679_is_self_knowledge_and_action_consistent_or_not_investigating.pdf` | Persona congruence metrics |
| Self-Cognition in LLMs | Xu et al. | 2024 | `papers/2407.01505_self_cognition_in_large_language_models_an_exploratory_study.pdf` | Self-consistency probes |

See `papers/README.md` for details.

### Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K (main) | HuggingFace `openai/gsm8k` | 7,473 train / 1,319 test | Math reasoning | `datasets/gsm8k_main/` | Core CoT benchmark |
| TruthfulQA (generation) | HuggingFace `truthfulqa/truthful_qa` | 817 val | Truthfulness | `datasets/truthfulqa_generation/` | Faithfulness-relevant |
| Persona-Chat | HuggingFace `AlekseyKorshuk/persona-chat` | 17,878 train / 1,000 val | Persona consistency | `datasets/persona_chat/` | Persona drift analysis |
| BBH date_understanding | HuggingFace `lukaemon/bbh` | 250 test | Temporal reasoning | `datasets/bbh_date_understanding/` | Controlled reasoning subset |
| BBH logical_deduction_three_objects | HuggingFace `lukaemon/bbh` | 250 test | Logical reasoning | `datasets/bbh_logical_deduction_three_objects/` | Controlled reasoning subset |

See `datasets/README.md` for download and loading instructions.

### Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| chain-of-thought-hub | https://github.com/FranxYao/chain-of-thought-hub | CoT benchmark references | `code/chain-of-thought-hub/` | Includes GSM8K/MMLU/BBH tooling |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | Standardized eval framework | `code/lm-evaluation-harness/` | Useful for baseline automation |
| simple-evals | https://github.com/openai/simple-evals | Lightweight eval scripts | `code/simple-evals/` | CoT-oriented evaluation style |

See `code/README.md` for details.

### Resource Gathering Notes

#### Search Strategy
- Started with paper-finder output (broad recall).
- Performed targeted arXiv retrieval for CoT, faithfulness, and persona consistency.
- Prioritized papers and datasets enabling direct CoT vs non-CoT comparisons.

#### Selection Criteria
- Direct relevance to convergence/faithfulness/persona consistency.
- Public accessibility and reproducibility.
- Availability of benchmark datasets or reusable evaluation code.

#### Challenges Encountered
- One persona dataset (`bavard/personachat_truecased`) was incompatible with current `datasets` package behavior (legacy script format).
- Replaced with compatible `AlekseyKorshuk/persona-chat` dataset.

#### Gaps and Workarounds
- Limited standardized latent-representation convergence benchmarks.
- Workaround: combine answer convergence (self-consistency), faithfulness proxies, and persona congruence metrics.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: GSM8K + BBH subsets + TruthfulQA + Persona-Chat.
2. **Baseline methods**: direct answer, zero-shot CoT, few-shot CoT, self-consistency CoT.
3. **Evaluation metrics**: accuracy, answer agreement across sampled traces, faithfulness proxy metrics, persona congruence score.
4. **Code to adapt/reuse**: `code/lm-evaluation-harness/` for task execution; `code/chain-of-thought-hub/` for benchmark conventions; `code/simple-evals/` for lightweight sanity checks.

## Research Execution Log (2026-02-22)
- Implemented and ran controlled CoT vs no-CoT experiments using real API models: `gpt-4.1` (OpenAI) and `anthropic/claude-sonnet-4.5` (OpenRouter).
- Used local pre-gathered datasets: GSM8K, BBH date/logical subsets, and Persona-Chat.
- Saved raw outputs and reproducibility artifacts to `results/raw/`.
- Computed statistical analyses and visualizations in `results/`.
- Main findings documented in `REPORT.md`.
