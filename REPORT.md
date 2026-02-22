# REPORT: Does Chain of Thought Cause Models to Converge More?

## 1. Executive Summary
This study tests whether Chain-of-Thought (CoT) prompting increases convergence in LLM behavior across (a) reasoning answers, (b) semantic output representations, and (c) persona behavior consistency.

Using real API models (`gpt-4.1` and `anthropic/claude-sonnet-4.5`) on local pre-gathered datasets (GSM8K, BBH date/logical subsets, Persona-Chat), CoT produced **mixed effects**: it improved GSM8K agreement/accuracy in some cases, but did not consistently improve convergence across tasks and significantly reduced persona stability for Claude.

Practical implication: CoT should not be assumed to globally increase model convergence. It can improve convergence on specific reasoning tasks while simultaneously reducing stability in persona-conditioned generation.

## 2. Goal
### Hypothesis
Investigate whether use of CoT causes increased convergence in model representations and persona, rather than only improving surface task performance.

### Why Important
If CoT changes reliability characteristics, this impacts deployment choices in multi-model systems, agent workflows, and safety-critical conversational settings.

### Problem Solved
Provides a unified measurement protocol across answer agreement, semantic convergence, and persona consistency under matched experimental controls.

### Expected Impact
Helps practitioners choose prompting strategies based on stability/convergence goals instead of accuracy-only optimization.

## 3. Data Construction
### Dataset Description
- GSM8K (`datasets/gsm8k_main/`): arithmetic reasoning benchmark, 1,319 test items.
- BBH Date Understanding (`datasets/bbh_date_understanding/`): 250 test items.
- BBH Logical Deduction (3 objects) (`datasets/bbh_logical_deduction_three_objects/`): 250 test items.
- Persona-Chat (`datasets/persona_chat/`): 1,000 validation dialogues.

### Example Samples
| Dataset | Input Example (truncated) | Label/Target |
|---|---|---|
| GSM8K | "Natalia sold clips... How many altogether?" | `72` |
| BBH Date | "Today is Christmas Eve of 1937..." | `(B)` |
| BBH Logic | "Three birds... fixed order..." | `(A)` |

Persona-Chat sample persona traits:
```text
- i like to remodel homes .
- i like to go hunting .
- i like to shoot a bow .
- my favorite holiday is halloween .
```

### Data Quality
From `results/raw/data_profile.json`:
- Missing values:
  - `gsm8k_missing_question`: 0
  - `bbh_date_missing_input`: 0
  - `bbh_logic_missing_input`: 0
  - `persona_missing_personality`: 0
- Outlier policy: no row deletion; bounded deterministic subsampling.
- Class distribution: N/A (task-specific mixed benchmarks).

### Preprocessing Steps
1. Deterministic subsampling with seed 42 (`results/raw/sample_manifest.json`):
   - GSM8K: 30
   - BBH date: 30
   - BBH logic: 30
   - Persona: 30
2. Gold-answer normalization:
   - GSM8K: parse numeric final answer after `####`.
   - BBH: parse option letter `(A)-(F)`.
3. Output parsing:
   - Reasoning tasks: parse `FINAL_ANSWER:`.
   - Persona tasks: parse `RESPONSE:`.
4. Embedding computation using `text-embedding-3-small` with cache (`results/raw/embedding_cache.json`).

### Train/Val/Test Splits
No model training/fine-tuning was performed.
- Evaluation-only protocol on benchmark test/validation splits.
- Deterministic sampled subsets used consistently across conditions.

## 4. Experiment Description
### Methodology
#### High-Level Approach
Compare `no_cot` vs `cot` prompt conditions across two real frontier models, then evaluate convergence with both discrete agreement metrics and continuous embedding-space metrics.

#### Why This Method
- Directly maps to hypothesis dimensions (answer, representation, persona).
- Uses objective labels where available (GSM8K/BBH).
- Adds persona-specific metrics for behavior stability.

Alternatives considered but not used:
- Latent activation-level convergence (not accessible for closed APIs).
- Human persona annotation (higher cost/time in this session).

### Implementation Details
#### Tools and Libraries
- Python 3.12
- `openai 2.21.0`
- `datasets 4.5.0`
- `numpy 2.3.5`
- `pandas 2.3.3`
- `scipy 1.17.0`
- `statsmodels 0.14.6`
- `matplotlib 3.10.8`
- `seaborn 0.13.2`

#### Algorithms/Models
- Model A: `gpt-4.1` (OpenAI).
- Model B: `anthropic/claude-sonnet-4.5` (via OpenRouter OpenAI-compatible API).
- Embeddings: `text-embedding-3-small`.

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| Seed | 42 | fixed reproducibility |
| Reasoning temperature | 0.0 | deterministic answers |
| Persona temperature | 0.7 | enable stochastic consistency checks |
| Max tokens | 350 | practical output cap |
| Persona repeats | 2 | minimal stability estimate |
| Alpha | 0.05 | standard significance threshold |
| Multiple-comparison correction | FDR (Benjamini-Hochberg) | controls false discovery rate |

#### Training Procedure or Analysis Pipeline
1. Run API calls with retries/backoff and request-hash caching in `src/run_experiments.py`.
2. Parse outputs and compute correctness/agreement metrics.
3. Embed output texts and persona profiles.
4. Compute cross-model semantic convergence and persona adherence/stability.
5. Run paired tests (`paired t` when normality held, else `Wilcoxon`).
6. Apply FDR correction over all hypothesis tests.

### Experimental Protocol
#### Reproducibility Information
- Number of runs for averaging: 1 main run + 1 deterministic analysis rerun.
- Random seed: 42.
- Hardware: 2x NVIDIA RTX 3090 (24GB each) available; API-based evaluation did not require GPU compute.
- Batch sizes: training/inference batch-size not applicable (remote API calls).
- Runtime: experiment execution ~23 minutes for 600 API outputs; analysis ~7 seconds.

#### Evaluation Metrics
- Accuracy (GSM8K/BBH): exact normalized match.
- Cross-model answer agreement: exact-match agreement between two models per item.
- Cross-model semantic cosine: cosine similarity between model output embeddings.
- Persona adherence: cosine(response embedding, persona-profile embedding).
- Persona stability: within-model cosine similarity across repeated stochastic generations.

### Raw Results
#### Tables
Reasoning accuracy:

| Task | Model | No-CoT | CoT |
|---|---|---:|---:|
| GSM8K | GPT-4.1 | 0.600 | 0.900 |
| GSM8K | Claude Sonnet 4.5 | 0.900 | 0.900 |
| BBH date | GPT-4.1 | 0.333 | 0.267 |
| BBH date | Claude Sonnet 4.5 | 0.900 | 0.833 |
| BBH logic | GPT-4.1 | 0.933 | 1.000 |
| BBH logic | Claude Sonnet 4.5 | 1.000 | 0.933 |

Cross-model answer agreement:

| Task | No-CoT | CoT |
|---|---:|---:|
| GSM8K | 0.633 | 0.900 |
| BBH date | 0.300 | 0.300 |
| BBH logic | 0.933 | 0.933 |

Persona metrics:

| Metric | Model | No-CoT | CoT |
|---|---|---:|---:|
| Persona adherence | GPT-4.1 | 0.385 | 0.389 |
| Persona adherence | Claude Sonnet 4.5 | 0.439 | 0.416 |
| Persona stability | GPT-4.1 | 0.758 | 0.735 |
| Persona stability | Claude Sonnet 4.5 | 0.857 | 0.753 |

#### Visualizations
- `results/plots/accuracy_by_task_condition.png`
- `results/plots/semantic_convergence.png`
- `results/plots/persona_adherence.png`
- `results/plots/persona_stability.png`

#### Output Locations
- Results JSON: `results/metrics.json`
- Statistical tests: `results/hypothesis_tests.csv`
- Parsed frame: `results/analysis_frame.csv`
- Plots: `results/plots/`
- Raw model outputs: `results/raw/model_outputs.jsonl`

## 5. Result Analysis
### Key Findings
1. CoT increased GSM8K performance and cross-model agreement for GPT-4.1, but effects were task-dependent.
2. CoT did **not** yield consistent convergence gains across all tasks; BBH date/logical showed neutral or reversed changes.
3. Persona stability decreased under CoT, strongest for Claude Sonnet 4.5.

### Hypothesis Testing Results
- Most tests were not significant after FDR correction.
- Strongest significant effect:
  - Persona stability (Claude): No-CoT > CoT
  - `p = 0.00085`, FDR-adjusted `p = 0.0145`, effect size `d = -0.679`, 95% CI for (CoT-NoCoT) `[-0.159, -0.049]`.
- Near-significant (uncorrected) but not FDR-significant:
  - GPT-4.1 GSM8K accuracy increase with CoT (`p = 0.0067`, FDR `0.0566`).
  - GSM8K cross-model agreement increase (`p = 0.0114`, FDR `0.0647`).

### Comparison to Baselines
- CoT outperformed no-CoT clearly on GSM8K for GPT-4.1.
- CoT did not outperform no-CoT on BBH date and partially regressed Claude on BBH logical/date.
- For persona outcomes, no-CoT was often as good or better.

### Surprises and Insights
- GPT-4.1 no-CoT underperformed sharply on sampled GSM8K, while CoT closed most of that gap.
- Claude showed high baseline persona stability, but CoT noticeably reduced it.
- Convergence behavior is axis-specific: gains in one domain do not transfer automatically.

### Error Analysis
- Reasoning failures:
  - BBH date mistakes often linked to option extraction/temporal confusion.
  - GPT-4.1 no-CoT GSM8K failures often incorrect final arithmetic value despite concise format.
- Persona failures:
  - CoT responses introduced explanatory meta-text that diluted style consistency.

### Limitations
- Small subset size (30 per benchmark segment) limits power.
- Persona metrics rely on embedding proxies, not human annotation.
- API models are black-box; no internal activation-level representation analysis.
- Two repeats for persona stability is minimal; more repeats would improve reliability.

## 6. Conclusions
CoT does not universally increase convergence. It improved convergence-related outcomes on GSM8K for GPT-4.1, but not on BBH tasks and significantly reduced persona stability for Claude Sonnet 4.5.

Overall, evidence supports a **conditional** view: CoT may increase convergence for some structured reasoning outcomes but can reduce persona-level consistency and does not reliably increase global cross-model convergence.

### Implications
- Practical: choose CoT selectively by task and desired stability axis.
- Theoretical: answer-level convergence and persona/semantic convergence should be modeled as distinct phenomena.

### Confidence in Findings
Moderate confidence for directional patterns, high confidence for the Claude persona-stability drop, lower confidence for borderline effects due to sample size.

## 7. Next Steps
### Immediate Follow-ups
1. Increase sample sizes (>=100 per task) to tighten confidence intervals and FDR robustness.
2. Add self-consistency decoding condition (multi-sample vote) to isolate CoT vs decoding effects.

### Alternative Approaches
- Human-rated persona consistency rubric with inter-annotator agreement.
- Entailment-based scoring for persona adherence instead of pure embedding cosine.

### Broader Extensions
- Extend to additional models (Gemini 2.5 Pro, GPT-5).
- Test adversarial persona prompts and long-horizon multi-turn drift.

### Open Questions
- Does hidden reasoning (no explicit rationale output) preserve GSM8K gains without persona stability costs?
- Are convergence effects primarily due to output-format constraints rather than reasoning itself?

## References
Primary references used in this study are listed in `literature_review.md` and `resources.md`, including:
- Wei et al. (2022), Kojima et al. (2022), Wang et al. (2022)
- Turpin et al. (2023), Xiao et al. (2023), Ley et al. (2024)
- Alikhani et al. (2024), Ai et al. (2024), Xu et al. (2024)
