# Research Plan: Does Chain of Thought Cause Models to Converge More?

## Motivation & Novelty Assessment

### Why This Research Matters
LLM outputs are increasingly used in settings where consistency across prompts, samples, and model families matters for trust and deployment safety. If Chain of Thought (CoT) changes convergence behavior, it affects reliability, auditing, and multi-model interoperability. Understanding whether CoT amplifies or reduces convergence helps practitioners choose prompting strategies for stable behavior rather than only peak accuracy.

### Gap in Existing Work
Prior work has shown that CoT can improve task accuracy and self-consistency, while separate literature shows rationale unfaithfulness and persona drift. However, the pre-gathered review indicates few studies jointly measure (a) answer-level convergence, (b) semantic representation convergence, and (c) persona convergence under matched prompting conditions across modern API models.

### Our Novel Contribution
We run a unified, controlled evaluation that compares no-CoT vs CoT across two real frontier models, quantifying convergence at three levels: answer agreement, embedding-space output convergence, and persona adherence/consistency. This directly tests whether CoT increases global convergence or only local task agreement.

### Experiment Justification
- Experiment 1: Reasoning-task convergence on GSM8K + BBH subsets. Needed to test whether CoT changes answer agreement and accuracy on tasks with objective labels.
- Experiment 2: Representation convergence in embedding space. Needed because answer match alone can hide divergent rationales and semantic trajectories.
- Experiment 3: Persona convergence on Persona-Chat. Needed to test whether CoT stabilizes or destabilizes persona expression under conversational generation.

## Research Question
Does Chain-of-Thought prompting lead to increased convergence in LLM behavior (answers, semantic representations, and persona expression), or does it primarily improve task performance without broader convergence?

## Background and Motivation
Literature review in `literature_review.md` shows a mixed picture: CoT and self-consistency often improve accuracy, but faithfulness studies report mismatches between stated reasoning and latent decision factors. Persona studies indicate nontrivial drift across contexts. This motivates a direct empirical test of convergence beyond accuracy.

## Hypothesis Decomposition
- H1 (Answer convergence): CoT increases answer agreement and task accuracy on structured reasoning tasks.
- H2 (Representation convergence): CoT increases semantic convergence of model outputs (higher cosine similarity in embedding space).
- H3 (Persona convergence): CoT decreases persona consistency by encouraging verbose task-oriented reasoning that dilutes role-conditioned style.
- H0: No significant difference between no-CoT and CoT on convergence metrics.

Independent variables:
- Prompt condition: no-CoT vs CoT.
- Model family: OpenAI GPT-4.1 and Claude Sonnet 4.5 (via OpenRouter).
- Task type: reasoning vs persona response generation.

Dependent variables:
- Accuracy, exact-match agreement, cross-model agreement.
- Embedding cosine similarity (within condition, across models/runs).
- Persona adherence similarity and within-model response consistency.

Confounds and controls:
- Control decoding and temperature by condition.
- Use identical datasets/examples across conditions.
- Use fixed random seeds and cached API responses.

## Proposed Methodology

### Approach
Use real API models and a common evaluation harness. Sample matched subsets from pre-downloaded datasets. Run no-CoT and CoT prompts under controlled settings, store raw outputs, compute metrics, and run paired statistical tests with effect sizes.

### Experimental Steps
1. Load and validate local datasets (`datasets/*`), extract task-ready examples.
2. Build standardized prompt templates for no-CoT and CoT.
3. Query two real models with retry + caching and structured outputs.
4. Compute reasoning metrics (accuracy, agreement) for labeled tasks.
5. Compute semantic convergence via OpenAI embeddings on responses.
6. Compute persona adherence and consistency on Persona-Chat samples.
7. Run statistical tests (paired t-test or Wilcoxon based on normality) and effect sizes.
8. Generate plots/tables and compile report.

### Baselines
- Baseline A: Direct answer prompting (no-CoT).
- Baseline B: CoT prompting (single sample).
- Diagnostic baseline: random-answer sanity floor for labeled tasks.

### Evaluation Metrics
- Task accuracy (GSM8K, BBH).
- Cross-model exact-answer agreement rate.
- Embedding convergence: cosine similarity between model outputs per item.
- Persona adherence: cosine(response, persona profile embedding).
- Persona stability: average pairwise cosine across repeated generations for same model-condition-item.

### Statistical Analysis Plan
- Significance level: alpha = 0.05.
- Normality check: Shapiro-Wilk on paired differences.
- Primary tests: paired t-test if normal, otherwise Wilcoxon signed-rank.
- Report p-values, 95% CI of mean difference, Cohen's d for paired effects.
- Correct multiple comparisons with Benjamini-Hochberg FDR.

## Expected Outcomes
Support for "CoT causes more convergence" would require significant improvements in multiple convergence metrics (not just accuracy). Mixed outcomes (e.g., higher accuracy but unchanged or worse persona convergence) would support partial-convergence interpretations.

## Timeline and Milestones
- M1 (20 min): setup, resource verification, planning.
- M2 (45 min): implement evaluation pipeline and data loading.
- M3 (60 min): run API experiments with caching.
- M4 (40 min): analysis, statistics, plots.
- M5 (30 min): REPORT.md + README.md + validation rerun.

## Potential Challenges
- API transient failures or rate limits: handled with retries/backoff and caching.
- Output formatting variability: robust parsers and fallback extraction.
- Cost/time growth: bounded sample sizes and deterministic subsampling.
- Persona metric noise: use multiple runs and confidence intervals.

## Success Criteria
- End-to-end reproducible pipeline in `src/` with saved raw outputs.
- Statistical comparison of no-CoT vs CoT on all planned metrics.
- Clear evidence in `REPORT.md` that either supports or refutes increased convergence.
