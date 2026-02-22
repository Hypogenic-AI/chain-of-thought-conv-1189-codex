# Literature Review: Does Chain-of-Thought Cause Models to Converge More?

## Research Area Overview
This review studies whether Chain-of-Thought (CoT) prompting increases convergence in large language models (LLMs), where convergence means stable internal reasoning traces, stable outputs across samples/prompts, and stable persona behavior across contexts. The literature indicates a mixed pattern: CoT often improves task accuracy and can improve answer-level convergence (e.g., under self-consistency), but textual rationales are frequently unfaithful to latent computation, and persona-conditioned behavior can still drift.

## Review Scope

### Research Question
Does CoT cause LLMs to converge more in representation and persona, or does it mainly improve surface-level answer quality without faithful internal convergence?

### Inclusion Criteria
- CoT methods or CoT faithfulness analysis
- LLM reasoning benchmarks (math, logic, commonsense)
- Persona/self-consistency studies in LLM outputs
- Publicly accessible papers and reproducible datasets/code

### Exclusion Criteria
- Papers unrelated to LLM reasoning or persona consistency
- Non-public or inaccessible papers
- Purely theoretical work without empirical LLM evaluation

### Time Frame
2022-2024 (plus foundational context)

### Sources
- paper-finder output (local service)
- arXiv retrieval
- HuggingFace datasets
- GitHub repositories

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-02-22 | "chain of thought convergence model representations persona large language models" | paper-finder | 308 | Used as broad candidate pool; manually curated for direct relevance |
| 2026-02-22 | "chain of thought prompting large language models reasoning" | arXiv API | multiple | Selected foundational CoT papers |
| 2026-02-22 | "chain of thought faithfulness language models" | arXiv API | multiple | Selected faithfulness-focused papers |
| 2026-02-22 | "persona consistency large language models" | arXiv API | multiple | Selected persona/self-knowledge papers |

## Screening Results

| Paper | Title Screen | Abstract Screen | Full-Text/Chunk Screen | Notes |
|------|--------------|-----------------|------------------------|-------|
| 2201.11903 | Include | Include | Include (deep) | Foundational CoT effects and scaling |
| 2205.11916 | Include | Include | Include | Zero-shot CoT baseline |
| 2203.11171 | Include | Include | Include | Convergence via self-consistency voting |
| 2305.04388 | Include | Include | Include (deep) | Unfaithful CoT explanations |
| 2301.13379 | Include | Include | Include | Faithful CoT methods |
| 2406.10625 | Include | Include | Include (deep) | Faithfulness-accuracy tradeoff |
| 2305.17306 | Include | Include | Include | Benchmark infrastructure |
| 2405.20253 | Include | Include | Include | Persona-steered behavior analysis |
| 2402.14679 | Include | Include | Include (deep) | Self-knowledge/action consistency |
| 2407.01505 | Include | Include | Include | Self-cognition consistency signals |

## Key Papers

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Jason Wei et al.
- **Year**: 2022
- **Source**: arXiv (2201.11903)
- **Key Contribution**: Demonstrates large gains from CoT prompting for sufficiently large models.
- **Methodology**: Few-shot exemplars with intermediate reasoning steps.
- **Datasets Used**: GSM8K, StrategyQA, CSQA, Date Understanding, Sports Understanding.
- **Results**: Strong accuracy gains at larger scales; CoT can hurt smaller models.
- **Code Available**: Partly via subsequent benchmark repos (e.g., CoT Hub).
- **Relevance to Our Research**: Baseline evidence that CoT can increase output stability/accuracy, but not proof of internal convergence.

### Paper 2: Large Language Models are Zero-Shot Reasoners
- **Authors**: Takeshi Kojima et al.
- **Year**: 2022
- **Source**: arXiv (2205.11916)
- **Key Contribution**: Introduces zero-shot trigger for reasoning ("Let's think step by step").
- **Methodology**: Prompting-only intervention without demonstrations.
- **Datasets Used**: Arithmetic and commonsense reasoning tasks.
- **Results**: Major zero-shot gains, especially on multi-step tasks.
- **Code Available**: No official full benchmark package; easy to replicate.
- **Relevance to Our Research**: Enables controlled CoT-vs-non-CoT ablation with minimal confounds.

### Paper 3: Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors**: Xuezhi Wang et al.
- **Year**: 2022
- **Source**: arXiv (2203.11171)
- **Key Contribution**: Replaces greedy CoT with sampled reasoning paths plus answer marginalization.
- **Methodology**: Decode multiple CoTs, choose consensus answer.
- **Datasets Used**: GSM8K and other reasoning benchmarks.
- **Results**: Consistent gains across tasks.
- **Code Available**: Community implementations exist.
- **Relevance to Our Research**: Directly operationalizes convergence as agreement across multiple trajectories.

### Paper 4: Language Models Don't Always Say What They Think
- **Authors**: Miles Turpin et al.
- **Year**: 2023
- **Source**: arXiv (2305.04388)
- **Key Contribution**: Shows CoT explanations can be systematically unfaithful.
- **Methodology**: Bias-feature interventions and explanation-behavior mismatch analysis.
- **Datasets Used**: BBH variants, BBQ.
- **Results**: CoT text often omits true decision factors.
- **Code Available**: Partial artifacts in paper resources.
- **Relevance to Our Research**: Evidence against equating textual convergence with representation convergence.

### Paper 5: Faithful Chain-of-Thought Reasoning
- **Authors**: Yaozong Xiao et al.
- **Year**: 2023
- **Source**: arXiv (2301.13379)
- **Key Contribution**: Proposes methods to improve rationale faithfulness.
- **Methodology**: Faithfulness-aware constraints/objectives for generated rationales.
- **Datasets Used**: Reasoning benchmarks with explanation settings.
- **Results**: Better explanation faithfulness while retaining performance.
- **Code Available**: Limited public release (check paper links).
- **Relevance to Our Research**: Candidate baseline when testing whether CoT can be made to converge internally.

### Paper 6: On the Hardness of Faithful CoT Reasoning in LLMs
- **Authors**: Dan Ley et al.
- **Year**: 2024
- **Source**: arXiv (2406.10625)
- **Key Contribution**: Finds it difficult to jointly optimize accuracy and CoT faithfulness.
- **Methodology**: Early-answering faithfulness metrics, intervention and in-context strategies.
- **Datasets Used**: AQuA, LogiQA, TruthfulQA.
- **Results**: No single intervention consistently improves both faithfulness and accuracy.
- **Code Available**: Referenced in paper materials.
- **Relevance to Our Research**: Strong recent evidence that convergence in explicit CoT does not guarantee latent convergence.

### Paper 7: Chain-of-Thought Hub
- **Authors**: Yao Fu et al.
- **Year**: 2023
- **Source**: arXiv (2305.17306)
- **Key Contribution**: Centralized evaluation across complex reasoning tasks.
- **Methodology**: Unified prompts/leaderboards for CoT-heavy tasks.
- **Datasets Used**: GSM8K, MATH, MMLU, BBH, HumanEval.
- **Results**: Highlights large variance across models/tasks.
- **Code Available**: Yes (GitHub).
- **Relevance to Our Research**: Provides practical evaluation backbone.

### Paper 8: Evaluating LLM Biases in Persona-Steered Generation
- **Authors**: Malihe Alikhani et al.
- **Year**: 2024
- **Source**: arXiv (2405.20253)
- **Key Contribution**: Quantifies behavior changes under persona steering.
- **Methodology**: Controlled persona prompts and bias analysis.
- **Datasets Used**: Persona-oriented generation settings.
- **Results**: Persona prompts alter outputs and can amplify biases.
- **Code Available**: Limited.
- **Relevance to Our Research**: Useful for persona-convergence analysis under CoT.

### Paper 9: Is Self-knowledge and Action Consistent or Not
- **Authors**: Yiming Ai et al.
- **Year**: 2024
- **Source**: arXiv (2402.14679)
- **Key Contribution**: Measures mismatch between claimed personality and generated actions.
- **Methodology**: Questionnaire-vs-scenario congruence metrics.
- **Datasets Used**: Custom personality statements/scenarios.
- **Results**: LLMs show weaker congruence than humans.
- **Code Available**: Not prominent.
- **Relevance to Our Research**: Direct operationalization of persona convergence.

### Paper 10: Self-Cognition in LLMs: An Exploratory Study
- **Authors**: Weichen Xu et al.
- **Year**: 2024
- **Source**: arXiv (2407.01505)
- **Key Contribution**: Explores self-cognition consistency in model outputs.
- **Methodology**: Self-referential probing tasks.
- **Datasets Used**: Self-cognition evaluation sets.
- **Results**: Incomplete and model-dependent self-consistency.
- **Code Available**: Limited.
- **Relevance to Our Research**: Additional evidence for/against persona and self-representation convergence.

## Common Methodologies
- CoT prompting: used in 2201.11903, 2205.11916, 2305.17306.
- Multi-sample consensus (self-consistency): 2203.11171.
- Faithfulness evaluation with intervention/truncation: 2305.04388, 2406.10625.
- Persona/self-consistency probes: 2405.20253, 2402.14679, 2407.01505.

## Standard Baselines
- Direct answer prompting (no CoT).
- Zero-shot CoT prompt trigger.
- Few-shot CoT exemplars.
- Self-consistency decoding over CoT samples.
- Faithfulness-aware CoT variants (where available).

## Evaluation Metrics
- Task accuracy (GSM8K, BBH subtasks, TruthfulQA).
- Agreement/convergence metrics across multiple CoT samples.
- Faithfulness metrics (early-answering style and intervention sensitivity).
- Persona congruence metrics (self-report vs behavior alignment).
- Bias sensitivity metrics under controlled persona cues.

## Datasets in the Literature
- GSM8K: used for arithmetic reasoning and CoT gains.
- BBH: used for challenging reasoning and bias-intervention analyses.
- TruthfulQA: used for truthfulness and faithfulness stress testing.
- Persona-oriented dialogue/scenario sets: used for consistency and bias analyses.

## Gaps and Opportunities
- Gap 1: Few studies jointly evaluate answer convergence, latent-faithfulness convergence, and persona convergence in one protocol.
- Gap 2: Most CoT evaluations rely on textual traces; latent-state convergence measures are less standardized.
- Gap 3: Persona consistency benchmarks are less mature than reasoning benchmarks.

## Recommendations for Our Experiment
- **Recommended datasets**: GSM8K + BBH (date/logical) for reasoning, TruthfulQA for reliability, Persona-Chat for persona drift.
- **Recommended baselines**: no-CoT, zero-shot CoT, few-shot CoT, self-consistency CoT.
- **Recommended metrics**: answer accuracy, inter-sample agreement, faithfulness proxy via truncation/intervention, persona congruence score.
- **Methodological considerations**:
  - Separate answer-level convergence from explanation faithfulness.
  - Keep decoding settings fixed except CoT intervention.
  - Track convergence over repeated seeds/prompts.
  - Evaluate persona stability across both neutral and adversarial prompts.

## Deep Reading Notes
Detailed chunk-by-chunk reading artifacts for four key papers are stored in:
- `papers/pages/`
- `papers/deep_read_chunks_notes.md`
