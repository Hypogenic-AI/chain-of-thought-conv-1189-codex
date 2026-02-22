# CoT Convergence Study

This project runs a controlled, API-based evaluation of whether Chain-of-Thought (CoT) prompting increases convergence in modern LLM behavior.
It measures convergence across reasoning answers, semantic output similarity, and persona consistency using real models (`gpt-4.1` and `claude-sonnet-4.5`).

## Key Findings
- CoT improved GSM8K accuracy and cross-model agreement for GPT-4.1 on the sampled subset.
- CoT did not consistently improve convergence on BBH date/logical tasks.
- CoT significantly reduced persona stability for Claude Sonnet 4.5 (FDR-adjusted `p=0.0145`).
- Overall: CoT effects are task- and metric-dependent, not globally convergence-increasing.

## Reproduce
1. Activate env and install deps:
```bash
source .venv/bin/activate
uv sync
```
2. Ensure API keys are set:
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

3. Run experiments:
```bash
python src/run_experiments.py
```

4. Run analysis:
```bash
python src/analyze_results.py
```

## File Structure
- `planning.md`: motivation, novelty, and preregistered plan.
- `src/run_experiments.py`: data loading + API execution + caching.
- `src/analyze_results.py`: metrics, statistics, and plotting.
- `results/raw/model_outputs.jsonl`: raw model outputs.
- `results/metrics.json`: aggregated metrics and hypothesis tests.
- `results/plots/`: generated figures.
- `REPORT.md`: full research report.

See `REPORT.md` for full methodology, statistical tests, and limitations.
