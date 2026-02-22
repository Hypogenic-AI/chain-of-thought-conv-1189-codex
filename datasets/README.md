# Downloaded Datasets

This directory contains local datasets for experiments on CoT effects on reasoning convergence and persona consistency.

Data files are intentionally excluded from git via `datasets/.gitignore`.

## Dataset 1: GSM8K (main)
- Source: `openai/gsm8k` (HuggingFace)
- Local path: `datasets/gsm8k_main/`
- Size: train 7,473; test 1,319
- Task: grade-school math reasoning
- Format: HuggingFace `DatasetDict`
- License: see dataset card on HuggingFace

### Download Instructions
Using HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k_main")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k_main")
```

### Sample Data
`datasets/gsm8k_main/samples/samples.json`

## Dataset 2: TruthfulQA (generation)
- Source: `truthfulqa/truthful_qa` (HuggingFace)
- Local path: `datasets/truthfulqa_generation/`
- Size: validation 817
- Task: truthfulness and factual robustness
- Format: HuggingFace `DatasetDict`
- License: see dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa_generation")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthfulqa_generation")
```

### Sample Data
`datasets/truthfulqa_generation/samples/samples.json`

## Dataset 3: Persona-Chat
- Source: `AlekseyKorshuk/persona-chat` (HuggingFace)
- Local path: `datasets/persona_chat/`
- Size: train 17,878; validation 1,000
- Task: persona-conditioned dialogue consistency
- Format: HuggingFace `DatasetDict`
- License: see dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("AlekseyKorshuk/persona-chat")
dataset.save_to_disk("datasets/persona_chat")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/persona_chat")
```

### Sample Data
`datasets/persona_chat/samples/samples.json`

## Dataset 4: BBH (Date Understanding)
- Source: `lukaemon/bbh` config `date_understanding`
- Local path: `datasets/bbh_date_understanding/`
- Size: test 250
- Task: symbolic/temporal reasoning
- Format: HuggingFace `DatasetDict`

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("lukaemon/bbh", "date_understanding")
dataset.save_to_disk("datasets/bbh_date_understanding")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/bbh_date_understanding")
```

### Sample Data
`datasets/bbh_date_understanding/samples/samples.json`

## Dataset 5: BBH (Logical Deduction: Three Objects)
- Source: `lukaemon/bbh` config `logical_deduction_three_objects`
- Local path: `datasets/bbh_logical_deduction_three_objects/`
- Size: test 250
- Task: logical deduction
- Format: HuggingFace `DatasetDict`

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
dataset.save_to_disk("datasets/bbh_logical_deduction_three_objects")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/bbh_logical_deduction_three_objects")
```

### Sample Data
`datasets/bbh_logical_deduction_three_objects/samples/samples.json`

## Quick Validation Summary
- Verified splits and record counts for all datasets.
- Saved per-dataset sample files with first 5 examples per split.
- Summary JSON: `datasets/datasets_summary.json`.
