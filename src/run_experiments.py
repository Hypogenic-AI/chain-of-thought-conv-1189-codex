#!/usr/bin/env python3
"""Run CoT vs no-CoT convergence experiments using real LLM APIs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_from_disk
from openai import OpenAI

SEED = 42
RNG = random.Random(SEED)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
LOG_DIR = ROOT / "logs"

DATASETS = {
    "gsm8k": ROOT / "datasets" / "gsm8k_main",
    "bbh_date": ROOT / "datasets" / "bbh_date_understanding",
    "bbh_logic": ROOT / "datasets" / "bbh_logical_deduction_three_objects",
    "persona": ROOT / "datasets" / "persona_chat",
}

SAMPLE_SIZES = {
    "gsm8k": 30,
    "bbh_date": 30,
    "bbh_logic": 30,
    "persona": 30,
}

MODELS = [
    {
        "name": "gpt-4.1",
        "provider": "openai",
    },
    {
        "name": "anthropic/claude-sonnet-4.5",
        "provider": "openrouter",
    },
]

CONDITIONS = ["no_cot", "cot"]
PERSONA_REPEATS = 2


@dataclass
class Clients:
    openai_client: OpenAI
    openrouter_client: OpenAI


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_clients() -> Clients:
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    return Clients(
        openai_client=OpenAI(api_key=openai_key),
        openrouter_client=OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"),
    )


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_final_answer(text: str) -> str:
    m = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_space(m.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_response(text: str) -> str:
    m = re.search(r"RESPONSE\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_space(m.group(1))
    return normalize_space(text)


def parse_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if nums:
        return nums[-1].replace(",", "")
    return normalize_space(answer)


def parse_gsm8k_pred(pred: str) -> str:
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", pred)
    if nums:
        return nums[-1].replace(",", "")
    return normalize_space(pred)


def parse_bbh_choice(text: str) -> str:
    m = re.search(r"\(([A-Z])\)", text)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-F])\b", text)
    if m2:
        return m2.group(1)
    return normalize_space(text)


def chat_completion(clients: Clients, model: str, provider: str, messages: List[Dict], temperature: float) -> str:
    client = clients.openai_client if provider == "openai" else clients.openrouter_client
    last_error = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=350,
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                # OpenAI SDK can occasionally return content blocks.
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)
    raise RuntimeError(f"API failure for {model}: {last_error}")


def make_reasoning_prompt(task_input: str, condition: str) -> List[Dict[str, str]]:
    if condition == "cot":
        user = (
            "Solve the problem. Provide concise reasoning, then put the final answer on its own line as "
            "'FINAL_ANSWER: <answer>'.\n\n"
            f"Problem:\n{task_input}"
        )
    else:
        user = (
            "Solve the problem and output only one line in this exact format: "
            "'FINAL_ANSWER: <answer>'. Do not include reasoning.\n\n"
            f"Problem:\n{task_input}"
        )
    return [
        {
            "role": "system",
            "content": "You are a careful reasoning assistant.",
        },
        {"role": "user", "content": user},
    ]


def make_persona_prompt(persona: List[str], history: List[str], condition: str) -> List[Dict[str, str]]:
    persona_text = "\n".join([f"- {p}" for p in persona])
    history_text = "\n".join([f"- {h}" for h in history[-4:]])

    if condition == "cot":
        user = (
            "Given this persona and conversation history, reply as the assistant. "
            "First provide one short reasoning line prefixed with 'REASONING:', then provide the final reply "
            "prefixed with 'RESPONSE:'. Keep response to 1-2 sentences.\n\n"
            f"Persona:\n{persona_text}\n\nHistory:\n{history_text}"
        )
    else:
        user = (
            "Given this persona and conversation history, reply as the assistant. "
            "Output one line prefixed with 'RESPONSE:' and keep it to 1-2 sentences.\n\n"
            f"Persona:\n{persona_text}\n\nHistory:\n{history_text}"
        )

    return [
        {
            "role": "system",
            "content": "Stay consistent with the provided persona while remaining natural.",
        },
        {"role": "user", "content": user},
    ]


def request_id(payload: Dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_existing(path: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[row["request_id"]] = row
    return records


def append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def stratified_sample_indices(total: int, k: int, rng: random.Random) -> List[int]:
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:k])


def build_data_manifest() -> Dict:
    gsm8k = load_from_disk(str(DATASETS["gsm8k"]))["test"]
    bbh_date = load_from_disk(str(DATASETS["bbh_date"]))["test"]
    bbh_logic = load_from_disk(str(DATASETS["bbh_logic"]))["test"]
    persona = load_from_disk(str(DATASETS["persona"]))["validation"]

    manifest = {
        "seed": SEED,
        "sizes": SAMPLE_SIZES,
        "indices": {
            "gsm8k": stratified_sample_indices(len(gsm8k), SAMPLE_SIZES["gsm8k"], RNG),
            "bbh_date": stratified_sample_indices(len(bbh_date), SAMPLE_SIZES["bbh_date"], RNG),
            "bbh_logic": stratified_sample_indices(len(bbh_logic), SAMPLE_SIZES["bbh_logic"], RNG),
            "persona": stratified_sample_indices(len(persona), SAMPLE_SIZES["persona"], RNG),
        },
    }

    return manifest


def save_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def main() -> None:
    set_seed()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    clients = load_clients()
    manifest_path = RAW_DIR / "sample_manifest.json"
    outputs_path = RAW_DIR / "model_outputs.jsonl"
    data_profile_path = RAW_DIR / "data_profile.json"
    run_meta_path = RAW_DIR / "run_metadata.json"

    manifest = build_data_manifest()
    save_json(manifest_path, manifest)

    # Data quality and profile summary.
    gsm8k = load_from_disk(str(DATASETS["gsm8k"]))["test"]
    bbh_date = load_from_disk(str(DATASETS["bbh_date"]))["test"]
    bbh_logic = load_from_disk(str(DATASETS["bbh_logic"]))["test"]
    persona = load_from_disk(str(DATASETS["persona"]))["validation"]

    profile = {
        "datasets": {
            "gsm8k_test_rows": len(gsm8k),
            "bbh_date_rows": len(bbh_date),
            "bbh_logic_rows": len(bbh_logic),
            "persona_validation_rows": len(persona),
        },
        "missing_checks": {
            "gsm8k_missing_question": int(sum(1 for x in gsm8k if not x["question"])),
            "bbh_date_missing_input": int(sum(1 for x in bbh_date if not x["input"])),
            "bbh_logic_missing_input": int(sum(1 for x in bbh_logic if not x["input"])),
            "persona_missing_personality": int(sum(1 for x in persona if not x["personality"])),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(data_profile_path, profile)

    existing = load_existing(outputs_path)
    completed_new = 0

    def maybe_call_and_store(payload: Dict, messages: List[Dict], temperature: float) -> None:
        nonlocal completed_new
        rid = request_id(payload)
        if rid in existing:
            return

        text = chat_completion(
            clients=clients,
            model=payload["model"],
            provider=payload["provider"],
            messages=messages,
            temperature=temperature,
        )
        row = {
            **payload,
            "request_id": rid,
            "raw_output": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(outputs_path, row)
        existing[rid] = row
        completed_new += 1
        if completed_new % 20 == 0:
            print(f"Completed new calls: {completed_new}", flush=True)

    # Reasoning tasks
    for model in MODELS:
        for condition in CONDITIONS:
            temp = 0.0

            for i in manifest["indices"]["gsm8k"]:
                ex = gsm8k[i]
                payload = {
                    "task": "gsm8k",
                    "item_id": int(i),
                    "repeat": 0,
                    "condition": condition,
                    "model": model["name"],
                    "provider": model["provider"],
                    "input": ex["question"],
                    "gold": parse_gsm8k_gold(ex["answer"]),
                }
                messages = make_reasoning_prompt(ex["question"], condition)
                maybe_call_and_store(payload, messages, temp)

            for task_name, dataset in [("bbh_date", bbh_date), ("bbh_logic", bbh_logic)]:
                for i in manifest["indices"][task_name]:
                    ex = dataset[i]
                    payload = {
                        "task": task_name,
                        "item_id": int(i),
                        "repeat": 0,
                        "condition": condition,
                        "model": model["name"],
                        "provider": model["provider"],
                        "input": ex["input"],
                        "gold": parse_bbh_choice(ex["target"]),
                    }
                    messages = make_reasoning_prompt(ex["input"], condition)
                    maybe_call_and_store(payload, messages, temp)

    # Persona task with repeated stochastic generations.
    for model in MODELS:
        for condition in CONDITIONS:
            temp = 0.7
            for i in manifest["indices"]["persona"]:
                ex = persona[i]
                for rep in range(PERSONA_REPEATS):
                    history = ex["utterances"][-1]["history"]
                    payload = {
                        "task": "persona",
                        "item_id": int(i),
                        "repeat": rep,
                        "condition": condition,
                        "model": model["name"],
                        "provider": model["provider"],
                        "input": "\n".join(history[-4:]),
                        "persona_profile": ex["personality"],
                    }
                    messages = make_persona_prompt(ex["personality"], history, condition)
                    maybe_call_and_store(payload, messages, temp)

    run_meta = {
        "seed": SEED,
        "models": MODELS,
        "conditions": CONDITIONS,
        "sample_sizes": SAMPLE_SIZES,
        "persona_repeats": PERSONA_REPEATS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "sample_manifest": str(manifest_path.relative_to(ROOT)),
            "outputs": str(outputs_path.relative_to(ROOT)),
            "data_profile": str(data_profile_path.relative_to(ROOT)),
        },
    }
    save_json(run_meta_path, run_meta)

    print("Experiment run complete.")
    print(f"Outputs: {outputs_path}")


if __name__ == "__main__":
    main()
