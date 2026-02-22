#!/usr/bin/env python3
"""Analyze CoT vs no-CoT convergence experiment outputs."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "raw" / "model_outputs.jsonl"
OUT_METRICS = ROOT / "results" / "metrics.json"
OUT_FRAME = ROOT / "results" / "analysis_frame.csv"
PLOTS_DIR = ROOT / "results" / "plots"
CACHE_PATH = ROOT / "results" / "raw" / "embedding_cache.json"


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_embedding_cache() -> Dict[str, List[float]]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_embedding_cache(cache: Dict[str, List[float]]) -> None:
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def embed_texts(texts: Iterable[str], client: OpenAI, cache: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    missing = []
    for txt in texts:
        if txt in cache:
            out[txt] = np.array(cache[txt], dtype=np.float64)
        else:
            missing.append(txt)

    batch_size = 100
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        if not batch:
            continue
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for txt, emb in zip(batch, resp.data):
            vec = emb.embedding
            cache[txt] = vec
            out[txt] = np.array(vec, dtype=np.float64)

    # Include cached values loaded during loop.
    for txt in texts:
        if txt not in out:
            out[txt] = np.array(cache[txt], dtype=np.float64)

    return out


def paired_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float | str | None]:
    diffs = y - x
    if np.allclose(diffs, 0):
        return {
            "test": "degenerate",
            "p_value": 1.0,
            "statistic": 0.0,
            "effect_size_d": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "normality_p": None,
        }

    normality_p = None
    if len(diffs) >= 3:
        try:
            normality_p = float(shapiro(diffs).pvalue)
        except Exception:  # noqa: BLE001
            normality_p = None

    if normality_p is not None and normality_p > 0.05:
        stat, p = ttest_rel(y, x)
        test_name = "paired_t"
    else:
        stat, p = wilcoxon(y, x, zero_method="wilcox", correction=False)
        test_name = "wilcoxon"

    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    effect_d = float(np.mean(diffs) / std) if std > 0 else 0.0
    se = std / math.sqrt(len(diffs)) if len(diffs) > 0 else 0.0
    ci_low = float(np.mean(diffs) - 1.96 * se)
    ci_high = float(np.mean(diffs) + 1.96 * se)

    return {
        "test": test_name,
        "p_value": float(p),
        "statistic": float(stat),
        "effect_size_d": effect_d,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "normality_p": normality_p,
    }


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(RAW_PATH)
    if not rows:
        raise RuntimeError("No rows found in model_outputs.jsonl")

    records = []
    persona_profiles = {}

    for r in rows:
        task = r["task"]
        base = {
            "task": task,
            "item_id": int(r["item_id"]),
            "repeat": int(r.get("repeat", 0)),
            "condition": r["condition"],
            "model": r["model"],
            "provider": r["provider"],
            "gold": r.get("gold"),
            "raw_output": r["raw_output"],
        }

        if task == "gsm8k":
            ans = parse_gsm8k_pred(extract_final_answer(r["raw_output"]))
            gold = str(r["gold"])
            base.update(
                {
                    "prediction": ans,
                    "content_for_embedding": ans,
                    "correct": int(ans == gold),
                }
            )
        elif task in {"bbh_date", "bbh_logic"}:
            ans = parse_bbh_choice(extract_final_answer(r["raw_output"]))
            gold = parse_bbh_choice(str(r["gold"]))
            base.update(
                {
                    "prediction": ans,
                    "content_for_embedding": ans,
                    "correct": int(ans == gold),
                }
            )
        elif task == "persona":
            resp = extract_response(r["raw_output"])
            persona = r.get("persona_profile", [])
            persona_text = " ".join(persona)
            persona_profiles[(task, int(r["item_id"]))] = persona_text
            base.update(
                {
                    "prediction": resp,
                    "content_for_embedding": resp,
                    "correct": None,
                    "persona_text": persona_text,
                }
            )
        else:
            continue

        records.append(base)

    df = pd.DataFrame(records)
    df.to_csv(OUT_FRAME, index=False)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = ensure_embedding_cache()
    all_texts = set(df["content_for_embedding"].astype(str).tolist())
    all_texts.update([v for v in persona_profiles.values()])
    embeddings = embed_texts(all_texts, client, cache)
    save_embedding_cache(cache)

    # Attach embeddings and persona adherence.
    content_vecs = [embeddings[str(t)] for t in df["content_for_embedding"].tolist()]
    df["embedding_norm"] = [float(np.linalg.norm(v)) for v in content_vecs]

    persona_adherence = []
    for _, row in df.iterrows():
        if row["task"] == "persona":
            resp_vec = embeddings[str(row["content_for_embedding"])]
            ptxt = row["persona_text"]
            persona_vec = embeddings[ptxt]
            persona_adherence.append(cosine(resp_vec, persona_vec))
        else:
            persona_adherence.append(np.nan)
    df["persona_adherence"] = persona_adherence

    # Cross-model convergence on same task/item/repeat/condition.
    cross_rows = []
    grouped = df.groupby(["task", "item_id", "repeat", "condition"])
    for (task, item_id, repeat, condition), g in grouped:
        if len(g) != 2:
            continue
        a = g.iloc[0]
        b = g.iloc[1]
        va = embeddings[str(a["content_for_embedding"])]
        vb = embeddings[str(b["content_for_embedding"])]
        cross_rows.append(
            {
                "task": task,
                "item_id": item_id,
                "repeat": repeat,
                "condition": condition,
                "model_pair": f"{a['model']}|{b['model']}",
                "semantic_cosine": cosine(va, vb),
                "exact_agreement": int(str(a["prediction"]).strip() == str(b["prediction"]).strip())
                if task != "persona"
                else np.nan,
            }
        )
    cross_df = pd.DataFrame(cross_rows)

    # Persona within-model stability across repeats.
    stability_rows = []
    persona_df = df[df["task"] == "persona"].copy()
    for (model, condition, item_id), g in persona_df.groupby(["model", "condition", "item_id"]):
        if len(g) < 2:
            continue
        sims = []
        idx = g.index.tolist()
        for i, j in combinations(idx, 2):
            vi = embeddings[str(df.loc[i, "content_for_embedding"])]
            vj = embeddings[str(df.loc[j, "content_for_embedding"])]
            sims.append(cosine(vi, vj))
        stability_rows.append(
            {
                "model": model,
                "condition": condition,
                "item_id": item_id,
                "within_model_persona_stability": float(np.mean(sims)) if sims else np.nan,
            }
        )
    stability_df = pd.DataFrame(stability_rows)

    # Aggregated descriptive metrics.
    desc = defaultdict(dict)

    reasoning_df = df[df["task"].isin(["gsm8k", "bbh_date", "bbh_logic"])].copy()
    acc_table = (
        reasoning_df.groupby(["task", "model", "condition"])["correct"].mean().reset_index().rename(columns={"correct": "accuracy"})
    )

    cross_reasoning = cross_df[cross_df["task"].isin(["gsm8k", "bbh_date", "bbh_logic"])].copy()
    agreement_table = (
        cross_reasoning.groupby(["task", "condition"])["exact_agreement"]
        .mean()
        .reset_index()
        .rename(columns={"exact_agreement": "cross_model_answer_agreement"})
    )

    semantic_table = (
        cross_df.groupby(["task", "condition"])["semantic_cosine"].mean().reset_index().rename(columns={"semantic_cosine": "cross_model_semantic_cosine"})
    )

    persona_adherence_table = (
        persona_df.groupby(["model", "condition"])["persona_adherence"].mean().reset_index()
    )
    persona_stability_table = (
        stability_df.groupby(["model", "condition"])["within_model_persona_stability"].mean().reset_index()
    )

    # Inferential tests: CoT vs no-CoT paired by item.
    tests = []

    for task in ["gsm8k", "bbh_date", "bbh_logic"]:
        for model in df["model"].unique():
            g = reasoning_df[(reasoning_df["task"] == task) & (reasoning_df["model"] == model)]
            pivot = g.pivot_table(index="item_id", columns="condition", values="correct", aggfunc="first").dropna()
            if len(pivot) < 5:
                continue
            res = paired_test(pivot["no_cot"].to_numpy(dtype=float), pivot["cot"].to_numpy(dtype=float))
            tests.append(
                {
                    "metric": "accuracy",
                    "task": task,
                    "model": model,
                    "n": int(len(pivot)),
                    "mean_no_cot": float(pivot["no_cot"].mean()),
                    "mean_cot": float(pivot["cot"].mean()),
                    **res,
                }
            )

    # Cross-model answer agreement tests by task.
    for task in ["gsm8k", "bbh_date", "bbh_logic"]:
        g = cross_reasoning[cross_reasoning["task"] == task]
        pivot = g.pivot_table(index=["item_id", "repeat"], columns="condition", values="exact_agreement", aggfunc="first").dropna()
        if len(pivot) >= 5:
            res = paired_test(pivot["no_cot"].to_numpy(dtype=float), pivot["cot"].to_numpy(dtype=float))
            tests.append(
                {
                    "metric": "cross_model_answer_agreement",
                    "task": task,
                    "model": "cross_model",
                    "n": int(len(pivot)),
                    "mean_no_cot": float(pivot["no_cot"].mean()),
                    "mean_cot": float(pivot["cot"].mean()),
                    **res,
                }
            )

    # Persona adherence and stability tests.
    for model in persona_df["model"].unique():
        g = persona_df[persona_df["model"] == model]
        pivot = g.pivot_table(index=["item_id", "repeat"], columns="condition", values="persona_adherence", aggfunc="first").dropna()
        if len(pivot) >= 5:
            res = paired_test(pivot["no_cot"].to_numpy(dtype=float), pivot["cot"].to_numpy(dtype=float))
            tests.append(
                {
                    "metric": "persona_adherence",
                    "task": "persona",
                    "model": model,
                    "n": int(len(pivot)),
                    "mean_no_cot": float(pivot["no_cot"].mean()),
                    "mean_cot": float(pivot["cot"].mean()),
                    **res,
                }
            )

    for model in stability_df["model"].unique():
        g = stability_df[stability_df["model"] == model]
        pivot = g.pivot_table(index="item_id", columns="condition", values="within_model_persona_stability", aggfunc="first").dropna()
        if len(pivot) >= 5:
            res = paired_test(pivot["no_cot"].to_numpy(dtype=float), pivot["cot"].to_numpy(dtype=float))
            tests.append(
                {
                    "metric": "persona_stability",
                    "task": "persona",
                    "model": model,
                    "n": int(len(pivot)),
                    "mean_no_cot": float(pivot["no_cot"].mean()),
                    "mean_cot": float(pivot["cot"].mean()),
                    **res,
                }
            )

    # Cross-model semantic convergence tests by task.
    for task in cross_df["task"].unique():
        g = cross_df[cross_df["task"] == task]
        pivot = g.pivot_table(index=["item_id", "repeat"], columns="condition", values="semantic_cosine", aggfunc="first").dropna()
        if len(pivot) >= 5:
            res = paired_test(pivot["no_cot"].to_numpy(dtype=float), pivot["cot"].to_numpy(dtype=float))
            tests.append(
                {
                    "metric": "cross_model_semantic_cosine",
                    "task": task,
                    "model": "cross_model",
                    "n": int(len(pivot)),
                    "mean_no_cot": float(pivot["no_cot"].mean()),
                    "mean_cot": float(pivot["cot"].mean()),
                    **res,
                }
            )

    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        reject, p_adj, _, _ = multipletests(tests_df["p_value"].astype(float), alpha=0.05, method="fdr_bh")
        tests_df["p_value_fdr_bh"] = p_adj
        tests_df["significant_fdr_bh"] = reject

    # Save tables.
    acc_table.to_csv(ROOT / "results" / "accuracy_table.csv", index=False)
    agreement_table.to_csv(ROOT / "results" / "cross_model_agreement_table.csv", index=False)
    semantic_table.to_csv(ROOT / "results" / "semantic_convergence_table.csv", index=False)
    persona_adherence_table.to_csv(ROOT / "results" / "persona_adherence_table.csv", index=False)
    persona_stability_table.to_csv(ROOT / "results" / "persona_stability_table.csv", index=False)
    tests_df.to_csv(ROOT / "results" / "hypothesis_tests.csv", index=False)

    # Plots.
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=acc_table, x="task", y="accuracy", hue="condition", errorbar=None)
    plt.title("Reasoning Accuracy by Task and Condition")
    plt.ylim(0, 1)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_task_condition.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=semantic_table, x="task", y="cross_model_semantic_cosine", hue="condition", errorbar=None)
    plt.title("Cross-Model Semantic Convergence (Cosine)")
    plt.xlabel("Task")
    plt.ylabel("Cosine Similarity")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "semantic_convergence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=persona_adherence_table, x="model", y="persona_adherence", hue="condition", errorbar=None)
    plt.title("Persona Adherence by Model")
    plt.xlabel("Model")
    plt.ylabel("Persona Adherence (Cosine)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "persona_adherence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=persona_stability_table, x="model", y="within_model_persona_stability", hue="condition", errorbar=None)
    plt.title("Within-Model Persona Stability")
    plt.xlabel("Model")
    plt.ylabel("Cosine Similarity Across Repeats")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "persona_stability.png", dpi=200)
    plt.close()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": int(len(df)),
        "n_cross_records": int(len(cross_df)),
        "descriptive_tables": {
            "accuracy": acc_table.to_dict(orient="records"),
            "cross_model_answer_agreement": agreement_table.to_dict(orient="records"),
            "cross_model_semantic_convergence": semantic_table.to_dict(orient="records"),
            "persona_adherence": persona_adherence_table.to_dict(orient="records"),
            "persona_stability": persona_stability_table.to_dict(orient="records"),
        },
        "hypothesis_tests": tests_df.to_dict(orient="records"),
        "files": {
            "analysis_frame": str(OUT_FRAME.relative_to(ROOT)),
            "plots_dir": str(PLOTS_DIR.relative_to(ROOT)),
            "hypothesis_tests": "results/hypothesis_tests.csv",
        },
    }

    with OUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print(f"Metrics: {OUT_METRICS}")


if __name__ == "__main__":
    main()
