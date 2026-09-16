"""Evaluate the chatbot's RAG pipeline with RAGAS.

Runs every question in eval/testset.json through the same pipeline as the
/chat endpoint (backend/main.py:run_chat), then scores the results with:

  - faithfulness       Is every claim in the answer supported by the retrieved context?
  - answer_relevancy   Does the answer actually address the question?
  - context_precision  Are the relevant chunks ranked at the top of the retrieval?
  - context_recall     Does the retrieved context cover the facts in the reference answer?

plus a non-LLM retrieval check:

  - hit_rate@5         Is the article the question was written from among the top-5 chunks?
  - mrr                Mean reciprocal rank of that article.

Results are written to eval/results/<timestamp>/ (samples.csv + summary.json).

Usage:
    python eval/run_ragas.py                    # full run
    python eval/run_ragas.py --limit 3          # quick smoke test
    python eval/run_ragas.py --judge-model gpt-4o-mini
    python eval/run_ragas.py --context-chars 500 --label ctx500   # override chunk truncation
"""

import argparse
import asyncio
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
load_dotenv(os.path.join(ROOT, ".env"))

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
from ragas.prompt.metrics.base_prompt import BasePrompt  # noqa: E402

TESTSET_FILE = os.path.join(ROOT, "eval", "testset.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="LLM used by RAGAS to score answers (default: gpt-4o, stronger than the gpt-4o-mini generator)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel requests (default: 4)")
    parser.add_argument(
        "--context-chars",
        help="Override backend CONTEXT_CHARS for this run: an integer, or 'full' for whole chunks",
    )
    parser.add_argument("--label", default="", help="Suffix for the results folder name")
    parser.add_argument(
        "--prompt-language",
        default="german",
        help="Language RAGAS prompts are adapted to; must match the chatbot's language (default: german)",
    )
    return parser.parse_args()


# ------------- Step 1: run the chatbot pipeline -------------

def run_pipeline(samples: list[dict], concurrency: int, context_chars: str | None) -> list[dict]:
    # Imported here because it loads the embedding cache (~150 MB) on import
    import main as pipeline

    if context_chars is not None:
        pipeline.CONTEXT_CHARS = None if context_chars == "full" else int(context_chars)

    def run_one(sample: dict) -> dict:
        # Fresh history per question: each sample is an independent single-turn chat
        answer, results, contexts = pipeline.run_chat(sample["question"], history=[])
        retrieved_urls = [doc.get("url", "") for doc in results]
        rank = retrieved_urls.index(sample["source_url"]) + 1 if sample["source_url"] in retrieved_urls else None
        return {
            **sample,
            "response": answer,
            "retrieved_contexts": contexts,
            "retrieved_urls": retrieved_urls,
            "source_rank": rank,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = []
        for i, row in enumerate(pool.map(run_one, samples), 1):
            print(f"  [{i}/{len(samples)}] answered: {row['question'][:70]}...")
            rows.append(row)
    return rows


# ------------- Step 2: score with RAGAS -------------

async def adapt_prompts(metric, language: str, llm) -> None:
    """Translate a metric's few-shot examples into the evaluation language.

    RAGAS prompts are English by default. For answer_relevancy this matters a lot:
    it generates questions back from the German answer and compares their
    embeddings with the German user question, so English questions score low even
    for perfect answers (measured: 0.27 -> 0.82 on the same answer after adapting).
    """
    for attr, prompt in list(vars(metric).items()):
        if isinstance(prompt, BasePrompt):
            setattr(metric, attr, await prompt.adapt(language, llm))


async def score_rows(rows: list[dict], judge_model: str, concurrency: int, language: str) -> None:
    client = AsyncOpenAI()
    # Default max_tokens (1024) truncates faithfulness' statement extraction on long answers
    llm = llm_factory(judge_model, client=client, max_tokens=4096)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    metrics = {
        "faithfulness": (
            Faithfulness(llm=llm),
            lambda r: dict(user_input=r["question"], response=r["response"], retrieved_contexts=r["retrieved_contexts"]),
        ),
        "answer_relevancy": (
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            lambda r: dict(user_input=r["question"], response=r["response"]),
        ),
        "context_precision": (
            ContextPrecisionWithReference(llm=llm),
            lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["retrieved_contexts"]),
        ),
        "context_recall": (
            ContextRecall(llm=llm),
            lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["retrieved_contexts"]),
        ),
    }

    if language.lower() != "english":
        print(f"  adapting RAGAS prompts to {language}")
        await asyncio.gather(*(adapt_prompts(metric, language, llm) for metric, _ in metrics.values()))

    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(row: dict, name: str) -> None:
        metric, build_inputs = metrics[name]
        async with semaphore:
            try:
                result = await metric.ascore(**build_inputs(row))
                row[name] = float(result.value)
            except Exception as e:  # one failing judge call should not abort the whole run
                row[name] = math.nan
                row.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")

    tasks = [score_one(row, name) for row in rows for name in METRIC_NAMES]
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % len(METRIC_NAMES) == 0 or done == len(tasks):
            print(f"  scored {done}/{len(tasks)} metric calls")


# ------------- Step 3: report -------------

def mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_report(rows: list[dict], args) -> str:
    import main as pipeline

    folder = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else "")
    out_dir = os.path.join(RESULTS_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n_samples": len(rows),
        "generator_model": "gpt-4o-mini",
        "judge_model": args.judge_model,
        "prompt_language": args.prompt_language,
        "embedding_model": "text-embedding-3-small",
        "top_k": len(rows[0]["retrieved_urls"]) if rows else 0,
        "context_chars": pipeline.CONTEXT_CHARS or "full",
        "metrics": {name: round(mean([r[name] for r in rows]), 4) for name in METRIC_NAMES},
        "retrieval": {
            "hit_rate@5": round(sum(r["source_rank"] is not None for r in rows) / len(rows), 4),
            "mrr": round(sum(1 / r["source_rank"] for r in rows if r["source_rank"]) / len(rows), 4),
        },
        "failed_metric_calls": sum(len(r.get("errors", [])) for r in rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fields = ["question", "source_url", "source_rank", *METRIC_NAMES, "response", "reference", "retrieved_urls", "errors"]
    with open(os.path.join(out_dir, "samples.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                **r,
                **{name: round(r[name], 4) for name in METRIC_NAMES},
                "retrieved_urls": " | ".join(r["retrieved_urls"]),
                "errors": " | ".join(r.get("errors", [])),
            })

    print("\n=========== RAGAS Evaluation ===========")
    print(f"Samples: {summary['n_samples']}   Judge: {args.judge_model}   Commit: {summary['git_commit']}")
    for name, value in {**summary["metrics"], **summary["retrieval"]}.items():
        print(f"  {name:<20} {value:.3f}")
    if summary["failed_metric_calls"]:
        print(f"  ⚠️  {summary['failed_metric_calls']} metric calls failed (see samples.csv 'errors')")

    print("\nLowest faithfulness:")
    for r in sorted(rows, key=lambda r: (math.isnan(r["faithfulness"]), r["faithfulness"]))[:3]:
        print(f"  {r['faithfulness']:.2f}  {r['question'][:80]}")

    return out_dir


def main():
    args = parse_args()

    with open(TESTSET_FILE, encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Step 1/2: running {len(samples)} questions through the chatbot pipeline")
    rows = run_pipeline(samples, args.concurrency, args.context_chars)

    print(f"Step 2/2: scoring with RAGAS (judge: {args.judge_model})")
    asyncio.run(score_rows(rows, args.judge_model, args.concurrency, args.prompt_language))

    out_dir = write_report(rows, args)
    print(f"\nResults written to {os.path.relpath(out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
