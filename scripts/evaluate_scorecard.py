#!/usr/bin/env python3
"""Reproducible API scorecards for a running turnkey endpoint.

The harness deliberately uses only the Python standard library and the
turnkey benchmark helper.  It implements two public, generation-based
contracts:

* ``gsm8k_cot`` adapts EleutherAI lm-evaluation-harness v3's fixed eight-shot
  prompt and strict/flexible numerical extraction to a chat endpoint. Chat
  turns end at EOS, so completion-only stop strings are deliberately omitted:
  they can occur inside a thinking trace and truncate the visible answer.
* ``gpqa_diamond`` follows OpenAI simple-evals' zero-shot prompt, deterministic
  answer permutation, and ``Answer: $LETTER`` extraction.

Every result records dataset source, selected source indices, request usage,
latency, raw answer text, correctness, and speculative-decoding metric deltas.
Use the same seed/limit/concurrency for a candidate/control comparison.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

from benchmark_serving import get_metrics, spec_summary


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
GPQA_DIAMOND_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)

GSM8K_FEWSHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6. The answer "
        "is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how "
        "many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The "
        "answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
        "12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many computers "
        "are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is "
        "29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end "
        "of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf "
        "balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money "
        "does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = "
        "15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The "
        "answer is 8.",
    ),
)

GPQA_PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""


def fetch_text(url: str, timeout: int = 120) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "turnkey-scorecard/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_gsm8k() -> tuple[list[dict[str, Any]], str]:
    raw = fetch_text(GSM8K_URL)
    rows = [json.loads(line) for line in raw.splitlines() if line]
    return rows, hashlib.sha256(raw.encode()).hexdigest()


def load_gpqa() -> tuple[list[dict[str, Any]], str]:
    raw = fetch_text(GPQA_DIAMOND_URL)
    rows = list(csv.DictReader(io.StringIO(raw)))
    return rows, hashlib.sha256(raw.encode()).hexdigest()


def select_indices(total: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= total:
        return list(range(total))
    return random.Random(seed).sample(range(total), limit)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "")
    if value.endswith("."):
        value = value[:-1]
    return value


def gsm8k_prompt(question: str) -> str:
    parts: list[str] = []
    for example, answer in GSM8K_FEWSHOT:
        parts.extend((f"Q: {example}", f"A: {answer}"))
    parts.extend((f"Q: {question}", "A:"))
    return "\n\n".join(parts)


def gsm8k_extract(text: str) -> str | None:
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+)\.?", text)
    if strict:
        return normalize_number(strict[-1])
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not flexible:
        return None
    left, right = flexible[-1]
    return normalize_number(left or right)


def extract_answer(task: str, text: str) -> str | None:
    if task == "gsm8k_cot":
        return gsm8k_extract(text)
    matches = re.findall(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", text)
    return matches[-1].upper() if matches else None


def make_examples(
    task: str, limit: int, seed: int
) -> tuple[list[dict[str, Any]], str, str]:
    if task == "gsm8k_cot":
        rows, dataset_sha256 = load_gsm8k()
        indices = select_indices(len(rows), limit, seed)
        examples = []
        for index in indices:
            row = rows[index]
            examples.append(
                {
                    "source_index": index,
                    "prompt": gsm8k_prompt(row["question"]),
                    "expected": normalize_number(row["answer"].split("####")[-1]),
                }
            )
        return examples, GSM8K_URL, dataset_sha256

    rows, dataset_sha256 = load_gpqa()
    rng = random.Random(seed)
    indices = (
        list(range(len(rows)))
        if limit <= 0 or limit >= len(rows)
        else rng.sample(range(len(rows)), limit)
    )
    examples = []
    for index in indices:
        row = rows[index]
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        permutation = rng.sample(range(4), 4)
        choices = [options[item] for item in permutation]
        expected = "ABCD"[choices.index(row["Correct Answer"])]
        examples.append(
            {
                "source_index": index,
                "permutation": permutation,
                "prompt": GPQA_PROMPT.format(
                    question=row["Question"],
                    a=choices[0],
                    b=choices[1],
                    c=choices[2],
                    d=choices[3],
                ),
                "expected": expected,
            }
        )
    return examples, GPQA_DIAMOND_URL, dataset_sha256


def api_request(
    url: str,
    model: str,
    example: dict[str, Any],
    task: str,
    max_tokens: int,
    timeout: int,
    api_key: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": example["prompt"]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed + int(example["source_index"]),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        return {
            "ok": False,
            "source_index": example["source_index"],
            "elapsed_s": round(time.perf_counter() - started, 6),
            "error": f"HTTP {error.code}: {detail}",
        }
    except Exception as error:  # noqa: BLE001 - evidence must retain transport failure
        return {
            "ok": False,
            "source_index": example["source_index"],
            "elapsed_s": round(time.perf_counter() - started, 6),
            "error": f"{type(error).__name__}: {error}",
        }

    message = document["choices"][0]["message"]
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    # Score the user-visible answer only. A tentative value in hidden thinking
    # is diagnostic evidence, not a completed response. Keeping the separate
    # extraction makes length/degeneration failures easy to classify.
    extracted = extract_answer(task, content)
    reasoning_extracted = extract_answer(task, reasoning)
    usage = document.get("usage") or {}
    return {
        "ok": True,
        "source_index": example["source_index"],
        "permutation": example.get("permutation"),
        "expected": example["expected"],
        "extracted": extracted,
        "reasoning_extracted": reasoning_extracted,
        "correct": extracted == example["expected"],
        "elapsed_s": round(time.perf_counter() - started, 6),
        "finish_reason": document["choices"][0].get("finish_reason"),
        "usage": usage,
        "content": content,
        "reasoning_content": reasoning,
    }


def write_atomic(path: str, document: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k_cot", "gpqa_diamond"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="repeatable key=value identity attached to the result",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metadata: dict[str, str] = {}
    for item in args.metadata:
        if "=" not in item or item.startswith("="):
            parser.error("--metadata values must be key=value")
        key, value = item.split("=", 1)
        metadata[key] = value

    examples, dataset_url, dataset_sha256 = make_examples(
        args.task, args.limit, args.seed
    )
    before = get_metrics(args.base_url.rstrip("/"), args.api_key)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                api_request,
                args.base_url,
                args.model,
                example,
                args.task,
                args.max_tokens,
                args.timeout,
                args.api_key,
                args.seed,
            )
            for example in examples
        ]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            good = sum(item.get("correct", False) for item in results)
            failed = sum(not item.get("ok", False) for item in results)
            print(
                f"[{count}/{len(futures)}] correct={good} errors={failed} "
                f"source={result['source_index']}",
                flush=True,
            )
    wall_s = time.perf_counter() - started
    after = get_metrics(args.base_url.rstrip("/"), args.api_key)
    results.sort(key=lambda item: item["source_index"])
    completed = [item for item in results if item.get("ok")]
    correct = sum(item["correct"] for item in completed)
    usage_prompt = sum(int(item["usage"].get("prompt_tokens", 0)) for item in completed)
    usage_output = sum(
        int(item["usage"].get("completion_tokens", 0)) for item in completed
    )
    document = {
        "schema": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "task": args.task,
        "contract": (
            "EleutherAI lm-evaluation-harness gsm8k_cot v3 eight-shot, chat adaptation"
            if args.task == "gsm8k_cot"
            else "OpenAI simple-evals GPQAEval zero-shot"
        ),
        "dataset_url": dataset_url,
        "dataset_sha256": dataset_sha256,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata": metadata,
        "seed": args.seed,
        "requested_limit": args.limit,
        "concurrency": args.concurrency,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "summary": {
            "requests": len(results),
            "completed": len(completed),
            "errors": len(results) - len(completed),
            "correct": correct,
            "accuracy": round(correct / len(completed), 6) if completed else None,
            "wall_s": round(wall_s, 6),
            "prompt_tokens": usage_prompt,
            "completion_tokens": usage_output,
            "prompt_throughput_tok_s": round(usage_prompt / wall_s, 3),
            "output_throughput_tok_s": round(usage_output / wall_s, 3),
            "mean_request_s": round(
                statistics.mean(item["elapsed_s"] for item in completed), 6
            )
            if completed
            else None,
            "speculative": spec_summary(before, after),
        },
        "results": results,
    }
    write_atomic(args.out, document)
    print(json.dumps(document["summary"], indent=1), flush=True)
    return 0 if len(completed) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
