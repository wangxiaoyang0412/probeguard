"""P(True) baseline (Kadavath et al. 2022, arXiv:2207.05221): post-hoc, one call per question.
Ask the model whether the given answer is correct, read the top-20 logprobs of the first
generated token, P(True) = p_true / (p_true + p_false). Thinking off, temp=0. The prompt
is fixed in this file.
Usage (server):
  python -m abstain_layer.ptrue_conf --vendor-dir ... --full-jsonl ... \
      --dataset-path ... --out results/ptrue_medqa.jsonl --base-url http://localhost:8000/v1
"""
import argparse
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SYS = ("You are a careful medical assistant. Judge whether a given answer to a "
       "multiple-choice question is correct. Reply with exactly one word: True or False.")
USER_TMPL = """### Question
{question}

### Options
{options}

### Proposed answer: {answer}

Is the proposed answer correct? Reply with exactly one word: True or False."""

MAX_SCAN_POSITIONS = 4  # scan the first few generated tokens for True/False mass


def truefalse_mass(top_logprobs: List[Any]) -> Tuple[float, float]:
    """Sum probability mass of True-like and False-like tokens at one position.

    Args:
        top_logprobs: list of objects/dicts with .token and .logprob.

    Returns:
        (p_true, p_false) raw probability masses.
    """
    p_true, p_false = 0.0, 0.0
    for entry in top_logprobs:
        tok = entry["token"] if isinstance(entry, dict) else entry.token
        lp = entry["logprob"] if isinstance(entry, dict) else entry.logprob
        t = tok.strip().lower()
        if t.startswith("true"):
            p_true += math.exp(lp)
        elif t.startswith("false"):
            p_false += math.exp(lp)
    return p_true, p_false


def score_ptrue(logprob_content: List[Any]) -> Tuple[Optional[float], float, float]:
    """Normalized P(True) from the first generated position carrying True/False mass.

    Args:
        logprob_content: per-token entries, each with .top_logprobs.

    Returns:
        (ptrue or None, p_true_raw, p_false_raw).
    """
    for pos in logprob_content[:MAX_SCAN_POSITIONS]:
        tops = pos["top_logprobs"] if isinstance(pos, dict) else pos.top_logprobs
        p_true, p_false = truefalse_mass(tops)
        if p_true + p_false > 0.0:
            return p_true / (p_true + p_false), p_true, p_false
    return None, 0.0, 0.0


def one(item: Tuple[Dict, str, bool], client: Any, model_name: str) -> Dict:
    qd, final, correct = item
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": USER_TMPL.format(
                          question=qd["question"], options=qd["option_str"], answer=final)}],
            temperature=0.0, max_tokens=4, logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        content = resp.choices[0].message.content or ""
        lp = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
        ptrue, p_t, p_f = score_ptrue(lp)
        parse = "logprob"
        if ptrue is None:  # fallback: parse the generated text itself
            t = content.strip().lower()
            if t.startswith("true"):
                ptrue, parse = 1.0, "text"
            elif t.startswith("false"):
                ptrue, parse = 0.0, "text"
            else:
                parse = "none"
        return {"qid": qd["id"], "correct": correct, "ptrue": ptrue,
                "p_true_raw": p_t, "p_false_raw": p_f, "gen": content[:40], "parse": parse}
    except Exception as e:  # noqa: BLE001 — record and continue; per-question fault tolerance
        return {"qid": qd["id"], "error": str(e)[:150]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-dir", required=True)
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.vendor_dir).resolve()))
    from utils import get_query  # noqa: E402 — vendor option_str formatting, same as verbalized baseline

    from openai import OpenAI  # noqa: E402

    recs = [json.loads(l) for l in open(args.full_jsonl)]
    dataset = {d["id"]: d for d in json.load(open(args.dataset_path))}
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["qid"] for l in open(args.out)}
    items = []
    for r in recs:
        idx = int(r["qid"].split("_")[-1])
        if idx in done:
            continue
        qd = dict(dataset[idx])
        qd["option_str"] = get_query(qd)["option_str"]
        items.append((qd, r["final_answer"], r["correct"]))
    print(f"ptrue: todo={len(items)} (done={len(done)})", flush=True)
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    n = 0
    with open(args.out, "a") as fp:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(lambda it: one(it, client, args.model_name), items):
                fp.write(json.dumps(res) + "\n")
                fp.flush()
                n += 1
                if n % 200 == 0:
                    print(f"progress {n}/{len(items)}", flush=True)
    print("PTRUE DONE", flush=True)


if __name__ == "__main__":
    main()
