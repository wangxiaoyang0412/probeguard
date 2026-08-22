"""Verbalized-confidence baseline (post-hoc self-reported 0-100, one call per question).
The prompt is fixed in this file. Usage (server):
  python -m abstain_layer.verbalized_conf --vendor-dir ... --full-jsonl ... \
      --dataset-path ... --out results/verbalized_medqa.jsonl --base-url http://localhost:8000/v1
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING)
from liquid import Template  # noqa: E402
from microservice import CustomLanguageModel  # noqa: E402
from utils import get_query, inference  # noqa: E402

SYS = ("You are a careful medical assistant. Rate how confident you are that a given "
       "answer to a multiple-choice question is correct. Reply with a single integer "
       "from 0 to 100 inside <conf></conf> tags, then one short sentence of justification.")
USER = Template("""### Question
{{question}}

### Options
{{options}}

### Proposed answer: {{answer}}

How confident are you (0-100) that this answer is correct? Reply with <conf>N</conf> first.""")


def one(item, model):
    qd, final, correct = item
    try:
        q = get_query(qd)
        _, text, _ = inference(SYS, USER.render(question=qd["question"],
                                                options=q["option_str"], answer=final),
                               model, temperature=0.)[0]
        m = re.search(r"<conf>\s*(\d+)\s*</conf>", text)
        conf = int(m.group(1)) if m else None
        return {"qid": qd["id"], "correct": correct, "verbalized_conf": conf,
                "parse_ok": m is not None}
    except Exception as e:
        return {"qid": qd["id"], "error": str(e)[:150]}


def main():
    recs = [json.loads(l) for l in open(args.full_jsonl)]
    dataset = {d["id"]: d for d in json.load(open(args.dataset_path))}
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["qid"] for l in open(args.out)}
    items = [(dataset[int(r["qid"].split("_")[-1])], r["final_answer"], r["correct"])
             for r in recs if int(r["qid"].split("_")[-1]) not in done]
    print(f"verbalized-conf: todo={len(items)} (done={len(done)})", flush=True)
    model = CustomLanguageModel(args.model_name, logging.getLogger(), base_url=args.base_url)
    n = 0
    with open(args.out, "a") as fp:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(lambda it: one(it, model), items):
                fp.write(json.dumps(res) + "\n")
                fp.flush()
                n += 1
                if n % 200 == 0:
                    print(f"progress {n}/{len(items)}", flush=True)
    print("VERBALIZED DONE", flush=True)


if __name__ == "__main__":
    main()
