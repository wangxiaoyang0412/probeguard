"""Counter-evidence probe pilot (intervention specified before evaluation data were inspected).

Actively stress-test unanimous-layer MedQA questions:
  1. Generate counter-hypothesis retrieval queries (thinking on, temp 0 — same spec as the
     base pipeline's conflict-query generation);
  2. Retrieve counter-evidence (BM25 + rerank, same params as the base pipeline);
  3. Re-answer one round with 8 candidates given the counter-evidence (devil's-advocate framing);
  4. Record the probe block: counter_queries / counter_docs / survival (keep vs flip).
Pilot scope: all unanimous wrong questions + an equal-size sample of unanimous correct
ones (seed 42) -> paired survival curves.
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--vendor-dir", required=True)
parser.add_argument("--full-jsonl", required=True, help="converted full-run JSONL (with rationales)")
parser.add_argument("--dataset-path", required=True, help="vendor dataset json (question text/options)")
parser.add_argument("--out", required=True)
parser.add_argument("--base-url", default=None)
parser.add_argument("--model-name", default="qwen3-8b")
parser.add_argument("--n-candidates", type=int, default=8)
parser.add_argument("--n-correct-sample", type=int, default=100)
parser.add_argument("--all-unanimous", action="store_true",
                    help="probe the whole unanimous layer (ignore sampling); resumable (append mode, skips existing qids)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--workers", type=int, default=6)
args = parser.parse_args()

VENDOR = Path(args.vendor_dir).resolve()
sys.path.insert(0, str(VENDOR))
import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from liquid import Template  # noqa: E402
from microservice import CustomLanguageModel  # noqa: E402
from utils import RetrievalService, combine_docs, get_query, inference, judger  # noqa: E402

SYS_COUNTER_QUERY = """\
### Role & Goal:
You are a skeptical medical expert. All assistants unanimously chose the same answer to a question. Your task is to actively search for evidence that could CHALLENGE or CONTRADICT that consensus: evidence supporting plausible alternative options, known exceptions, contraindications, or updated guidelines.

### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries, 2-4 total, for a **BM25** retriever)"""

USER_COUNTER_QUERY = Template("""\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

---

All assistants unanimously answered: {{consensus}}.

Generate 2-4 precise BM25 queries to retrieve evidence that could challenge this consensus (e.g., evidence favoring other options or against {{consensus}}). Give your queries in the given format.""")

SYS_REASK = """You are a medical assistant, please answer my medical questions. Give your final choice (capital option) closed in tag `<answer>X</answer>` after your analysis.
Your response should be as detailed as possible, but please do not use any subheadings."""

USER_REASK = Template("""\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Challenge Documents
A skeptical reviewer retrieved the following documents that may challenge the previous consensus:
{{documents}}

---

Previously, all assistants unanimously answered {{consensus}}. A devil's advocate now asks: in light of the challenge documents above, critically re-examine whether {{consensus}} is really correct, or whether another option is better supported. Weigh the challenge evidence seriously, but do not change your answer unless the evidence genuinely warrants it. Give your analysis and final choice.""")


def probe_one(item, model, retriever):
    import time as _t
    qd, consensus, correct = item
    t0 = _t.time()
    try:
        q = get_query(qd)
        # 1) counter-hypothesis query generation (thinking on / temp 0, same spec as base-pipeline conflict queries)
        _, qtext, qinfo = inference(
            SYS_COUNTER_QUERY,
            USER_COUNTER_QUERY.render(question=qd["question"], options=q["option_str"],
                                      consensus=consensus),
            model, enable_thinking=True, temperature=0.)[0]
        queries = list(set(s.strip() for s in
                           re.findall(r"\[Query .*?\](.*?)$", qtext, re.MULTILINE)))[:4]
        if not queries:
            return {"qid": qd["id"], "error": "no_queries"}
        # 2) counter-evidence retrieval (same params as the base pipeline)
        docs, doc_ids, scores = [], [], []
        rfunc = partial(retriever.retrieve, total_k=32, top_k=2,
                        combine_docs=False, use_reranker=True)
        for qq in queries:
            d, s = rfunc(qq)
            for doc, sc in zip(d, s):
                if doc["id"] not in doc_ids:
                    docs.append(doc)
                    doc_ids.append(doc["id"])
                    scores.append(float(sc))
        doc_text = combine_docs(docs, combine_sep="\n")
        # 3) re-answer with 8 candidates
        results = inference(
            SYS_REASK,
            USER_REASK.render(question=qd["question"], options=q["option_str"],
                              documents=doc_text, consensus=consensus),
            model, n=args.n_candidates)
        answers = []
        for r in results:
            ans, _ = judger(r[1], qd["answer"], single_answer=True)
            answers.append(ans)
        n_keep = sum(1 for a in answers if a == consensus)
        from collections import Counter
        maj_after = Counter(answers).most_common(1)[0][0]
        approx_tokens = len(qinfo.get("logprobs", [])) + \
            sum(len(r[2].get("logprobs", [])) for r in results)
        return {
            "qid": qd["id"], "consensus": consensus, "consensus_correct": correct,
            "elapsed_s": round(_t.time() - t0, 1), "approx_completion_tokens": approx_tokens,
            "counter_queries": queries, "counter_doc_ids": [str(d) for d in doc_ids],
            "counter_doc_scores": scores,
            "reask_answers": answers, "n_keep": n_keep,
            "keep_rate": n_keep / len(answers),
            "majority_after": maj_after, "flipped": maj_after != consensus,
            "flip_to_gold": (not correct) and maj_after == qd["answer"],
        }
    except Exception as e:  # one failed question must not kill the pilot
        return {"qid": qd["id"], "error": str(e)[:200]}


def main():
    random.seed(args.seed)
    recs = [json.loads(l) for l in open(args.full_jsonl, encoding="utf-8")]
    ua = [r for r in recs if r["rounds"][0]["agreement_rate"] == 1.0]
    if args.all_unanimous:
        sample = ua
    else:
        wrong = [r for r in ua if not r["correct"]]
        right = [r for r in ua if r["correct"]]
        sample = wrong + random.sample(right, min(args.n_correct_sample, len(right)))
    # resume support: append mode, skip existing qids
    done_qids = set()
    if Path(args.out).exists():
        for l in open(args.out, encoding="utf-8"):
            done_qids.add(json.loads(l)["qid"])
    dataset = {d["id"]: d for d in json.load(open(args.dataset_path, encoding="utf-8"))}
    items = []
    for r in sample:
        num_id = int(r["qid"].split("_")[-1])
        if num_id in done_qids:
            continue
        items.append((dataset[num_id], r["final_answer"], r["correct"]))
    print(f"probe: total_target={len(sample)} already_done={len(done_qids)} todo={len(items)}", flush=True)

    model = CustomLanguageModel(args.model_name, logger, base_url=args.base_url)
    retriever = RetrievalService()
    done = 0
    with open(args.out, "a", encoding="utf-8") as fp:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(lambda it: probe_one(it, model, retriever), items):
                fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                fp.flush()
                done += 1
                if done % 20 == 0:
                    print(f"probe progress {done}/{len(items)}")
    print(f"PROBE PILOT DONE -> {args.out}")


if __name__ == "__main__":
    main()
