"""Probe control arms: isolate what drives the counter-evidence flip asymmetry.

Three single-variable controls against the original counter-evidence probe
(probe_pilot.py, which stored per-question counter_queries in its output JSONL):

  arm=nodocs   devil's-advocate instruction, NO documents
               -> measures pure instruction / social pressure;
  arm=randdocs devil's-advocate instruction + irrelevant MedCorp documents
               (retrieved with a seeded cyclic derangement of the stored
               counter-queries: real corpus documents, wrong question)
               -> measures "documents present at all";
  arm=neutral  neutral instruction + the SAME counter-evidence documents
               (deterministic re-retrieval from this question's stored queries)
               -> measures evidence content without adversarial framing.

All arms reuse the stored queries (no query generation) and the original
retrieval parameters (BM25 total_k=32 + reranker top_k=2, dedup by doc id).
Output schema matches probe_pilot.py plus {arm, src_qid, n_docs}.
Resumable: append mode, existing qids in --out are skipped.
"""
import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--vendor-dir", required=True)
parser.add_argument("--probe-jsonl", required=True,
                    help="original probe output with counter_queries per qid")
parser.add_argument("--dataset-path", required=True, help="vendor dataset json")
parser.add_argument("--arm", required=True, choices=["nodocs", "randdocs", "neutral"])
parser.add_argument("--out", required=True)
parser.add_argument("--base-url", default=None)
parser.add_argument("--model-name", default="qwen3-8b")
parser.add_argument("--n-candidates", type=int, default=8)
parser.add_argument("--limit", type=int, default=0, help="smoke: only first N items")
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

SYS_REASK = """You are a medical assistant, please answer my medical questions. Give your final choice (capital option) closed in tag `<answer>X</answer>` after your analysis.
Your response should be as detailed as possible, but please do not use any subheadings."""

# arm=nodocs: identical devil's-advocate framing, documents block removed.
USER_NODOCS = Template("""\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

---

Previously, all assistants unanimously answered {{consensus}}. A devil's advocate now asks: critically re-examine whether {{consensus}} is really correct, or whether another option is better supported. Reconsider seriously, but do not change your answer unless the reconsideration genuinely warrants it. Give your analysis and final choice.""")

# arm=randdocs: byte-identical to probe_pilot.py's USER_REASK; only the documents differ.
USER_WITHDOCS = Template("""\
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

# arm=neutral: same documents, adversarial framing removed (consensus anchor kept).
USER_NEUTRAL = Template("""\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Additional Documents
A reviewer retrieved the following additional documents related to the question:
{{documents}}

---

Previously, all assistants unanimously answered {{consensus}}. Taking the documents above into account, give your analysis and final choice.""")


def retrieve_docs(queries, retriever):
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
    return docs, doc_ids, scores


def probe_one(item, model, retriever):
    import time as _t
    qd, consensus, correct, queries, src_qid = item
    t0 = _t.time()
    try:
        q = get_query(qd)
        doc_ids, scores = [], []
        if args.arm == "nodocs":
            user = USER_NODOCS.render(question=qd["question"], options=q["option_str"],
                                      consensus=consensus)
        else:
            docs, doc_ids, scores = retrieve_docs(queries, retriever)
            doc_text = combine_docs(docs, combine_sep="\n")
            tmpl = USER_WITHDOCS if args.arm == "randdocs" else USER_NEUTRAL
            user = tmpl.render(question=qd["question"], options=q["option_str"],
                               documents=doc_text, consensus=consensus)
        results = inference(SYS_REASK, user, model, n=args.n_candidates)
        answers = []
        for r in results:
            ans, _ = judger(r[1], qd["answer"], single_answer=True)
            answers.append(ans)
        n_keep = sum(1 for a in answers if a == consensus)
        maj_after = Counter(answers).most_common(1)[0][0]
        approx_tokens = sum(len(r[2].get("logprobs", [])) for r in results)
        return {
            "qid": qd["id"], "arm": args.arm, "src_qid": src_qid,
            "consensus": consensus, "consensus_correct": correct,
            "elapsed_s": round(_t.time() - t0, 1),
            "approx_completion_tokens": approx_tokens,
            "used_queries": queries, "doc_ids": [str(d) for d in doc_ids],
            "doc_scores": scores, "n_docs": len(doc_ids),
            "reask_answers": answers, "n_keep": n_keep,
            "keep_rate": n_keep / len(answers),
            "majority_after": maj_after, "flipped": maj_after != consensus,
            "flip_to_gold": (not correct) and maj_after == qd["answer"],
        }
    except Exception as e:  # one failed question must not kill the run
        return {"qid": qd["id"], "arm": args.arm, "error": str(e)[:200]}


def main():
    recs = [json.loads(l) for l in open(args.probe_jsonl, encoding="utf-8")]
    recs = [r for r in recs if "error" not in r]
    dataset = {d["id"]: d for d in json.load(open(args.dataset_path, encoding="utf-8"))}

    # seeded cyclic derangement for arm=randdocs (no fixed point by construction)
    qids = [r["qid"] for r in recs]
    rng = random.Random(args.seed)
    perm = qids[:]
    rng.shuffle(perm)
    shifted = perm[1:] + perm[:1]
    donor = dict(zip(perm, shifted))  # qid -> qid whose queries it borrows
    queries_by_qid = {r["qid"]: r["counter_queries"] for r in recs}

    done_qids = set()
    if Path(args.out).exists():
        for l in open(args.out, encoding="utf-8"):
            done_qids.add(json.loads(l)["qid"])
    items = []
    for r in recs:
        if r["qid"] in done_qids:
            continue
        src_qid = donor[r["qid"]] if args.arm == "randdocs" else r["qid"]
        items.append((dataset[r["qid"]], r["consensus"], r["consensus_correct"],
                      queries_by_qid[src_qid], src_qid))
    if args.limit:
        items = items[:args.limit]
    print(f"probe_control arm={args.arm}: total={len(recs)} "
          f"already_done={len(done_qids)} todo={len(items)}", flush=True)

    model = CustomLanguageModel(args.model_name, logger, base_url=args.base_url)
    retriever = RetrievalService() if args.arm != "nodocs" else None
    done = 0
    with open(args.out, "a", encoding="utf-8") as fp:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(lambda it: probe_one(it, model, retriever), items):
                fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                fp.flush()
                done += 1
                if done % 20 == 0:
                    print(f"arm={args.arm} progress {done}/{len(items)}", flush=True)
    print(f"PROBE_CONTROL_{args.arm.upper()}_DONE -> {args.out}")


if __name__ == "__main__":
    main()
