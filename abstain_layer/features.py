"""Feature extraction: pure-function implementation of the frozen feature definitions
(families 0/1/2/4 + baseline scores), fixed before evaluation data were inspected.

Input = one unified-schema record (dict, see abstain_layer.schema); output = {feature name: float|None}.
None means the feature is undefined for this question (e.g. no retrieval event, no verifier
score); downstream evaluation handles missingness.
Degenerate-case conventions (recorded here, where the frozen spec was silent):
  - slope features are 0.0 for sequences shorter than 2 (no change).
  - mono_frac is 1.0 for length < 2 (trivially monotone).
  - osc_index is 0 for length < 3.
"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


# ---------- basic helpers ----------

def _dist(round_: Dict[str, Any]) -> Dict[str, float]:
    answers = [c["answer"] for c in round_["candidates"]]
    n = len(answers)
    return {a: cnt / n for a, cnt in Counter(answers).items()}


def _majority(round_: Dict[str, Any]) -> str:
    return round_["majority_answer"]


def _agreement_seq(rec: Dict[str, Any]) -> List[float]:
    return [r["agreement_rate"] for r in rec["rounds"]]


def _slope(ys: Sequence[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = range(1, n + 1)
    mx = (n + 1) / 2
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Base-2 JS divergence, range [0, 1]."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}

    def kl(a: Dict[str, float]) -> float:
        return sum(a.get(k, 0.0) * math.log2(a.get(k, 0.0) / m[k])
                   for k in keys if a.get(k, 0.0) > 0)

    return (kl(p) + kl(q)) / 2


def _retrieval_rounds(rec: Dict[str, Any]) -> List[int]:
    """Indices (0-based, into the rounds list) of rounds where retrieval happened."""
    return [i for i, r in enumerate(rec["rounds"]) if r["retrieval"]]


# ---------- family 0: existing consensus-process features ----------

def family0(rec: Dict[str, Any], t_max: int = 8) -> Dict[str, Optional[float]]:
    rounds = rec["rounds"]
    agree = _agreement_seq(rec)
    majorities = [_majority(r) for r in rounds]

    f: Dict[str, Optional[float]] = {}
    f["f0_1_r1_agreement"] = agree[0]
    f["f0_2_convergence_round"] = float(
        rec["stopped_at_round"] if rec["stop_reason"] == "converged" else t_max + 1)
    f["f0_3_majority_flips"] = float(
        sum(1 for a, b in zip(majorities, majorities[1:]) if a != b))
    f["f0_4_agreement_slope"] = _slope(agree)
    f["f0_5_agreement_auc"] = sum(agree) / len(agree)

    deltas = [agree[i + 1] - agree[i] for i in _retrieval_rounds(rec)
              if i + 1 < len(agree)]
    f["f0_6_post_retrieval_delta_mean"] = sum(deltas) / len(deltas) if deltas else None
    f["f0_6_post_retrieval_delta_last"] = deltas[-1] if deltas else None

    margins = []
    for r in rounds:
        scores = sorted((c["verifier_score"] for c in r["candidates"]
                         if c["verifier_score"] is not None), reverse=True)
        margins.append(scores[0] - scores[1] if len(scores) >= 2 else None)
    if any(m is None for m in margins):
        f["f0_7_verifier_margin_last"] = None
        f["f0_7_verifier_margin_slope"] = None
    else:
        f["f0_7_verifier_margin_last"] = margins[-1]
        f["f0_7_verifier_margin_slope"] = _slope(margins)

    diffs = [b - a for a, b in zip(agree, agree[1:]) if b != a]
    f["f0_8_osc_index"] = float(
        sum(1 for a, b in zip(diffs, diffs[1:]) if (a > 0) != (b > 0)))

    rr = _retrieval_rounds(rec)
    if rr and rr[0] + 1 < len(rounds):
        f["f0_9_cb_flip"] = float(majorities[0] != majorities[rr[0] + 1])
    else:
        f["f0_9_cb_flip"] = None
    f["f0_10_cb_final_match"] = float(majorities[0] == rec["final_answer"])
    return f


# ---------- family 1: answer-distribution trajectory (ablation material only) ----------

def family1(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    dists = [_dist(r) for r in rec["rounds"]]
    ents = [_entropy(d) for d in dists]
    f: Dict[str, Optional[float]] = {}
    f["f1_1_entropy_first"] = ents[0]
    f["f1_1_entropy_last"] = ents[-1]
    f["f1_1_entropy_slope"] = _slope(ents)
    if len(ents) >= 2:
        f["f1_1_entropy_mono_frac"] = (
            sum(1 for a, b in zip(ents, ents[1:]) if b <= a) / (len(ents) - 1))
    else:
        f["f1_1_entropy_mono_frac"] = 1.0
    js = [_js_divergence(a, b) for a, b in zip(dists, dists[1:])]
    f["f1_2_js_adjacent_mean"] = sum(js) / len(js) if js else 0.0
    f["f1_2_js_adjacent_max"] = max(js) if js else 0.0
    f["f1_2_js_adjacent_last"] = js[-1] if js else 0.0
    final_delta = {rec["final_answer"]: 1.0}
    f["f1_3_js_to_final_slope"] = _slope(
        [_js_divergence(d, final_delta) for d in dists])
    return f


# ---------- family 2: minority persistence (defined at the answer-option level) ----------

def family2(rec: Dict[str, Any]) -> Dict[str, float]:
    rounds = rec["rounds"]
    t_stop = len(rounds)
    final = rec["final_answer"]
    n = len(rounds[0]["candidates"])
    present: Dict[str, List[bool]] = {}
    for r in rounds:
        d = _dist(r)
        for a, share in d.items():
            if a != final and share >= 1.0 / n - 1e-9:
                present.setdefault(a, [False] * t_stop)
    for a in present:
        for i, r in enumerate(rounds):
            present[a][i] = _dist(r).get(a, 0.0) >= 1.0 / n - 1e-9

    f: Dict[str, float] = {}
    if not present:
        f["f2_1_minority_persistence"] = 0.0
        f["f2_2_minority_max_run"] = 0.0
        f["f2_3_minority_turnover"] = 0.0
        f["f2_4_runnerup_share_slope"] = 0.0
        return f

    f["f2_1_minority_persistence"] = max(
        sum(v) / t_stop for v in present.values())

    def max_run(flags: List[bool]) -> int:
        best = cur = 0
        for x in flags:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best

    f["f2_2_minority_max_run"] = max(
        max_run(v) / t_stop for v in present.values())
    f["f2_3_minority_turnover"] = len(present) / t_stop
    runnerup = max(present, key=lambda a: sum(_dist(r).get(a, 0.0) for r in rounds))
    f["f2_4_runnerup_share_slope"] = _slope(
        [_dist(r).get(runnerup, 0.0) for r in rounds])
    return f


# ---------- family 4: retrieval-side behavior (trajectory sub-features available for free) ----------

def _token_set(q: str) -> set:
    return set(q.lower().split())


def family4(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    events = [r["retrieval"] for r in rec["rounds"] if r["retrieval"]]
    f: Dict[str, Optional[float]] = {}
    queries = [q for e in events for q in e["queries"]]
    if len(queries) >= 2:
        dists = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                a, b = _token_set(queries[i]), _token_set(queries[j])
                union = a | b
                dists.append(1 - len(a & b) / len(union) if union else 0.0)
        f["f4_1_query_diversity"] = sum(dists) / len(dists)
    else:
        f["f4_1_query_diversity"] = None

    if events:
        seen: set = set()
        repeats = []
        new_counts = []
        for e in events:
            docs = set(e["doc_ids"])
            repeats.append(len(docs & seen) / len(docs) if docs else 0.0)
            new_counts.append(len(docs - seen))
            seen |= docs
        f["f4_2_doc_repeat_mean"] = sum(repeats) / len(repeats)
        f["f4_2_doc_repeat_last"] = repeats[-1]
        f["f4_3_new_doc_saturation"] = (
            new_counts[-1] / len(seen) if seen else 0.0)
    else:
        f["f4_2_doc_repeat_mean"] = None
        f["f4_2_doc_repeat_last"] = None
        f["f4_3_new_doc_saturation"] = None
    return f


# ---------- baseline scores (implemented here, used as evaluation references) ----------

def baselines(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    last = rec["rounds"][-1]
    f: Dict[str, Optional[float]] = {}
    f["bl_terminal_agreement"] = last["agreement_rate"]  # snapshot agreement baseline; the primary reference
    lps = [c["seq_logprob"] for c in last["candidates"] if c["seq_logprob"] is not None]
    f["bl_seq_logprob_mean"] = sum(lps) / len(lps) if lps else None
    return f


def extract_features(rec: Dict[str, Any], t_max: int = 8) -> Dict[str, Optional[float]]:
    """All features + baselines, with stable key order. Family 3 needs an embedder; see rationale_features.py."""
    out: Dict[str, Optional[float]] = {}
    out.update(family0(rec, t_max=t_max))
    out.update(family1(rec))
    out.update(family2(rec))
    out.update(family4(rec))
    out.update(baselines(rec))
    return out
