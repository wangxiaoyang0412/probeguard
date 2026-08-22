"""Family 3: answer-conditioned rationale features (the core claim features).

Design frozen before evaluation data were inspected:
  Main implementation = NLI bidirectional-entailment clustering (semantic-entropy style)
      over extracted key claim sentences (family3_nli; primary criterion = bidirectional,
      backup = non_contradiction, both pre-specified).
  Ablation implementation = greedy cosine clustering of embeddings over the full text
      (family3; inject a long-window encoder such as BGE-M3).
Both paths share one injection-style interface for the ablation matrix; unit tests use deterministic fakes.
"""
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

Embedder = Callable[[List[str]], List[List[float]]]
# Takes a list of sentence pairs, returns each pair's NLI label: 'entailment' | 'neutral' | 'contradiction'
NLIJudge = Callable[[List[Tuple[str, str]]], List[str]]

_CUE = re.compile(
    r"\b(answer|option|choice|correct|therefore|thus|hence|most likely|best)\b",
    re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_claims(rationale: str) -> str:
    """Key-claim extraction rule, frozen before evaluation data were inspected (part of the feature definition; no post-hoc tuning).

    Strip <answer> tags; from the last 5 sentences keep those containing conclusion
    cue words (at most 3, order preserved); fall back to the last sentence if none match.
    """
    text = _ANSWER_TAG.sub("", rationale).strip()
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return ""
    tail = sentences[-5:]
    hits = [s for s in tail if _CUE.search(s)][-3:]
    return " ".join(hits) if hits else sentences[-1]


def _nli_clusters(claims: List[str], judge: NLIJudge,
                  criterion: str) -> List[List[int]]:
    """Semantic-entropy-style greedy clustering: new members compare only against each cluster's first member (as in Kuhn et al.)."""
    clusters: List[List[int]] = []
    for i, claim in enumerate(claims):
        placed = False
        for cluster in clusters:
            rep = claims[cluster[0]]
            fwd, bwd = judge([(claim, rep), (rep, claim)])
            if criterion == "bidirectional":
                same = fwd == "entailment" and bwd == "entailment"
            elif criterion == "non_contradiction":
                same = fwd != "contradiction" and bwd != "contradiction"
            else:
                raise ValueError(f"unknown criterion: {criterion}")
            if same:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def _cluster_features(n_items: int, clusters: List[List[int]],
                      prefix: str) -> Dict[str, float]:
    sizes = [len(c) / n_items for c in clusters]
    return {
        f"{prefix}_cluster_count": float(len(clusters)),
        f"{prefix}_entropy": -sum(p * math.log(p) for p in sizes if p > 0),
    }


def family3_nli(rec: Dict[str, Any], judge: NLIJudge) -> Dict[str, Optional[float]]:
    """Main implementation: rationale semantic entropy (primary + backup criteria computed in one pass)."""
    f: Dict[str, Optional[float]] = {}
    last_rats = _majority_rationales(rec["rounds"][-1])
    claims = [extract_claims(r) for r in last_rats]
    if len(claims) < 2:
        for crit in ("bidir", "noncon"):
            f[f"f3_1_{crit}_cluster_count"] = 1.0 if claims else None
            f[f"f3_2_{crit}_entropy"] = 0.0 if claims else None
    else:
        for crit, name in (("bidirectional", "bidir"), ("non_contradiction", "noncon")):
            clusters = _nli_clusters(claims, judge, crit)
            feats = _cluster_features(len(claims), clusters, "x")
            f[f"f3_1_{name}_cluster_count"] = feats["x_cluster_count"]
            f[f"f3_2_{name}_entropy"] = feats["x_entropy"]

    # f3_3: per-round fraction of same-cluster claim pairs within the consensus (primary criterion), then its slope
    coherences = []
    for r in rec["rounds"]:
        rats = _majority_rationales(r)
        cl = [extract_claims(x) for x in rats]
        if len(cl) < 2:
            continue
        pairs = [(cl[i], cl[j]) for i in range(len(cl)) for j in range(i + 1, len(cl))]
        flat: List[Tuple[str, str]] = []
        for a, b in pairs:
            flat += [(a, b), (b, a)]
        labels = judge(flat)
        same = [labels[2 * k] == "entailment" and labels[2 * k + 1] == "entailment"
                for k in range(len(pairs))]
        coherences.append(sum(same) / len(same))
    if len(coherences) >= 2:
        n = len(coherences)
        mx = (n + 1) / 2
        my = sum(coherences) / n
        denom = sum((x - mx) ** 2 for x in range(1, n + 1))
        f["f3_3_coherence_slope"] = sum(
            (x - mx) * (y - my) for x, y in zip(range(1, n + 1), coherences)) / denom
    elif len(coherences) == 1:
        f["f3_3_coherence_slope"] = 0.0
    else:
        f["f3_3_coherence_slope"] = None
    return f


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _greedy_clusters(vecs: List[List[float]], tau: float) -> List[List[int]]:
    clusters: List[List[int]] = []
    centroids: List[List[float]] = []
    for i, v in enumerate(vecs):
        for ci, cent in enumerate(centroids):
            if _cos(v, cent) >= tau:
                clusters[ci].append(i)
                members = clusters[ci]
                centroids[ci] = [
                    sum(vecs[m][d] for m in members) / len(members)
                    for d in range(len(v))]
                break
        else:
            clusters.append([i])
            centroids.append(list(v))
    return clusters


def _majority_rationales(round_: Dict[str, Any]) -> List[str]:
    m = round_["majority_answer"]
    return [c["rationale"] for c in round_["candidates"] if c["answer"] == m]


def family3(rec: Dict[str, Any], embed: Embedder,
            tau: float = 0.80) -> Dict[str, Optional[float]]:
    f: Dict[str, Optional[float]] = {}
    last_rats = _majority_rationales(rec["rounds"][-1])
    if len(last_rats) < 2:
        f["f3_1_rationale_cluster_count"] = 1.0 if last_rats else None
        f["f3_2_rationale_entropy"] = 0.0 if last_rats else None
    else:
        vecs = embed(last_rats)
        clusters = _greedy_clusters(vecs, tau)
        f["f3_1_rationale_cluster_count"] = float(len(clusters))
        n = len(last_rats)
        sizes = [len(c) / n for c in clusters]
        f["f3_2_rationale_entropy"] = -sum(p * math.log(p) for p in sizes if p > 0)

    coherences = []
    for r in rec["rounds"]:
        rats = _majority_rationales(r)
        if len(rats) < 2:
            continue
        vecs = embed(rats)
        sims = [_cos(vecs[i], vecs[j])
                for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        coherences.append(sum(sims) / len(sims))
    if len(coherences) >= 2:
        n = len(coherences)
        mx = (n + 1) / 2
        my = sum(coherences) / n
        denom = sum((x - mx) ** 2 for x in range(1, n + 1))
        f["f3_3_rationale_coherence_slope"] = sum(
            (x - mx) * (y - my) for x, y in zip(range(1, n + 1), coherences)) / denom
    elif len(coherences) == 1:
        f["f3_3_rationale_coherence_slope"] = 0.0
    else:
        f["f3_3_rationale_coherence_slope"] = None
    return f
