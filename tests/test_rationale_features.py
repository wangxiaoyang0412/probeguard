import gzip
import json
import math
from pathlib import Path

from abstain_layer.rationale_features import family3

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_medqa50.jsonl.gz"


def fake_embed(texts):
    """Deterministic fake embedder: maps the first semantic keyword to an orthogonal basis vector; same-source rationales get the same vector."""
    base = {"mechanism": [1.0, 0.0, 0.0], "dosage": [0.0, 1.0, 0.0],
            "anatomy": [0.0, 0.0, 1.0]}
    out = []
    for t in texts:
        vec = [0.1, 0.1, 0.1]
        for key, v in base.items():
            if key in t:
                vec = v
                break
        out.append(vec)
    return out


def _round(answers, rationales):
    from collections import Counter
    counts = Counter(answers)
    maj, majn = counts.most_common(1)[0]
    return {"t": 1, "candidates": [
        {"answer": a, "rationale": r, "text_hash": "h", "seq_logprob": -1.0,
         "verifier_score": None} for a, r in zip(answers, rationales)],
        "agreement_rate": majn / len(answers), "n_distinct_answers": len(counts),
        "majority_answer": maj, "retrieval": None}


def _rec(rounds):
    final = rounds[-1]["majority_answer"]
    return {"qid": "t", "dataset": "T", "gold": final, "final_answer": final,
            "correct": True, "stopped_at_round": len(rounds),
            "stop_reason": "converged", "rounds": rounds, "probe": None}


def test_true_consensus_single_cluster():
    rec = _rec([_round(["C"] * 4, ["mechanism x"] * 4)])
    f = family3(rec, fake_embed)
    assert f["f3_1_rationale_cluster_count"] == 1.0
    assert f["f3_2_rationale_entropy"] == 0.0


def test_false_consensus_split_rationales():
    rec = _rec([_round(["C"] * 4,
                       ["mechanism x", "mechanism x", "dosage y", "anatomy z"])])
    f = family3(rec, fake_embed)
    assert f["f3_1_rationale_cluster_count"] == 3.0
    assert f["f3_2_rationale_entropy"] > 0.9        # entropy of a 2/1/1 split


def test_fixture_runs_with_fake_embedder():
    recs = [json.loads(line) for line in gzip.open(FIXTURE, "rt", encoding="utf-8")]
    for rec in recs[:10]:
        f = family3(rec, fake_embed)
        for k, v in f.items():
            assert v is None or math.isfinite(v), (k, v)
