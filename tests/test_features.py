import gzip
import json
import math
from pathlib import Path

from abstain_layer.features import extract_features, family0, family1, family2, family4

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_medqa50.jsonl.gz"


def _round(t, answers, retrieval=None, verifier=None):
    from collections import Counter
    counts = Counter(answers)
    maj, majn = counts.most_common(1)[0]
    cands = []
    for i, a in enumerate(answers):
        cands.append({"answer": a, "rationale": f"reason {a} {i}",
                      "text_hash": f"h{t}{i}", "seq_logprob": -2.0,
                      "verifier_score": verifier[i] if verifier else None})
    return {"t": t, "candidates": cands, "agreement_rate": majn / len(answers),
            "n_distinct_answers": len(counts), "majority_answer": maj,
            "retrieval": retrieval}


def _rec(rounds, gold="C", stop_reason="converged"):
    final = rounds[-1]["majority_answer"]
    return {"qid": "t_1", "dataset": "T", "gold": gold, "final_answer": final,
            "correct": final == gold, "stopped_at_round": rounds[-1]["t"],
            "stop_reason": stop_reason, "rounds": rounds, "probe": None}


def test_family0_flip_and_retrieval_delta():
    rec = _rec([
        _round(1, ["A", "A", "B", "B"], retrieval={"queries": ["q"], "doc_ids": ["d1"]}),
        _round(2, ["C", "C", "C", "B"], retrieval={"queries": ["q2"], "doc_ids": ["d1", "d2"]}),
        _round(3, ["C", "C", "C", "C"]),
    ])
    f = family0(rec)
    assert f["f0_1_r1_agreement"] == 0.5
    assert f["f0_3_majority_flips"] == 1.0          # A->C (flip at round 2), C->C
    assert abs(f["f0_6_post_retrieval_delta_mean"] - ((0.75 - 0.5) + (1.0 - 0.75)) / 2) < 1e-9
    assert f["f0_9_cb_flip"] == 1.0                 # round-1 mode A != post-retrieval mode C
    assert f["f0_10_cb_final_match"] == 0.0
    assert f["f0_7_verifier_margin_last"] is None    # intrinsic variant has no verifier scores


def test_family0_unconverged_records_tmax_plus_one():
    rec = _rec([_round(t, ["A", "B", "A", "B"]) for t in range(1, 9)],
               stop_reason="max_rounds")
    assert family0(rec, t_max=8)["f0_2_convergence_round"] == 9.0


def test_family1_entropy_monotone():
    rec = _rec([
        _round(1, ["A", "B", "C", "D"]),
        _round(2, ["A", "A", "B", "B"]),
        _round(3, ["A", "A", "A", "A"]),
    ])
    f = family1(rec)
    assert f["f1_1_entropy_first"] > f["f1_1_entropy_last"] == 0.0
    assert f["f1_1_entropy_mono_frac"] == 1.0
    assert 0.0 < f["f1_2_js_adjacent_mean"] <= 1.0


def test_family2_persistent_vs_noisy_minority():
    # B is a persistent minority (present every round); D/E are noise minorities (one round each)
    persistent = _rec([
        _round(1, ["C", "C", "B", "D"]),
        _round(2, ["C", "C", "B", "E"]),
        _round(3, ["C", "C", "C", "B"]),
    ])
    fp = family2(persistent)
    assert fp["f2_1_minority_persistence"] == 1.0    # B present 3/3 rounds
    assert fp["f2_2_minority_max_run"] == 1.0
    assert fp["f2_3_minority_turnover"] == 3 / 3     # three dissenting options: B/D/E
    noisy = _rec([
        _round(1, ["C", "C", "C", "D"]),
        _round(2, ["C", "C", "C", "E"]),
        _round(3, ["C", "C", "C", "A"]),
    ])
    fn = family2(noisy)
    assert fn["f2_2_minority_max_run"] == 1 / 3      # no option persists across rounds
    assert fn["f2_1_minority_persistence"] == 1 / 3


def test_family2_unanimous_all_zero():
    rec = _rec([_round(1, ["C", "C", "C", "C"])])
    assert all(v == 0.0 for v in family2(rec).values())


def test_family4_doc_repeat_and_saturation():
    rec = _rec([
        _round(1, ["A", "B", "A", "B"],
               retrieval={"queries": ["aspirin mechanism", "heparin dosing"],
                          "doc_ids": ["d1", "d2"]}),
        _round(2, ["A", "A", "A", "B"],
               retrieval={"queries": ["aspirin mechanism of action"],
                          "doc_ids": ["d1", "d2", "d3"]}),
        _round(3, ["A", "A", "A", "A"]),
    ])
    f = family4(rec)
    assert f["f4_2_doc_repeat_last"] == 2 / 3        # d1,d2 repeated
    assert f["f4_3_new_doc_saturation"] == 1 / 3     # only d3 is new in the last event
    assert 0.0 < f["f4_1_query_diversity"] < 1.0


def test_fixture_all_records_extract_cleanly():
    recs = [json.loads(line) for line in gzip.open(FIXTURE, "rt", encoding="utf-8")]
    assert len(recs) == 50
    keys = None
    for rec in recs:
        f = extract_features(rec)
        if keys is None:
            keys = set(f)
        assert set(f) == keys                        # stable key set
        for k, v in f.items():
            assert v is None or (isinstance(v, float) and math.isfinite(v)), (k, v)
        assert f["bl_terminal_agreement"] == rec["rounds"][-1]["agreement_rate"]
    # regression anchor from the smoke run: 30/50 round-1 unanimous
    assert sum(1 for r in recs if r["rounds"][0]["agreement_rate"] == 1.0) == 30
