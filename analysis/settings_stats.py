"""Dataset-level statistics for the Experimental Settings numbers (all items computed in one pass)."""
import json
import sys
from collections import Counter

import numpy as np

FILES = {
    "MedQA": "runs/medqa/qwen3-8b/exp_t1_full/full_medqa.jsonl",
    "MMLU-Pro": "runs/mmlu-pro/qwen3-8b/exp_t1_full/full_mmlupro.jsonl",
    "MedBullets": "runs/medbullets/qwen3-8b/exp_t1_full/full_medbullets.jsonl",
    "MedXpertQA": "runs/medxpertqa/qwen3-8b/exp_t1_full/full_medxpertqa.jsonl",
}
ANCHORS = {"MedQA": 0.770, "MMLU-Pro": 0.70, "MedBullets": 0.58, "MedXpertQA": 0.212}

for name, path in FILES.items():
    try:
        recs = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        print(f"== {name}: FILE MISSING ==")
        continue
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    se = (acc * (1 - acc) / n) ** 0.5
    floor = max(1.5, 2 * se * 100)
    anchor = ANCHORS[name]
    delta = (acc - anchor) * 100
    ua = [r for r in recs if r["rounds"][0]["agreement_rate"] == 1.0]
    da = [r for r in recs if r["rounds"][0]["agreement_rate"] < 1.0]
    ua_err = sum(1 for r in ua if not r["correct"])
    da_err = sum(1 for r in da if not r["correct"])
    rounds = Counter(r["stopped_at_round"] for r in recs)
    nonconv = sum(1 for r in recs if r["stop_reason"] == "max_rounds")
    with_ret = sum(1 for r in recs if any(rd["retrieval"] for rd in r["rounds"]))
    ret_rounds = [sum(1 for rd in r["rounds"] if rd["retrieval"]) for r in recs]
    n_opts_dist = Counter(len(set(c["answer"] for rd in r["rounds"] for c in rd["candidates"])) for r in recs)  # rough
    print(f"== {name} ==")
    print(f"n={n} acc={acc:.4f} SE={se*100:.2f}pp anchor={anchor} delta={delta:+.2f}pp floor=max(1.5,{2*se*100:.2f})={floor:.2f}pp verdict={'within' if abs(delta)<=floor else 'exceeds->escalated'}")
    print(f"unanimous: n={len(ua)} ({len(ua)/n:.1%}) err={ua_err} ({ua_err/max(len(ua),1):.3f})")
    print(f"disagree:  n={len(da)} err={da_err} ({da_err/max(len(da),1):.3f})")
    print(f"rounds_hist={dict(sorted(rounds.items()))} nonconverged={nonconv}")
    print(f"retrieval: >=1 event {with_ret}/{n} ({with_ret/n:.1%}), mean ret-rounds/question={np.mean(ret_rounds):.2f}")
    print(f"cal/test 50-50: {n//2}/{n - n//2}")
    print()
