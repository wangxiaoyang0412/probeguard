import json
import sys
from collections import Counter

path = sys.argv[1]
anchor = float(sys.argv[2])
recs = [json.loads(l) for l in open(path)]
n = len(recs)
acc = sum(r["correct"] for r in recs) / n
ua = [r for r in recs if r["rounds"][0]["agreement_rate"] == 1.0]
da = [r for r in recs if r["rounds"][0]["agreement_rate"] < 1.0]
verdict = "PASS" if abs(acc - anchor) <= 0.015 else "CHECK"
acc_ua = sum(r["correct"] for r in ua) / max(len(ua), 1)
acc_da = sum(r["correct"] for r in da) / max(len(da), 1)
print(f"n={n} acc={acc:.4f} anchor={anchor} delta={acc-anchor:+.4f} -> {verdict}")
print(f"r1_unanimous={len(ua)}/{n} ({len(ua)/n:.0%})  acc|ua={acc_ua:.3f}  acc|da={acc_da:.3f}")
print("stop:", dict(Counter(r["stop_reason"] for r in recs)))
print("rounds:", dict(sorted(Counter(r["stopped_at_round"] for r in recs).items())))
print("with_retrieval:", sum(1 for r in recs if any(rd["retrieval"] for rd in r["rounds"])))
