"""Full probe analysis on the unanimous layer + Mondrian S1 certification push (keep_rate + f3_5)."""
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from abstain_layer.phase1_conformal import evaluate, ltt_threshold
from abstain_layer.t3_evaluate import boot_ci_auroc

probe = {json.loads(l)["qid"]: json.loads(l)
         for l in open("results/probe_pilot_medqa.jsonl") if "error" not in json.loads(l)}
soft = {r["qid"]: float(r["f3_5_mean_bidir"])
        for r in csv.DictReader(open("results/soft_f3_medqa_ua.csv"))}

qids, y, keep, f5, cost_s, cost_tok = [], [], [], [], [], []
for q, r in probe.items():
    sq = f"medqa_{q}"
    if sq not in soft:
        continue
    qids.append(q)
    y.append(int(r["consensus_correct"]))
    keep.append(r["keep_rate"])
    f5.append(soft[sq])
    if "elapsed_s" in r:
        cost_s.append(r["elapsed_s"])
        cost_tok.append(r["approx_completion_tokens"])
y = np.array(y); keep = np.array(keep); f5 = np.array(f5)
n = len(y); wrong = (y == 0)
print(f"n={n} err={wrong.mean():.3f} (natural class proportions)")
print(f"probe cost (n={len(cost_s)}): median {np.median(cost_s):.0f}s / {np.median(cost_tok):.0f} tok, "
      f"p90 {np.percentile(cost_s,90):.0f}s")

def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)

for nm, s in (("keep_rate", keep), ("f3_5", f5), ("keep+f3_5", z(keep) + z(f5))):
    a = roc_auc_score(y, s)
    lo, hi = boot_ci_auroc(y, s)
    print(f"{nm:10s} AUROC={a:.3f} CI=[{lo:.3f},{hi:.3f}]")

# two-group breakdown (natural class proportions)
for name, g in (("wrong", wrong), ("right", ~wrong)):
    print(f"{name}: n={int(g.sum())} flip_rate={np.mean([probe[q]['flipped'] for i,q in enumerate(qids) if g[i]]):.3f} "
          f"mean_keep={keep[g].mean():.3f}")

# Mondrian S1 certification push
comb = z(keep) + z(f5)
idx = np.arange(n)
cal, te = train_test_split(idx, test_size=0.5, random_state=42, stratify=y)
print("\n== S1 certification with keep+f3_5 (LTT-Bonferroni, delta=0.025) ==")
for alpha in (0.05, 0.10, 0.15):
    for nm, s in (("keep+f3_5", comb), ("f3_5-only", f5)):
        thr, cov_cal = ltt_threshold(s[cal], y[cal], alpha, delta=0.025)
        cov, risk = evaluate(s[te], y[te], thr)
        print(f"  α={alpha:.2f} {nm:10s}: cal-cov={cov_cal:.2f} test-cov={cov:.3f} test-risk={risk:.4f}")
