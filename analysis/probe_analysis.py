"""Final probe pilot analysis: two-group breakdown, keep_rate AUROC, combination with f3_5, counter-evidence availability."""
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from abstain_layer.t3_evaluate import boot_ci_auroc

recs = [json.loads(l) for l in open("results/probe_pilot_medqa.jsonl")]
ok = [r for r in recs if "error" not in r]
w = [r for r in ok if not r["consensus_correct"]]
c = [r for r in ok if r["consensus_correct"]]
print(f"n={len(ok)} (wrong={len(w)}, right={len(c)}), errors={len(recs)-len(ok)}")

for name, g in (("wrong", w), ("right", c)):
    fl = sum(r["flipped"] for r in g) / len(g)
    kr = np.mean([r["keep_rate"] for r in g])
    print(f"{name}-consensus: flip_rate={fl:.3f} mean_keep={kr:.3f}")
fg = sum(r.get("flip_to_gold", False) for r in w)
print(f"flip_to_gold among wrong: {fg}/{len(w)}")

y = np.array([int(r["consensus_correct"]) for r in ok])
keep = np.array([r["keep_rate"] for r in ok])
a = roc_auc_score(y, keep)
lo, hi = boot_ci_auroc(y, keep)
print(f"\nkeep_rate AUROC (pilot 200) = {a:.3f} CI=[{lo:.3f},{hi:.3f}]")

# counter-evidence availability (p_3): top-1 rerank score of counter-docs
top1 = np.array([max(r["counter_doc_scores"]) if r["counter_doc_scores"] else np.nan
                 for r in ok])
m = ~np.isnan(top1)
a3 = roc_auc_score(y[m], -top1[m])  # hypothesis: higher availability -> more likely wrong
print(f"counter-evidence availability (neg top1 rerank) AUROC = {a3:.3f} (n={int(m.sum())})")

# combine with f3_5 (matched by question)
soft = {r["qid"]: float(r["f3_5_mean_bidir"])
        for r in csv.DictReader(open("results/soft_f3_medqa_ua.csv"))}
f5 = np.array([soft.get(f"medqa_{r['qid']}", np.nan) for r in ok])
mm = ~np.isnan(f5)
a5 = roc_auc_score(y[mm], f5[mm])
comb = 0.5 * (keep - keep.mean()) / (keep.std() + 1e-9) + \
       0.5 * (np.where(np.isnan(f5), np.nanmedian(f5), f5) - np.nanmean(f5)) / (np.nanstd(f5) + 1e-9)
ac = roc_auc_score(y, comb)
loc, hic = boot_ci_auroc(y, comb)
print(f"f3_5 alone on pilot subset = {a5:.3f} (n={int(mm.sum())})")
print(f"keep_rate + f3_5 (equal-weight z-sum) = {ac:.3f} CI=[{loc:.3f},{hic:.3f}]")

# in-pilot risk-coverage: abstain 20/30% by combined score
order = np.argsort(comb)
wrongmask = (y == 0)
for frac in (0.2, 0.3):
    k = int(round(frac * len(y)))
    m2 = np.zeros(len(y), bool)
    m2[order[:k]] = True
    print(f"abstain {frac:.0%} by comb: intercept {int((wrongmask&m2).sum())}/{int(wrongmask.sum())}, "
          f"residual err {wrongmask[~m2].mean():.3f} (base {wrongmask.mean():.3f})")
