"""Soft-aggregation evaluation readouts (full feature-set reporting)."""
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from abstain_layer.t3_evaluate import boot_ci_auroc, load

rows = list(csv.DictReader(open("results/soft_f3_medqa_ua.csv")))
y = np.array([int(r["correct"]) for r in rows])
f5 = np.array([float(r["f3_5_mean_bidir"]) for r in rows])
f6 = np.array([float(r["f3_6_mean_contradiction"]) for r in rows])
f7 = np.array([float(r["f3_7_max_bidir"]) for r in rows])
qids = [r["qid"] for r in rows]
n = len(y)
wrong = (y == 0)
print(f"n={n} wrong={int(wrong.sum())}")

# align hard entropy (same qid order)
Xq, yq, names, raw = load("results/features_medqa.csv")
qid2hard = {r["qid"]: r for r in raw}
hard_ent = np.array([float(qid2hard[q]["f3_2_bidir_entropy"]) for q in qids])

for nm, score in (("f3_5_mean_bidir", f5), ("f3_6_mean_contradiction", -f6),
                  ("f3_7_max_bidir", f7)):
    a = roc_auc_score(y, score)
    lo, hi = boot_ci_auroc(y, score)
    print(f"{nm}: AUROC={a:.4f} CI=[{lo:.4f},{hi:.4f}]")

# saturation rate (discreteness)
for nm, v in (("f3_5", f5), ("f3_6", f6)):
    top = (np.abs(v - v.max()) < 1e-9).mean()
    print(f"{nm} max-tie fraction: {top:.2%}  distinct values: {len(np.unique(np.round(v,6)))}")

# deciles (low f3_5 = suspect -> sort by -f3_5; same convention as hard entropy: D1 = most trusted)
order = np.argsort(-f5)
bins = np.array_split(order, 10)
print("f3_5 decile errs (D1=highest mean_bidir):", " ".join(f"{1-y[b].mean():.2f}" for b in bins))

# risk-coverage: abstain on the lowest f3_5
for frac in (0.10, 0.20, 0.30, 0.50):
    k = int(round(frac * n))
    m = np.zeros(n, bool)
    m[np.argsort(f5)[:k]] = True
    print(f"abstain {frac:.0%} by f3_5: intercept {int((wrong&m).sum())}/{int(wrong.sum())}, "
          f"residual err {wrong[~m].mean():.4f}")

# plateau-internal check: on questions with hard entropy = max, error rate by lower/upper f3_5 half
plateau = np.abs(hard_ent - np.nanmax(hard_ent)) < 1e-9
np_pl = int(plateau.sum())
f5p, yp = f5[plateau], y[plateau]
half = np_pl // 2
lo_half = np.argsort(f5p)[:half]      # low soft score (suspect)
hi_half = np.argsort(f5p)[half:]
print(f"plateau n={np_pl}: err(low f3_5 half)={1-yp[lo_half].mean():.3f} "
      f"vs err(high f3_5 half)={1-yp[hi_half].mean():.3f}")

# hard+soft combination (5-fold CV logreg)
Z = np.column_stack([hard_ent, f5, f6, f7])
oof = np.full(n, np.nan)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in skf.split(Z, y):
    mu, sd = Z[tr].mean(0), Z[tr].std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(max_iter=2000).fit((Z[tr]-mu)/sd, y[tr])
    oof[te] = clf.predict_proba((Z[te]-mu)/sd)[:, 1]
a = roc_auc_score(y, oof)
lo, hi = boot_ci_auroc(y, oof)
print(f"hard+soft combined CV AUROC={a:.4f} CI=[{lo:.4f},{hi:.4f}] (hard single-feature reference 0.578)")
