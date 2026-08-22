"""Soft f3 on the unanimous layer across three datasets: AUROC, risk-coverage@20/30%, vs the hard version."""
import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from abstain_layer.t3_evaluate import boot_ci_auroc

for ds, hard_auc in (("medqa", 0.578), ("mmlupro", 0.593), ("bullets", 0.364)):
    rows = list(csv.DictReader(open(f"results/soft_f3_{ds}_ua.csv")))
    y = np.array([int(r["correct"]) for r in rows])
    f5 = np.array([float(r["f3_5_mean_bidir"]) for r in rows])
    n = len(y)
    wrong = (y == 0)
    if len(set(y)) < 2:
        print(f"{ds}: degenerate labels")
        continue
    a = roc_auc_score(y, f5)
    lo, hi = boot_ci_auroc(y, f5)
    line = f"{ds:8s} n={n:4d} err={wrong.mean():.3f}  f3_5 AUROC={a:.3f} CI=[{lo:.3f},{hi:.3f}] (hard was {hard_auc})"
    for frac in (0.20, 0.30):
        k = int(round(frac * n))
        m = np.zeros(n, bool)
        m[np.argsort(f5)[:k]] = True
        line += f" | abst{frac:.0%}: {int((wrong & m).sum())}/{int(wrong.sum())}"
    print(line)
