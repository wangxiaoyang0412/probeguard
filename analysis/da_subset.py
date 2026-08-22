"""On the disagreement subset (r1<1.0): AUROC of combined process features vs r1, with delta CI."""
import numpy as np
from sklearn.metrics import roc_auc_score

from abstain_layer.t3_evaluate import PROCESS_PREFIXES, boot_ci_diff, cv_scores, load

X, y, names, _ = load("results/features_medqa.csv")
da = X[:, names.index("f0_1_r1_agreement")] < 1.0
Xd, yd = X[da], y[da]
proc_idx = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]
r1 = Xd[:, names.index("f0_1_r1_agreement")]

oof = cv_scores(Xd, yd, proc_idx, "logreg")
auc_c = roc_auc_score(yd, oof)
auc_r1 = roc_auc_score(yd, r1)
lo, hi = boot_ci_diff(yd, oof, r1)
print(f"disagreement subset n={int(da.sum())} acc={yd.mean():.4f}")
print(f"combined={auc_c:.4f}  r1={auc_r1:.4f}  delta={auc_c-auc_r1:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]")
