"""(1) Feature attribution for the combined model: full-data standardized logreg coefficients + family-level ablation (CV AUROC drop when a family is removed).
(2) Preliminary family-3 risk-coverage on the unanimous layer: how many unanimous errors are intercepted when abstaining on 20% by rationale entropy."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from abstain_layer.t3_evaluate import PROCESS_PREFIXES, cv_scores, load

X, y, names, _ = load("results/features_medqa.csv")
proc_idx = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]
proc_names = [names[j] for j in proc_idx]

# ---- (1a) standardized logreg coefficients (full-data fit, median imputation + missing indicators, direction only) ----
Xs = X[:, proc_idx]
med = np.nanmedian(Xs, axis=0)
med = np.where(np.isnan(med), 0.0, med)
miss = np.isnan(Xs).astype(float)
filled = np.where(np.isnan(Xs), med, Xs)
mu, sd = filled.mean(0), filled.std(0)
sd[sd == 0] = 1.0
Z = np.hstack([(filled - mu) / sd, miss])
clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z, y)
coefs = clf.coef_[0][:len(proc_names)]
order = np.argsort(-np.abs(coefs))
print("== top-10 |standardized coef| (full-data fit, direction only) ==")
for k in order[:10]:
    print(f"  {proc_names[k]:38s} {coefs[k]:+.3f}")

# ---- (1b) family-level ablation: CV AUROC drop from drop-one-family (more reliable attribution) ----
full_auc = roc_auc_score(y, cv_scores(X, y, proc_idx, "logreg"))
print(f"\n== family drop-one ablation (CV AUROC, full={full_auc:.4f}) ==")
fams = {"f0(consensus)": "f0_", "f1(distribution)": "f1_", "f2(minority)": "f2_",
        "f3(rationale)": "f3_", "f4(retrieval)": "f4_"}
for label, pre in fams.items():
    keep = [j for j in proc_idx if not names[j].startswith(pre)]
    auc = roc_auc_score(y, cv_scores(X, y, keep, "logreg"))
    only = [j for j in proc_idx if names[j].startswith(pre)]
    auc_only = roc_auc_score(y, cv_scores(X, y, only, "logreg")) if only else float("nan")
    print(f"  drop {label:12s}: {auc:.4f} (Δ={auc-full_auc:+.4f})   alone: {auc_only:.4f}")

# ---- (2) family-3 risk-coverage on the unanimous layer @ 20% abstention ----
ua = X[:, names.index("f0_1_r1_agreement")] == 1.0
yu = y[ua]
ent = X[ua, names.index("f3_2_bidir_entropy")]
ent = np.where(np.isnan(ent), np.nanmedian(ent), ent)
n = int(ua.sum())
n_abs = int(round(0.20 * n))
abstain_idx = np.argsort(-ent)[:n_abs]          # abstain on the top-20% by entropy
mask_abs = np.zeros(n, bool)
mask_abs[abstain_idx] = True
wrong = (yu == 0)
print(f"\n== family-3 risk-coverage on unanimous layer (n={n}, wrong={int(wrong.sum())}) ==")
print(f"  abstain 20% (n={n_abs}) by f3_2_bidir_entropy:")
print(f"    intercepted wrong: {int((wrong & mask_abs).sum())}/{int(wrong.sum())}")
print(f"    residual err on answered: {wrong[~mask_abs].mean():.4f} (base {wrong.mean():.4f})")
for frac in (0.10, 0.30, 0.50):
    k = int(round(frac * n))
    m = np.zeros(n, bool)
    m[np.argsort(-ent)[:k]] = True
    print(f"  abstain {frac:.0%}: intercept {int((wrong & m).sum())}/{int(wrong.sum())}, "
          f"residual err {wrong[~m].mean():.4f}")
# reference: expected interception of random 20% abstention = 20% of the error count
print(f"  [random-abstain 20% expectation: {0.2*wrong.sum():.1f} wrong intercepted]")
