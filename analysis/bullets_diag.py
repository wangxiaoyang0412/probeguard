"""Three Bullets diagnostics: ① frozen MedQA->Bullets transfer; ② single-feature AUROC table; ③ top-5 reduced CV."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from abstain_layer.t3_evaluate import PROCESS_PREFIXES, boot_ci_auroc, cv_scores, load

Xq, yq, names, _ = load("results/features_medqa.csv")
Xb, yb, names_b, _ = load("results/features_medbullets.csv")
assert names == names_b
proc_idx = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]

# ---- ① frozen transfer: all statistics and weights from MedQA, zero Bullets information ----
Xq_p, Xb_p = Xq[:, proc_idx], Xb[:, proc_idx]
med = np.nanmedian(Xq_p, axis=0)
med = np.where(np.isnan(med), 0.0, med)

def prep(a):
    miss = np.isnan(a).astype(float)
    return np.hstack([np.where(np.isnan(a), med, a), miss])

Zq, Zb = prep(Xq_p), prep(Xb_p)
mu, sd = Zq.mean(0), Zq.std(0)
sd[sd == 0] = 1.0
clf = LogisticRegression(max_iter=2000, C=1.0).fit((Zq - mu) / sd, yq)
s_transfer = clf.predict_proba((Zb - mu) / sd)[:, 1]
auc_t = roc_auc_score(yb, s_transfer)
lo, hi = boot_ci_auroc(yb, s_transfer)
print(f"① frozen transfer MedQA->Bullets: AUROC={auc_t:.4f} CI=[{lo:.4f},{hi:.4f}] (reference line 0.56)")

# ---- ② single-feature AUROC table (no fitting), top-12 by |AUROC-0.5| ----
print("② single-feature AUROC on Bullets (no fitting):")
rows = []
for j in proc_idx + [names.index("bl_terminal_agreement")]:
    x = Xb[:, j]
    m = ~np.isnan(x)
    if m.sum() < 30 or len(set(yb[m])) < 2:
        continue
    a = roc_auc_score(yb[m], x[m])
    rows.append((names[j], a, int(m.sum())))
rows.sort(key=lambda r: -abs(r[1] - 0.5))
for nm, a, n in rows[:12]:
    print(f"   {nm:38s} {a:.4f} (n={n})")

# ---- ③ top-5 reduced CV (top-5 chosen by single-feature AUROC on MedQA to avoid selection bias on Bullets) ----
qa_single = []
for j in proc_idx:
    x = Xq[:, j]
    m = ~np.isnan(x)
    if m.sum() < 100:
        continue
    qa_single.append((j, abs(roc_auc_score(yq[m], x[m]) - 0.5)))
top5 = [j for j, _ in sorted(qa_single, key=lambda t: -t[1])[:5]]
print(f"③ top-5 (selected on MedQA): {[names[j] for j in top5]}")
oof = cv_scores(Xb, yb, top5, "logreg")
auc5 = roc_auc_score(yb, oof)
lo5, hi5 = boot_ci_auroc(yb, oof)
print(f"   Bullets top-5 CV AUROC={auc5:.4f} CI=[{lo5:.4f},{hi5:.4f}] (all-features reference 0.487)")
