"""Two-dataset probe replication analysis + pooled cross-dataset calibration targeting α=0.10."""
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from abstain_layer.phase1_conformal import evaluate, ltt_threshold
from abstain_layer.t3_evaluate import boot_ci_auroc


def load_layer(probe_path, soft_path, prefix):
    probe = {json.loads(l)["qid"]: json.loads(l)
             for l in open(probe_path) if "error" not in json.loads(l)}
    soft = {r["qid"]: float(r["f3_5_mean_bidir"])
            for r in csv.DictReader(open(soft_path))}
    y, keep, f5 = [], [], []
    for q, r in probe.items():
        sq = f"{prefix}_{q}"
        if sq not in soft:
            continue
        y.append(int(r["consensus_correct"]))
        keep.append(r["keep_rate"])
        f5.append(soft[sq])
    return np.array(y), np.array(keep), np.array(f5), probe


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


layers = {}
for ds, pp, sp, pre in (
        ("medqa", "results/probe_pilot_medqa.jsonl", "results/soft_f3_medqa_ua.csv", "medqa"),
        ("mmlupro", "results/probe_mmlupro_ua.jsonl", "results/soft_f3_mmlupro_ua.csv", "mmlu-pro")):
    y, keep, f5, probe = load_layer(pp, sp, pre)
    layers[ds] = (y, keep, f5)
    wrong = (y == 0)
    fw = np.mean([r["flipped"] for r in probe.values() if not r["consensus_correct"]])
    fr = np.mean([r["flipped"] for r in probe.values() if r["consensus_correct"]])
    comb = z(keep) + z(f5)
    a_k = roc_auc_score(y, keep)
    a_c = roc_auc_score(y, comb)
    lo, hi = boot_ci_auroc(y, comb)
    med_s = np.median([r.get("elapsed_s", np.nan) for r in probe.values()
                       if "elapsed_s" in r]) if any("elapsed_s" in r for r in probe.values()) else float("nan")
    print(f"{ds:8s}: n={len(y)} err={wrong.mean():.3f} | flip(wrong)={fw:.3f} flip(right)={fr:.3f} "
          f"| keep AUROC={a_k:.3f} keep+f3_5={a_c:.3f} CI=[{lo:.3f},{hi:.3f}] | med_cost={med_s:.0f}s")

# per-dataset S1 certification (each dataset independently)
print("\n== per-dataset S1 certification (keep+f3_5) ==")
for ds, (y, keep, f5) in layers.items():
    comb = z(keep) + z(f5)
    idx = np.arange(len(y))
    cal, te = train_test_split(idx, test_size=0.5, random_state=42, stratify=y)
    for alpha in (0.10, 0.15):
        thr, cc = ltt_threshold(comb[cal], y[cal], alpha, delta=0.025)
        cov, risk = evaluate(comb[te], y[te], thr)
        print(f"  {ds:8s} α={alpha:.2f}: test-cov={cov:.3f} test-risk={risk:.4f}")

# cross-dataset pooled calibration (pooled calib, per-dataset test)
print("\n== pooled calibration (medqa+mmlupro cal halves merged) ==")
parts = {}
for ds, (y, keep, f5) in layers.items():
    comb = z(keep) + z(f5)  # z-score within dataset before pooling (scale alignment)
    idx = np.arange(len(y))
    cal, te = train_test_split(idx, test_size=0.5, random_state=42, stratify=y)
    parts[ds] = (comb, y, cal, te)
cal_scores = np.concatenate([parts[d][0][parts[d][2]] for d in parts])
cal_y = np.concatenate([parts[d][1][parts[d][2]] for d in parts])
for alpha in (0.05, 0.10, 0.15):
    thr, cc = ltt_threshold(cal_scores, cal_y, alpha, delta=0.025)
    line = f"  α={alpha:.2f}: cal-cov={cc:.2f}"
    for d in parts:
        comb, y, cal, te = parts[d]
        cov, risk = evaluate(comb[te], y[te], thr)
        line += f" | {d} test-cov={cov:.3f} risk={risk:.4f}"
    print(line)
