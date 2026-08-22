"""Paired-test artifact for combined(OOF logreg) vs r1_agreement (supplementary analysis).
Same protocol as t3_evaluate: 5-fold OOF + question-level paired bootstrap (seed 42), plus one-sided p-value.
CPU-only, runs locally: python analysis/paired_delta.py --results results --out results/paired_combined_vs_r1.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.t3_evaluate import cv_scores, load  # noqa: E402

N_BOOT = 2000
SEED = 42


def paired_delta(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> tuple:
    """Paired bootstrap over questions: delta = AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_onesided = float(np.mean(deltas <= 0))  # H1: delta > 0
    return float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b)), lo, hi, p_onesided


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/paired_combined_vs_r1.txt")
    args = parser.parse_args()
    lines = [f"combined(OOF logreg) vs r1_agreement — paired bootstrap B={N_BOOT}, seed={SEED}",
             "supplementary analysis", ""]
    for name, csv in (("MedQA", "features_medqa.csv"), ("MMLU-Pro", "features_mmlupro.csv")):
        X, y, cols, _ = load(Path(args.results) / csv)
        r1 = X[:, cols.index("f0_1_r1_agreement")]
        proc_idx = [i for i, c in enumerate(cols) if not c.startswith("bl_")]
        oof = cv_scores(X, y, proc_idx, "logreg")
        auc_c, auc_r = roc_auc_score(y, oof), roc_auc_score(y, np.nan_to_num(r1, nan=np.nanmedian(r1)))
        d, lo, hi, p = paired_delta(y, oof, np.nan_to_num(r1, nan=np.nanmedian(r1)))
        lines.append(f"{name}: n={len(y)} combined={auc_c:.4f} r1={auc_r:.4f} "
                     f"delta={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] p(one-sided,delta<=0)={p:.4f}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
