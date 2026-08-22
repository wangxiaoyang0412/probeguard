"""Discriminative-power evaluation: feature CSVs -> AUROC tables + routing decision.

Decision rule (fixed in advance):
  Auto-continue (the only automatic path): combined process-feature AUROC exceeds
      bl_terminal_agreement by >= +0.02 on both datasets -> AUTO-CONTINUE.
  Every other case -> ESCALATE: emit stratified results for a human decision
      (continue / downgrade / stop).
  This script never halts the project on its own; CIs, the strict reference, and
      sub-gates are reporting-only.
Usage: python -m abstain_layer.t3_evaluate --features results/features_medqa.csv \
        results/features_medxpertqa.csv --out-dir results/
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROCESS_PREFIXES = ("f0_", "f1_", "f2_", "f3_", "f4_")
BASELINE_COLS = ("bl_terminal_agreement", "bl_seq_logprob_mean")
META_COLS = ("qid", "dataset", "correct")
RNG_SEED = 42
N_BOOT = 1000
MAIN_GATE_MARGIN = 0.02


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    y = np.array([int(r["correct"]) for r in rows])
    feat_names = [k for k in rows[0] if k not in META_COLS]
    X = np.full((len(rows), len(feat_names)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(feat_names):
            v = r[k]
            if v not in ("", "None"):
                X[i, j] = float(v)
    return X, y, feat_names, rows


def single_auroc(x, y):
    m = ~np.isnan(x)
    if m.sum() < 10 or len(set(y[m])) < 2:
        return None
    return roc_auc_score(y[m], x[m])


def cv_scores(X, y, cols_idx, model_kind, seed=RNG_SEED):
    """5-fold CV with per-fold median imputation + missingness indicators + scaling (logreg only); returns per-question OOF scores."""
    Xs = X[:, cols_idx]
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, y):
        med = np.nanmedian(Xs[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)

        def prep(a):
            miss = np.isnan(a).astype(float)
            filled = np.where(np.isnan(a), med, a)
            return np.hstack([filled, miss])

        Xtr, Xte = prep(Xs[tr]), prep(Xs[te])
        if model_kind == "logreg":
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            model = LogisticRegression(max_iter=2000, C=1.0)
        else:
            model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]
    return oof


def boot_ci_diff(y, s_a, s_b, n_boot=N_BOOT, seed=RNG_SEED):
    """Question-level bootstrap 95% CI for AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        diffs.append(roc_auc_score(yy, s_a[idx]) - roc_auc_score(yy, s_b[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def boot_ci_auroc(y, s, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def risk_coverage(y, score, out_path):
    order = np.argsort(-score)
    err = 1 - y[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    with open(out_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["coverage", "risk"])
        w.writerows(zip(cov.tolist(), risk.tolist()))
    return float(np.trapezoid(risk, cov))  # AURC


def evaluate_dataset(path, out_dir):
    X, y, names, _ = load(path)
    ds = Path(path).stem.replace("features_", "")
    print(f"\n===== {ds}: n={len(y)}, acc={y.mean():.4f} =====")

    table = []
    for j, name in enumerate(names):
        a = single_auroc(X[:, j], y)
        if a is not None:
            lo, hi = boot_ci_auroc(y, np.where(np.isnan(X[:, j]),
                                               np.nanmedian(X[:, j]), X[:, j]))
            table.append((name, a, lo, hi))
    proc_idx = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]
    term_idx = names.index("bl_terminal_agreement")
    term = X[:, term_idx]

    results = {}
    for kind in ("logreg", "gbt"):
        oof = cv_scores(X, y, proc_idx, kind)
        auc = roc_auc_score(y, oof)
        d_lo, d_hi = boot_ci_diff(y, oof, term)
        results[kind] = (auc, d_lo, d_hi, oof)
        table.append((f"COMBINED_{kind}", auc, *boot_ci_auroc(y, oof)))

    term_auc = roc_auc_score(y, term)
    table.append(("bl_terminal_agreement(ref)", term_auc, *boot_ci_auroc(y, term)))
    # Methodological note: with early stopping, converged questions have terminal_agreement == 1.0,
    # so this baseline is near-constant. A fair snapshot baseline is round-1 agreement
    # (single-round sampling consistency), hence the strict reference reported alongside.
    r1 = X[:, names.index("f0_1_r1_agreement")]
    r1_auc = roc_auc_score(y, r1)
    table.append(("bl_r1_agreement(strict-ref)", r1_auc, *boot_ci_auroc(y, r1)))

    best_kind = max(results, key=lambda k: results[k][0])
    best_auc, d_lo, d_hi, best_oof = results[best_kind]
    delta = best_auc - term_auc
    main_pass = delta >= MAIN_GATE_MARGIN          # auto-continue condition (CI is reporting-only, not routed on)
    r1_lo, r1_hi = boot_ci_diff(y, best_oof, r1)
    strict_delta = best_auc - r1_auc
    strict_pass = strict_delta >= MAIN_GATE_MARGIN and r1_lo > 0   # reporting-only

    ua = X[:, names.index("f0_1_r1_agreement")] == 1.0
    sub_pass, sub_auc, sub_lo = None, None, None
    f3_idx = [j for j, n in enumerate(names) if n.startswith("f3_")]
    if ua.sum() >= 30 and len(set(y[ua])) == 2 and f3_idx:
        sub_oof = cv_scores(X[ua], y[ua], f3_idx, "logreg")
        sub_auc = roc_auc_score(y[ua], sub_oof)
        sub_lo, _ = boot_ci_auroc(y[ua], sub_oof)
        sub_pass = sub_auc > 0.5 and sub_lo > 0.5
    aurc = risk_coverage(y, best_oof, Path(out_dir) / f"risk_coverage_{ds}.csv")

    with open(Path(out_dir) / f"auroc_table_{ds}.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["feature", "auroc", "ci_lo", "ci_hi"])
        w.writerows(sorted(table, key=lambda r: -r[1]))

    print(f"combined[{best_kind}]={best_auc:.4f} vs terminal={term_auc:.4f} "
          f"delta={delta:+.4f} diff-CI=[{d_lo:+.4f},{d_hi:+.4f}] "
          f"-> {'AUTO-OK' if main_pass else 'ESCALATE'}")
    print(f"  [report-only strict] vs r1_agreement={r1_auc:.4f} delta={strict_delta:+.4f} "
          f"diff-CI=[{r1_lo:+.4f},{r1_hi:+.4f}] "
          f"({'clears' if strict_pass else 'below'} +0.02 w/ CI)")
    if sub_pass is not None:
        print(f"  [report-only unanimous-subset] n={int(ua.sum())} err={1-y[ua].mean():.3f} "
              f"f3-AUROC={sub_auc:.4f} CI-lo={sub_lo:.4f} "
              f"({'above' if sub_pass else 'not above'} 0.5 w/ CI)")
    else:
        print(f"  [report-only unanimous-subset] n={int(ua.sum())}: too few samples or one class, N/A")
    print(f"AURC(best)={aurc:.4f}")
    return main_pass, sub_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    mains, subs = [], []
    for path in args.features:
        m, s = evaluate_dataset(path, args.out_dir)
        mains.append(m)
        subs.append(s)
    overall = all(mains)
    if overall:
        print("\n########## PHASE0 ROUTING: AUTO-CONTINUE "
              "(delta >= +0.02 on both datasets) ##########")
    else:
        print("\n########## PHASE0 ROUTING: ESCALATE TO USER ##########")
        print("Gate not met on both datasets -> escalate per the decision rule: "
              "present stratified results (full AUROC tables / strict reference / "
              "unanimous subset / risk-coverage) for a human decision to "
              "continue / downgrade / stop. This pipeline cannot stop the project itself.")


if __name__ == "__main__":
    main()
