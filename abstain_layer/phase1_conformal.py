"""Phase 1 conformal abstention layer (pilot uses a numpy CRC implementation;
can later be cross-checked against Angelopoulos conformal-risk / TorchCP).

Procedure (per dataset x alpha x signal):
  1. Question-level random 50/50 split into calib/test (stratified by correctness, fixed seed).
  2. Scorer (logreg on process features): 5-fold CV inside calib yields honest OOF calib
     scores (keeps scorer overfitting from contaminating the threshold); then the final
     scorer is fit on all of calib to score test.
  3. CRC threshold: over candidate thresholds t (calib scores in descending order), answer
     set = {score >= t}; pick the maximum-coverage threshold with (errors+1)/(answered+1) <= alpha.
  4. Report on test: empirical selective error (should be around <= alpha), coverage, AURC.
Baselines: bl_terminal_agreement / f0_1_r1_agreement used directly as scores through the same CRC layer.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from abstain_layer.t3_evaluate import PROCESS_PREFIXES, load

SEED = 42


def fit_scorer(X, y, idx_train, proc_idx):
    """Fit a fixed scorer on train; cal/test share the same scorer -> scores are exchangeable."""
    Xf, yf = X[np.ix_(idx_train, proc_idx)], y[idx_train]
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def prep(a):
        miss = np.isnan(a).astype(float)
        return np.hstack([np.where(np.isnan(a), med, a), miss])

    Zf = prep(Xf)
    mu, sd = Zf.mean(0), Zf.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(max_iter=2000).fit((Zf - mu) / sd, yf)

    def score(idx_apply):
        Za = (prep(X[np.ix_(idx_apply, proc_idx)]) - mu) / sd
        return clf.predict_proba(Za)[:, 1]
    return score


def _binom_tail_pvalue(errors, answered, alpha):
    """p-value for H0: true selective risk > alpha, i.e. P[Bin(answered, alpha) <= errors]."""
    from scipy.stats import binom
    return float(binom.cdf(errors, answered, alpha))


def ltt_threshold(scores_cal, y_cal, alpha, delta=0.05,
                  coverage_grid=np.arange(0.05, 1.0001, 0.05)):
    """LTT-Bonferroni: over a coverage grid, test H0: risk > alpha at level delta/|grid|
    per point; take the largest coverage that passes. Valid for non-monotone selective
    risk, and not vetoed by a few errors at the top of the ranking.

    Each candidate threshold is a calibration order statistic, and the rule answers the
    questions scoring strictly above it. Conditional on that order statistic, the
    calibration questions above it are i.i.d. draws from the answered population, so the
    exact binomial test is run on those questions."""
    n = len(scores_cal)
    order = np.argsort(-scores_cal)
    errs = (y_cal[order] == 0).cumsum()
    thresholds = scores_cal[order]
    delta_each = delta / len(coverage_grid)
    best = (np.inf, 0.0)
    for c in coverage_grid:
        k = max(int(round(c * n)) - 1, 0)
        lam = thresholds[k]
        # answered means score > lam: the threshold question and its ties are not answered
        n_above = int(np.searchsorted(-thresholds, -lam, side="left"))
        if n_above == 0:
            continue
        p = _binom_tail_pvalue(int(errs[n_above - 1]), n_above, alpha)
        if p <= delta_each and n_above / n > best[1]:
            best = (lam, n_above / n)
    return best


def ltt_threshold_split(scores_cal, y_cal, alpha, delta=0.05,
                        coverage_grid=np.arange(0.05, 1.0001, 0.05),
                        grid_frac=0.2, seed=0):
    """Split-LTT variant used in protocol robustness checks: a held-out grid_frac of
    the calibration set fixes the candidate thresholds (its score quantiles at the
    coverage grid); the LTT-Bonferroni binomial tests run on the remaining part."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(scores_cal))
    n_grid = max(int(round(grid_frac * len(idx))), 1)
    grid_part, test_part = idx[:n_grid], idx[n_grid:]
    thresholds = np.unique(np.quantile(scores_cal[grid_part], 1.0 - coverage_grid))
    s, yy = scores_cal[test_part], y_cal[test_part]
    n = len(s)
    delta_each = delta / len(thresholds)
    best = (np.inf, 0.0)
    for lam in thresholds:
        ans = s > lam
        k = int(ans.sum())
        if k == 0:
            continue
        errs = int((yy[ans] == 0).sum())
        p = _binom_tail_pvalue(errs, k, alpha)
        if p <= delta_each and k / n > best[1]:
            best = (lam, k / n)
    return best


def evaluate(scores_te, y_te, thr):
    ans = scores_te > thr
    cov = ans.mean()
    risk = float((y_te[ans] == 0).mean()) if ans.any() else 0.0
    return cov, risk


def aurc(scores, y):
    order = np.argsort(-scores)
    err = (y[order] == 0)
    risk = err.cumsum() / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    return float(np.trapezoid(risk, cov))


def run_dataset(path, alphas, out_rows):
    X, y, names, _ = load(path)
    ds = Path(path).stem.replace("features_", "")
    proc_idx = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]
    idx = np.arange(len(y))
    # 40/30/30 three-way split: the scorer sees only train; cal/test are scored by the same frozen scorer (exchangeable)
    tr, rest = train_test_split(idx, test_size=0.6, random_state=SEED, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y[rest])

    score_fn = fit_scorer(X, y, tr, proc_idx)
    signals = {"process_combined": (score_fn(cal), score_fn(te))}
    for bl in ("bl_terminal_agreement", "f0_1_r1_agreement"):
        col = X[:, names.index(bl)]
        col = np.where(np.isnan(col), np.nanmedian(col), col)
        signals[bl] = (col[cal], col[te])

    print(f"===== {ds}: n={len(y)} (train={len(tr)}/cal={len(cal)}/test={len(te)}) acc={y.mean():.4f} =====")
    for sig, (s_cal, s_test) in signals.items():
        a_auc = roc_auc_score(y[te], s_test)
        for alpha in alphas:
            thr, cov_cal = ltt_threshold(np.asarray(s_cal), y[cal], alpha)
            cov, risk = evaluate(np.asarray(s_test), y[te], thr)
            ok = "OK " if risk <= alpha + 0.01 else "VIOL"
            print(f"  {sig:22s} α={alpha:.2f}: coverage={cov:.3f} sel-risk={risk:.4f} [{ok}] "
                  f"(cal-cov={cov_cal:.3f}, test-AUROC={a_auc:.3f}, AURC={aurc(np.asarray(s_test), y[te]):.4f})")
            out_rows.append({"dataset": ds, "signal": sig, "alpha": alpha,
                             "coverage": round(cov, 4), "sel_risk": round(risk, 4),
                             "cal_coverage": round(cov_cal, 4),
                             "test_auroc": round(a_auc, 4)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for p in args.features:
        run_dataset(p, args.alphas, rows)
    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
