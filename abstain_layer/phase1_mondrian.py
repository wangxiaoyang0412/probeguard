"""Phase 1: Mondrian stratified certification (strata g pre-specified before evaluation data were inspected; training-free scores).

Strata (pre-specified): S1 unanimous (r1==1.0); S2 disagreement then converged; S3 not converged (abstain on the whole stratum).
Training-free scores: S1 -> f3_5_mean_bidir (soft aggregation); S2 -> f0_5_agreement_auc.
Within each stratum: 50/50 cal/test split, LTT-Bonferroni (delta split across strata)
certifies the per-stratum threshold; S3 abstains outright.
Reports per-stratum coverage/risk plus globally composed coverage/risk.
Reference: r1 run globally on the same machinery (no stratification).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from abstain_layer.phase1_conformal import evaluate, ltt_threshold
from abstain_layer.t3_evaluate import load

SEED = 42


def load_soft(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {r["qid"]: float(r["f3_5_mean_bidir"]) for r in rows}


def certify_layer(scores, y, alpha, delta):
    idx = np.arange(len(y))
    if len(idx) < 40 or len(set(y)) < 2:
        return None  # stratum too small: no certification (abstain on the whole stratum)
    cal, te = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
    thr, cov_cal = ltt_threshold(scores[cal], y[cal], alpha, delta=delta)
    cov, risk = evaluate(scores[te], y[te], thr)
    return {"n_cal": len(cal), "n_test": len(te), "thr": float(thr),
            "cal_cov": cov_cal, "test_cov": cov, "test_risk": risk,
            "test_idx": te, "answered": scores[te] > thr, "y_test": y[te]}


def run(features_csv, soft_csv, alphas):
    X, y, names, raw = load(features_csv)
    ds = Path(features_csv).stem.replace("features_", "")
    soft = load_soft(soft_csv)
    qids = [r["qid"] for r in raw]
    r1 = X[:, names.index("f0_1_r1_agreement")]
    stop_reason = np.array([r for r in
                            (rr.get("dataset") for rr in raw)])  # placeholder unused
    conv_round = X[:, names.index("f0_2_convergence_round")]
    agree_auc = X[:, names.index("f0_5_agreement_auc")]
    agree_auc = np.where(np.isnan(agree_auc), np.nanmedian(agree_auc), agree_auc)

    s1 = r1 == 1.0
    s3 = conv_round == 9.0  # t_max+1 = not converged
    s2 = (~s1) & (~s3)
    f5 = np.array([soft.get(q, np.nan) for q in qids])
    f5_s1 = f5[s1]
    f5_s1 = np.where(np.isnan(f5_s1), np.nanmedian(f5_s1), f5_s1)

    print(f"===== {ds}: n={len(y)} | S1={int(s1.sum())} S2={int(s2.sum())} S3={int(s3.sum())} =====")
    for alpha in alphas:
        delta_layer = 0.05 / 2  # delta split across the two certified strata
        res1 = certify_layer(f5_s1, y[s1], alpha, delta_layer)
        res2 = certify_layer(agree_auc[s2], y[s2], alpha, delta_layer)
        # global composition (over the test halves): S3 abstains entirely
        tot_test, tot_ans, tot_err = 0, 0, 0
        for res, mask in ((res1, s1), (res2, s2)):
            if res is None:
                continue
            tot_test += res["n_test"]
            tot_ans += int(res["answered"].sum())
            tot_err += int(((res["y_test"] == 0) & res["answered"]).sum())
        tot_test += int(s3.sum()) // 2  # S3's test half, all abstained
        g_cov = tot_ans / tot_test if tot_test else 0.0
        g_risk = tot_err / tot_ans if tot_ans else 0.0
        line = f"  α={alpha:.2f}: "
        for tag, res in (("S1", res1), ("S2", res2)):
            if res is None:
                line += f"{tag}[abstain-all] "
            else:
                line += (f"{tag}[cov={res['test_cov']:.2f} risk={res['test_risk']:.3f}] ")
        line += f"| GLOBAL cov={g_cov:.3f} risk={g_risk:.3f}"
        print(line)
        # reference: r1 globally without stratification (same machinery)
        idx = np.arange(len(y))
        cal, te = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
        thr, _ = ltt_threshold(r1[cal], y[cal], alpha, delta=0.05)
        cov, risk = evaluate(r1[te], y[te], thr)
        print(f"          baseline r1-global: cov={cov:.3f} risk={risk:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="features_csv:soft_csv pairs")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    args = parser.parse_args()
    for pair in args.pairs:
        f_csv, s_csv = pair.split(":")
        run(f_csv, s_csv, args.alphas)


if __name__ == "__main__":
    main()
