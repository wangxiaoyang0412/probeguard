"""C8: fuel-augmented Mondrian certification on the MedQA unanimous layer.
Protocol identical to the repeated-split analysis (R=50 plain random 50/50 splits of
the MedQA S1 layer, LTT with delta=0.025, pre-registered score z(keep)+z(rse) with
per-dataset standardization); the only change is that the calibration side is enlarged
with the in-domain fuel set (GBaker train, unanimous layer). Test halves are untouched.
Usage: .venv/bin/python analysis/c8_fuel_cert.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.phase1_conformal import evaluate, ltt_threshold  # noqa: E402

R_SPLITS = 50
ALPHAS = (0.05, 0.10, 0.15, 0.20)


def zsum(keep: np.ndarray, rse: np.ndarray) -> np.ndarray:
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    return z(keep) + z(rse)


def load_layer(probe_path: str, soft_path: str, prefix: str) -> tuple:
    probe = {json.loads(l)["qid"]: json.loads(l) for l in open(probe_path)
             if "error" not in json.loads(l)}
    soft = {r["qid"]: float(r["f3_5_mean_bidir"]) for r in csv.DictReader(open(soft_path))}
    y, keep, rse = [], [], []
    for q, r in probe.items():
        sq = f"{prefix}_{q}"
        if sq in soft:
            y.append(int(r["consensus_correct"]))
            keep.append(r["keep_rate"])
            rse.append(soft[sq])
    return np.array(y), zsum(np.array(keep), np.array(rse))


def main() -> None:
    y_m, s_m = load_layer("results/probe_pilot_medqa.jsonl", "results/soft_f3_medqa_ua.csv", "medqa")
    y_f, s_f = load_layer("results/probe_fuel_gbaker.jsonl", "results/soft_f3_fuel_ua.csv", "medqa")
    print(f"MedQA S1 n={len(y_m)} err={100*(1-y_m.mean()):.1f}% | fuel n={len(y_f)} err={100*(1-y_f.mean()):.1f}%")
    idx = np.arange(len(y_m))
    for label, use_fuel in (("baseline (cal=half)", False), ("fuel-augmented (cal=half+fuel)", True)):
        print(f"== {label} ==")
        for alpha in ALPHAS:
            covs, risks = [], []
            for seed in range(R_SPLITS):
                cal, te = train_test_split(idx, test_size=0.5, random_state=seed)
                cal_s, cal_y = s_m[cal], y_m[cal]
                if use_fuel:
                    cal_s = np.concatenate([cal_s, s_f])
                    cal_y = np.concatenate([cal_y, y_f])
                thr, _ = ltt_threshold(cal_s, cal_y, alpha, delta=0.025)
                cov, risk = evaluate(s_m[te], y_m[te], thr)
                covs.append(cov)
                risks.append(risk if cov > 0 else np.nan)
            covs = np.array(covs)
            nz = int((covs > 0).sum())
            mrisk = np.nanmean(risks) if nz else float("nan")
            print(f"  a={alpha:.2f}: cov={covs.mean():.3f}+-{covs.std()/np.sqrt(R_SPLITS):.3f} "
                  f"nonzero {nz}/{R_SPLITS} risk|ans={mrisk:.4f}")


if __name__ == "__main__":
    main()
