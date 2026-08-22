"""Per-signal AURC (area under risk-coverage; lower better) for Table 1.
Signals mirror the main-table rows; combined uses the same 5-fold OOF as t3.
Usage: python analysis/aurc_table.py --results results --out-prefix results/aurc_table
Writes aurc_table_{medqa,mmlupro,medmcqa}.csv for gen_numbers to parse.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.t3_evaluate import cv_scores, load  # noqa: E402


def aurc(y: np.ndarray, score: np.ndarray) -> float:
    """Risk-coverage area: sort desc by score, cumulative error over coverage."""
    m = ~np.isnan(score)
    y, score = y[m], score[m]
    order = np.argsort(-score, kind="stable")
    err = 1 - y[order]
    cum_risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    return float(np.trapezoid(cum_risk, cov))


def merge_conf(rows_path: Path, key: str, qid_to_idx: dict, n: int) -> np.ndarray:
    s = np.full(n, np.nan)
    if not rows_path.exists():
        return s
    for line in open(rows_path):
        r = json.loads(line)
        if r.get(key) is not None and "error" not in r:
            i = qid_to_idx.get(int(r["qid"]))
            if i is not None:
                s[i] = float(r[key])
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out-prefix", default="results/aurc_table")
    args = parser.parse_args()
    R = Path(args.results)
    specs = (
        ("medqa", "features_medqa.csv", "verbalized_medqa.jsonl", "ptrue_medqa.jsonl"),
        ("mmlupro", "features_mmlupro.csv", "verbalized_mmlupro.jsonl", "ptrue_mmlupro.jsonl"),
        ("medmcqa", "features_medmcqa.csv", "verbalized_medmcqa.jsonl", "ptrue_medmcqa.jsonl"),
        ("medqa_huatuo", "features_medqa_huatuo.csv", None, None),
        ("mmlupro_huatuo", "features_mmlupro_huatuo.csv", None, None),
        ("medqa_llama", "features_medqa_llama.csv", None, None),
        ("mmlupro_llama", "features_mmlupro_llama.csv", None, None),
    )
    for ds, feat, verb, ptr in specs:
        X, y, cols, rows = load(R / feat)
        qid_to_idx = {int(r["qid"].split("_")[-1]): i for i, r in enumerate(rows)}
        proc_idx = [i for i, c in enumerate(cols) if not c.startswith("bl_")]
        signals = {
            "terminal": X[:, cols.index("bl_terminal_agreement")],
            "r1": X[:, cols.index("f0_1_r1_agreement")],
            "seq_logprob": X[:, cols.index("bl_seq_logprob_mean")],
            "ans_entropy": -X[:, cols.index("f1_1_entropy_first")],
            "combined": cv_scores(X, y, proc_idx, "logreg"),
        }
        if verb and (R / verb).exists():
            signals["verbalized"] = merge_conf(R / verb, "verbalized_conf", qid_to_idx, len(y))
        if ptr and (R / ptr).exists():
            signals["ptrue"] = merge_conf(R / ptr, "ptrue", qid_to_idx, len(y))
        out = Path(f"{args.out_prefix}_{ds}.csv")
        with open(out, "w") as fp:
            w = csv.writer(fp)
            w.writerow(["signal", "aurc"])
            for name, s in signals.items():
                w.writerow([name, f"{aurc(y, np.where(np.isnan(s), np.nanmedian(s), s)):.4f}"])
        print(f"{ds}: " + "  ".join(f"{n}={aurc(y, np.where(np.isnan(s), np.nanmedian(s), s)):.3f}"
                                     for n, s in signals.items()))


if __name__ == "__main__":
    main()
