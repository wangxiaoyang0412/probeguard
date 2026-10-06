"""C9: per-specialty view of the S1 certificate on MedQA (Appendix H).
The rule is calibrated globally on the unanimous layer (same procedure as holes.h3);
this table reports how its coverage and realized risk distribute across specialties.
Usage: .venv/bin/python analysis/specialty_table.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.phase1_conformal import ltt_threshold  # noqa: E402

ALPHA = 0.15
DELTA = 0.025  # per-stratum share, matching the main result
R = 50
MIN_N = 25    # specialties below this fold into "Other"


def load_s1():
    probe = {json.loads(l)["qid"]: json.loads(l) for l in open("results/probe_pilot_medqa.jsonl")
             if "error" not in json.loads(l)}
    soft = {r["qid"]: float(r["f3_5_mean_bidir"]) for r in csv.DictReader(open("results/soft_f3_medqa_ua.csv"))}
    labels = json.load(open("results/medqa_specialty_labels.json"))
    y, s, spec = [], [], []
    for q, r in probe.items():
        sq = f"medqa_{q}"
        if sq in soft:
            y.append(int(r["consensus_correct"]))
            s.append((r["keep_rate"], soft[sq]))
            spec.append(labels.get(str(q), "Unlabeled"))
    y = np.array(y)
    s = np.array(s)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    return z(s[:, 0]) + z(s[:, 1]), y, np.array(spec)


def main() -> None:
    comb, y, spec = load_s1()
    counts = defaultdict(int)
    for sp in spec:
        counts[sp] += 1
    keep_named = {sp for sp, c in counts.items() if c >= MIN_N and sp != "Unlabeled"}
    grp = np.array([sp if sp in keep_named else ("Unlabeled" if sp == "Unlabeled" else "Other")
                    for sp in spec])
    idx = np.arange(len(y))
    cov_acc = defaultdict(list)
    risk_acc = defaultdict(list)
    for seed in range(R):
        cal, te = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
        thr, _ = ltt_threshold(comb[cal], y[cal], ALPHA, delta=DELTA)
        if thr is None:
            continue
        ans = comb[te] > thr
        for g in set(grp[te]):
            m = grp[te] == g
            n_ans = (ans & m).sum()
            cov_acc[g].append(n_ans / max(m.sum(), 1))
            if n_ans > 0:
                risk_acc[g].append(1 - y[te][ans & m].mean())
    rows = []
    for g in sorted(cov_acc, key=lambda g: -np.mean(cov_acc[g])):
        if g == "Unlabeled":
            continue
        rows.append((g, counts.get(g, sum(c for s2, c in counts.items()
                                          if s2 not in keep_named and s2 != "Unlabeled")),
                     np.mean(cov_acc[g]), np.std(cov_acc[g]) / np.sqrt(len(cov_acc[g])),
                     100 * np.mean(risk_acc[g]) if risk_acc[g] else float("nan")))
    with open("results/specialty_cert_medqa.csv", "w") as fp:
        w = csv.writer(fp)
        w.writerow(["specialty", "n_ua", "coverage_mean", "coverage_se", "risk_pct_mean"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.1f}"])
    print(f"S1 layer n={len(y)}, labeled={sum(1 for g in grp if g != 'Unlabeled')}, "
          f"named specialties={len(keep_named)}, alpha={ALPHA}, R={R}")
    for g, n, cm, cse, rm in rows:
        print(f"{g:38s} n={n:4d}  cov={cm:.3f}±{cse:.3f}  risk={rm:5.1f}%")


if __name__ == "__main__":
    main()
