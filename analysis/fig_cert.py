"""fig_cert: certification validity across repeated splits (Appendix E / Sec 4.4 figure).
Strip plot of realized selective risk per calibration/test split for three certified
settings; each row carries its own dashed bound at alpha. Every point left of its bound
is the guarantee holding. Protocol identical to analysis/c8_fuel_cert.py.
Usage: .venv/bin/python analysis/fig_cert.py --out ../paper/figures/fig_cert
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pubfig import save_figure
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.phase1_conformal import evaluate, ltt_threshold  # noqa: E402
from analysis.c8_fuel_cert import R_SPLITS, load_layer  # noqa: E402

DOT_C = "#4C72B0"
BOUND_C = "#C44E52"
SPINE = "0.55"


def split_risks(y_m, s_m, y_f, s_f, alpha: float, use_fuel: bool) -> np.ndarray:
    idx = np.arange(len(y_m))
    risks = []
    for seed in range(R_SPLITS):
        cal, te = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y_m)
        cal_s, cal_y = s_m[cal], y_m[cal]
        if use_fuel:
            cal_s = np.concatenate([cal_s, s_f])
            cal_y = np.concatenate([cal_y, y_f])
        thr, _ = ltt_threshold(cal_s, cal_y, alpha, delta=0.025)
        cov, risk = evaluate(s_m[te], y_m[te], thr)
        if cov > 0:
            risks.append(risk)
    return np.array(risks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="../paper/figures/fig_cert")
    args = parser.parse_args()
    y_m, s_m = load_layer(f"{args.results}/probe_pilot_medqa.jsonl",
                          f"{args.results}/soft_f3_medqa_ua.csv", "medqa")
    y_f, s_f = load_layer(f"{args.results}/probe_fuel_gbaker.jsonl",
                          f"{args.results}/soft_f3_fuel_ua.csv", "medqa")

    rows = [
        (r"$\alpha=0.15$, base calibration", split_risks(y_m, s_m, y_f, s_f, 0.15, False), 0.15),
        (r"$\alpha=0.15$, with in-domain calibration", split_risks(y_m, s_m, y_f, s_f, 0.15, True), 0.15),
        (r"$\alpha=0.10$, with in-domain calibration", split_risks(y_m, s_m, y_f, s_f, 0.10, True), 0.10),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.0, "axes.labelsize": 5.7,
        "xtick.labelsize": 5.5, "ytick.labelsize": 6.0, "axes.labelpad": 2.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    for i, (label, risks, alpha) in enumerate(rows):
        yjit = i + rng.uniform(-0.16, 0.16, len(risks))
        ax.scatter(risks, yjit, s=7, color=DOT_C, alpha=0.55, lw=0, zorder=3)
        ax.plot([alpha, alpha], [i - 0.30, i + 0.30], ls="--", lw=1.0, color=BOUND_C, zorder=4)
        ax.scatter([risks.mean()], [i], marker="D", s=14, color="0.1", zorder=5)
        ax.text(alpha + 0.003, i + 0.24, rf"$\alpha$", fontsize=6.2, color=BOUND_C, va="center")
        print(f"{label}: n={len(risks)} mean={risks.mean():.4f} max={risks.max():.4f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Realized selective risk per calibration/test split")
    ax.set_xlim(0, 0.19)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=SPINE, labelcolor="0.15")
    paths = save_figure(fig, args.out + ".pdf", spec="nature", width=120, aspect_ratio=0.26)
    print("wrote:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
