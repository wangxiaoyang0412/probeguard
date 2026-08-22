"""fig_rc: risk-coverage curves per benchmark for the appendix.
Three panels (MedQA, MMLU-Pro medicine, MedMCQA), one curve per always-available
abstention signal (self-report baselines are compared in aggregate in Appendix F).
Combined uses the same 5-fold OOF protocol as t3. Coverage below 5% is omitted (noise).
Usage: .venv/bin/python analysis/fig_rc.py --results results --out ../paper/figures/fig_rc
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pubfig import save_figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.t3_evaluate import cv_scores, load  # noqa: E402

PANELS = (
    ("MedQA", "features_medqa.csv"),
    ("MMLU-Pro medicine", "features_mmlupro.csv"),
    ("MedMCQA", "features_medmcqa.csv"),
)
# Okabe-Ito for baselines; the process score is black and heavier.
STYLE = {
    "Terminal agreement": ("#999999", 0.9, ":"),
    "First-round agreement": ("#E69F00", 0.9, "-"),
    "Sequence log-probability": ("#56B4E9", 0.9, "-"),
    "Answer entropy": ("#009E73", 0.9, "-"),
    "Process score": ("#000000", 1.5, "-"),
}


def rc_curve(y: np.ndarray, score: np.ndarray) -> tuple:
    s = np.where(np.isnan(score), np.nanmedian(score), score)
    order = np.argsort(-s, kind="stable")
    err = 1 - y[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    return cov, np.cumsum(err) / np.arange(1, len(y) + 1)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="../paper/figures/fig_rc")
    args = parser.parse_args()
    R = Path(args.results)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, axes = plt.subplots(1, 3, sharey=False)
    for ax, (title, feat) in zip(axes, PANELS):
        X, y, cols, rows = load(R / feat)
        signals = {
            "Terminal agreement": X[:, cols.index("bl_terminal_agreement")],
            "First-round agreement": X[:, cols.index("f0_1_r1_agreement")],
            "Sequence log-probability": X[:, cols.index("bl_seq_logprob_mean")],
            "Answer entropy": -X[:, cols.index("f1_1_entropy_first")],
        }
        proc = [i for i, c in enumerate(cols) if not c.startswith("bl_")]
        signals["Process score"] = cv_scores(X, y, proc, "logreg")
        for name, s in signals.items():
            color, lw, ls = STYLE[name]
            cov, risk = rc_curve(y, s)
            ax.plot(cov, 100 * risk, color=color, lw=lw, ls=ls,
                    label=name, zorder=3 if name == "Process score" else 2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Coverage")
        ax.set_xlim(0.05, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        print(f"{title}: n={len(y)}")
    axes[0].set_ylabel("Selective risk (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=6.6, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(wspace=0.28)
    paths = save_figure(fig, args.out + ".pdf", spec="nature", width=140, aspect_ratio=0.38)
    print("wrote:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
