"""fig_flip: flip rates of wrong vs correct consensus under counter-evidence.
Single-panel grouped bars over (backbone x dataset) cells, Wilson 95% CIs;
per-cell keep-rate AUROC printed beneath each pair (computed from the same jsonl).
Auto-includes llama cells when their probe jsonls exist.
Usage: .venv/bin/python analysis/fig_flip.py --results results --out ../paper/figures/fig_flip
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pubfig import save_figure
from sklearn.metrics import roc_auc_score

WRONG_C = "#E64B35"   # vivid vermillion (Fig 3 runs saturated by design)
RIGHT_C = "#2E86DE"   # vivid azure
SPINE = "0.55"

CELLS = [  # (label, filename)
    ("MedQA\nQwen3-8B", "probe_pilot_medqa.jsonl"),
    ("MMLU-Pro\nQwen3-8B", "probe_mmlupro_ua.jsonl"),
    ("MedMCQA\nQwen3-8B", "probe_medmcqa_ua.jsonl"),
    ("MedQA\nHuatuo-o1", "probe_medqa_ua_huatuo.jsonl"),
    ("MMLU-Pro\nHuatuo-o1", "probe_mmlupro_ua_huatuo.jsonl"),
    ("MedQA\nLlama-3.1", "probe_medqa_ua_llama.jsonl"),
    ("MMLU-Pro\nLlama-3.1", "probe_mmlupro_ua_llama.jsonl"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cell_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if "error" not in r]
    if len(ok) < 50:  # still being written
        return None
    out = {}
    for key, grp in (("wrong", [r for r in ok if not r["consensus_correct"]]),
                     ("right", [r for r in ok if r["consensus_correct"]])):
        k = sum(1 for r in grp if r["flipped"])
        n = len(grp)
        lo, hi = wilson(k, n)
        out[key] = (100 * k / n, 100 * lo, 100 * hi, n)
    y = np.array([int(r["consensus_correct"]) for r in ok])
    keep = np.array([r["keep_rate"] for r in ok], float)
    out["keep_auroc"] = roc_auc_score(y, keep)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="../paper/figures/fig_flip")
    args = parser.parse_args()
    R = Path(args.results)

    labels, stats = [], []
    for label, fn in CELLS:
        s = cell_stats(R / fn)
        if s is not None:
            labels.append(label)
            stats.append(s)
    print(f"cells included: {len(labels)}")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.0, "axes.labelsize": 5.7,
        "xtick.labelsize": 5.2, "ytick.labelsize": 5.0, "axes.labelpad": 2.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    w = 0.36
    for off, key, color, name in ((-w / 2, "wrong", WRONG_C, "Wrong consensus"),
                                  (w / 2, "right", RIGHT_C, "Correct consensus")):
        vals = [s[key][0] for s in stats]
        errlo = [s[key][0] - s[key][1] for s in stats]
        errhi = [s[key][2] - s[key][0] for s in stats]
        ax.bar(x + off, vals, w, color=color, label=name, zorder=2)
        ax.errorbar(x + off, vals, yerr=[errlo, errhi], fmt="none",
                    ecolor="0.25", elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        his = [s[key][2] for s in stats]
        for xi, v, hi in zip(x + off, vals, his):
            ax.text(xi, hi + 0.8, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=5.4, color="0.2")
    # backbone group separators
    prev = labels[0].split("\n")[1]
    for i, lab in enumerate(labels[1:], 1):
        bb = lab.split("\n")[1]
        if bb != prev:
            ax.axvline(i - 0.5, color="0.85", lw=0.6, zorder=1)
            prev = bb
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flip rate (%)")
    ax.set_ylim(0, max(s["wrong"][2] for s in stats) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=SPINE, labelcolor="0.15")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=5.8)
    paths = save_figure(fig, args.out + ".pdf", spec="nature", width=140, aspect_ratio=0.26)
    print("wrote:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
