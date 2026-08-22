"""fig_formation: anatomy of consensus formation on MedQA (results opener triptych).
Three panel styles per venue convention survey (SConU / Conformal Alignment / CLM):
(left)  agreement trajectory per round, correct vs wrong, shaded SE bands (no error bars);
(mid)   bar+line dual-axis: questions concluding per round (bars) vs error rate (line),
        dashed reference at the overall error rate;
(right) horizontal AUROC bars on the unanimous layer, Blues ramp, chance dashed line.
Palette sampled from published figures: #C44E52/#4C72B0 series, #F5D0DA/#CCE1EC bands,
#B9C4D6->#7694C0->#296BBC ramp, #D2B48C baseline tan. White gridless panels.
Inputs: results/formation_medqa.json, probe_pilot_medqa.jsonl, soft_f3_medqa_ua.csv.
Usage: .venv/bin/python analysis/fig_formation.py --results results --out ../paper/figures/fig_formation
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pubfig import save_figure
from sklearn.metrics import roc_auc_score

WRONG_C, WRONG_BAND = "#C44E52", "#F5D0DA"
RIGHT_C, RIGHT_BAND = "#4C72B0", "#CCE1EC"
BAR_FILL = "#C3CBDF"
RAMP = ["#B9C4D6", "#7694C0", "#296BBC"]
TAN = "#D2B48C"
SPINE = "0.55"
LEFT_MAX_ROUND = 4    # later rounds have n too small for stable means
MID_MAX_ROUND = 5


def ua_scores(R: Path) -> dict:
    probe = {json.loads(l)["qid"]: json.loads(l) for l in open(R / "probe_pilot_medqa.jsonl")
             if "error" not in json.loads(l)}
    soft = {r["qid"]: float(r["f3_5_mean_bidir"]) for r in csv.DictReader(open(R / "soft_f3_medqa_ua.csv"))}
    y, keep, rse = [], [], []
    for q, r in probe.items():
        sq = f"medqa_{q}"
        if sq in soft:
            y.append(int(r["consensus_correct"]))
            keep.append(r["keep_rate"])
            rse.append(soft[sq])
    y, keep, rse = np.array(y), np.array(keep), np.array(rse)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    return {"Combined": roc_auc_score(y, z(keep) + z(rse)),
            "Keep rate": roc_auc_score(y, keep),
            "Rationale entropy": roc_auc_score(y, rse)}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=SPINE, labelcolor="0.15")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="../paper/figures/fig_formation")
    args = parser.parse_args()
    R = Path(args.results)
    form = json.load(open(R / "formation_medqa.json"))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.0, "axes.labelsize": 5.7,
        "xtick.labelsize": 5.0, "ytick.labelsize": 5.0, "axes.labelpad": 2.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, (ax1, ax2, spacer, ax3) = plt.subplots(
        1, 4, gridspec_kw={"wspace": 0.46, "width_ratios": [1, 1.1, 0.10, 1.0]})
    spacer.axis("off")

    # (left) agreement trajectory with shaded SE bands
    for key, color, band, marker in (
            ("wrong", WRONG_C, WRONG_BAND, "o"),
            ("correct", RIGHT_C, RIGHT_BAND, "s")):
        g = form[key]
        rounds = np.array([r for r in sorted(map(int, g["round_mean"])) if r <= LEFT_MAX_ROUND])
        means = np.array([g["round_mean"][str(r)] for r in rounds])
        ns = np.array([g["round_n"][str(r)] for r in rounds])
        se = np.sqrt(means * (1 - means) / ns)
        ax1.fill_between(rounds, means - se, means + se, color=band, alpha=0.75, lw=0, zorder=1)
        ax1.plot(rounds, means, color=color, marker=marker, ms=2.8, lw=1.2, zorder=3)
    ax1.set_title("Agreement over rounds", fontsize=7, pad=3)
    ax1.set_xlabel("Deliberation round")
    ax1.set_ylabel("Mean agreement rate")
    ax1.set_xticks(range(1, LEFT_MAX_ROUND + 1))
    ax1.set_ylim(0.76, 0.95)
    # SConU-style direct series labels, colored to match, no legend
    ax1.text(1.55, 0.935, "Correct consensus", fontsize=6.0, color=RIGHT_C)
    ax1.text(1.55, 0.782, "Wrong consensus", fontsize=6.0, color=WRONG_C)

    # (mid) questions concluding per round (bars) vs error rate at that round (line)
    cs = {int(k): v for k, v in form["correct"]["stop_counts"].items()}
    ws = {int(k): v for k, v in form["wrong"]["stop_counts"].items()}
    rounds = list(range(1, MID_MAX_ROUND + 1))
    totals = np.array([cs.get(r, 0) + ws.get(r, 0) for r in rounds])
    err = np.array([100 * ws.get(r, 0) / max(cs.get(r, 0) + ws.get(r, 0), 1) for r in rounds])
    overall = 100 * form["wrong"]["n"] / (form["wrong"]["n"] + form["correct"]["n"])
    ax2.bar(rounds, totals, color=BAR_FILL, edgecolor="white", lw=0.5, width=0.66, zorder=2)
    ax2.set_title("Outcome by stopping round", fontsize=7, pad=3)
    ax2.set_xlabel("Stopping round")
    ax2.set_ylabel("Questions concluded")
    ax2.set_xticks(rounds)
    ax2b = ax2.twinx()
    ax2b.plot(rounds, err, color=WRONG_C, marker="o", ms=2.8, lw=1.2, zorder=3)
    for r, e in zip(rounds, err):
        ax2b.annotate(f"{e:.0f}", (r, e), textcoords="offset points", xytext=(0, 4),
                      ha="center", fontsize=5.5, color=WRONG_C)
    ax2b.axhline(overall, color="0.35", lw=0.7, ls="--", zorder=1)
    ax2b.annotate("overall", (MID_MAX_ROUND + 0.42, overall), fontsize=5.3,
                  color="0.35", va="bottom", ha="right")
    ax2b.text(0.05, 0.93, "Error rate (%)", transform=ax2b.transAxes,
              fontsize=5.8, color=WRONG_C)
    ax2b.set_ylim(0, 78)
    ax2b.tick_params(axis="y", colors=WRONG_C, labelsize=5.0, pad=1.5)
    ax2b.spines[["top", "left"]].set_visible(False)
    ax2b.spines["right"].set_color(WRONG_C)
    ax2b.spines["bottom"].set_color(SPINE)

    # (right) unanimous-layer AUROC bars from the chance line
    scores = ua_scores(R)
    names = ["Vote-based", "Rationale entropy", "Keep rate", "Combined"]
    vals = [0.5, scores["Rationale entropy"], scores["Keep rate"], scores["Combined"]]
    colors = [TAN] + RAMP
    ypos = np.arange(len(names))
    ax3.barh(ypos, np.array(vals) - 0.5, left=0.5, color=colors,
             edgecolor="white", lw=0.5, height=0.62, zorder=2)
    for yi, v in zip(ypos, vals):
        ax3.text(v + 0.005, yi, "constant" if v == 0.5 else f"{v:.3f}",
                 va="center", ha="left", fontsize=6.0,
                 color="0.35" if v == 0.5 else "0.15")
    ax3.set_title("Unanimous layer", fontsize=7, pad=3)
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(names)
    ax3.axvline(0.5, color="0.2", lw=0.7, ls="--", zorder=3)
    ax3.set_xlim(0.5, 0.75)
    ax3.set_xticks([0.5, 0.6, 0.7])
    ax3.set_xlabel("AUROC")

    for ax in (ax1, ax2, ax3):
        style(ax)

    paths = save_figure(fig, args.out + ".pdf", spec="nature", width=140, aspect_ratio=0.20)
    print({k: round(v, 3) for k, v in scores.items()}, f"overall_err={overall:.1f}%")
    print("wrote:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
