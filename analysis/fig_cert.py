"""fig_cert: certification validity and power across the alpha range (SConU-form).
(Left) realized selective risk vs target alpha with the y=x bound diagonal: LTT arms stay
below the diagonal everywhere they answer; naive quantile calibration crosses it.
(Right) certified coverage vs alpha for the LTT arms (naive coverage is not comparable
where its bound fails and is omitted). R=50 stratified splits per point, protocol of
analysis/c8_fuel_cert.py; pre-registered claims remain the four ladder alphas.
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
from analysis.c8_fuel_cert import load_layer  # noqa: E402

R_SPLITS = 50
ALPHAS = np.round(np.arange(0.04, 0.221, 0.005), 3)
SPINE = "0.55"
DIAG_C = "0.2"


def plug_in_threshold(scores: np.ndarray, y: np.ndarray, alpha: float) -> float:
    order = np.argsort(-scores, kind="stable")
    err = 1 - y[order]
    cum = np.cumsum(err) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= alpha)[0]
    return np.inf if len(ok) == 0 else scores[order][ok[-1]]


def curve(y: np.ndarray, s: np.ndarray, extra=None, naive: bool = False) -> tuple:
    idx = np.arange(len(y))
    cov_m, risk_m, cov_se, risk_se = [], [], [], []
    for alpha in ALPHAS:
        covs, risks = [], []
        for seed in range(R_SPLITS):
            cal, te = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
            cs, cy = s[cal], y[cal]
            if extra is not None:
                cs = np.concatenate([cs, extra[1]])
                cy = np.concatenate([cy, extra[0]])
            thr = plug_in_threshold(cs, cy, alpha) if naive \
                else ltt_threshold(cs, cy, alpha, delta=0.025)[0]
            cov, risk = evaluate(s[te], y[te], thr)
            covs.append(cov)
            if cov > 0:
                risks.append(risk)
        cov_m.append(np.mean(covs))
        cov_se.append(np.std(covs))
        risk_m.append(np.mean(risks) if risks else np.nan)
        risk_se.append(np.std(risks) if risks else np.nan)
    return np.array(cov_m), np.array(risk_m), np.array(cov_se), np.array(risk_se)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="../paper/figures/fig_cert")
    args = parser.parse_args()
    R = args.results
    y_q, s_q = load_layer(f"{R}/probe_pilot_medqa.jsonl", f"{R}/soft_f3_medqa_ua.csv", "medqa")
    y_f, s_f = load_layer(f"{R}/probe_fuel_gbaker.jsonl", f"{R}/soft_f3_fuel_ua.csv", "medqa")
    y_h, s_h = load_layer(f"{R}/probe_medqa_ua_huatuo.jsonl", f"{R}/soft_f3_medqa_huatuo_ua.csv", "medqa")
    y_l, s_l = load_layer(f"{R}/probe_medqa_ua_llama.jsonl", f"{R}/soft_f3_medqa_llama_ua.csv", "medqa")

    arms = [
        ("Qwen3-8B", "#4C72B0", "-", curve(y_q, s_q)),
        ("Qwen3-8B + in-domain cal.", "#0B559F", "-", curve(y_q, s_q, extra=(y_f, s_f))),
        ("HuatuoGPT-o1-8B", "#008B8B", "-", curve(y_h, s_h)),
        ("Llama-3.1-8B", "#CD853F", "-", curve(y_l, s_l)),
    ]
    naive = ("Naive quantile", "#C44E52", "--", curve(y_q, s_q, naive=True))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.0, "axes.labelsize": 5.7,
        "xtick.labelsize": 5.0, "ytick.labelsize": 5.0, "axes.labelpad": 2.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"wspace": 0.30})

    # (left) validity: realized risk vs target alpha
    ax1.plot([0.04, 0.22], [0.04, 0.22], ls="--", lw=0.9, color=DIAG_C, zorder=2)
    for label, color, ls, (cov, risk, cse, rse) in arms + [naive]:
        ax1.fill_between(ALPHAS, risk - rse, risk + rse, color=color, alpha=0.22, lw=0, zorder=2)
        ax1.plot(ALPHAS, risk, color=color, ls=ls, lw=1.1, label=label, zorder=3)
    ax1.set_ylim(0.03, 0.148)
    ax1.text(0.1535, 0.1455, r"risk $=\alpha$", fontsize=5.2, color=DIAG_C,
             ha="left", va="top")
    ax1.set_xlabel(r"Target risk level $\alpha$")
    ax1.set_ylabel("Realized selective risk")
    ax1.set_xlim(0.04, 0.22)
    ax1.legend(frameon=False, fontsize=4.3, loc="upper left", handlelength=1.4, labelspacing=0.35)

    # (right) power: certified coverage vs alpha (LTT arms)
    i10, i15 = list(ALPHAS).index(0.10), list(ALPHAS).index(0.15)
    for label, color, ls, (cov, risk, cse, rse) in arms:
        ax2.fill_between(ALPHAS, cov - cse, cov + cse, color=color, alpha=0.22, lw=0, zorder=2)
        ax2.plot(ALPHAS, cov, color=color, ls=ls, lw=1.1, label=label, zorder=3)
    for a in (0.10, 0.15):
        ax2.axvline(a, color="0.85", lw=0.6, zorder=1)
    # series-colored value callouts at alpha=0.15, white-stroked for readability
    from matplotlib import patheffects
    spots = {"Qwen3-8B": (0.157, 0.50, "left"), "Qwen3-8B + in-domain cal.": (0.138, 0.945, "right"),
             "HuatuoGPT-o1-8B": (0.153, 0.952, "left"), "Llama-3.1-8B": (0.153, 0.812, "left")}
    for label, color, ls, (cov, risk, cse, rse) in arms:
        x0, y0, ha = spots[label]
        ax2.text(x0, y0, f"{cov[i15]:.3f}", fontsize=5.2, color=color, ha=ha, zorder=6,
                 path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")])
    # dark-red stars on the two pre-registered fuel-arm operating points
    fuel_cov = arms[1][3][0]
    ax2.scatter([0.10, 0.15], [fuel_cov[i10], fuel_cov[i15]], marker="*", s=34,
                color="#8B0000", zorder=5)
    ax2.legend(frameon=False, fontsize=4.3, loc="lower right", handlelength=1.4, labelspacing=0.35)
    ax2.set_xlabel(r"Target risk level $\alpha$")
    ax2.set_ylabel("Certified coverage")
    ax2.set_xlim(0.04, 0.22)
    ax2.set_ylim(0, 1.02)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
        ax.tick_params(colors=SPINE, labelcolor="0.15")

    paths = save_figure(fig, args.out + ".pdf", spec="nature", width=140, aspect_ratio=0.32)
    for label, _, _, (cov, risk, cse, rse) in arms + [naive]:
        print(f"{label}: cov@.10={cov[i10]:.3f} cov@.15={cov[i15]:.3f} risk@.10={risk[i10]:.3f}")
    print("wrote:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
