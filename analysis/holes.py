"""Supplementary robustness checks (h2-h5, CPU-only). Usage: python analysis/holes.py h2|h3|h4|h5"""
import csv
import json
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, ".")
from abstain_layer.phase1_conformal import evaluate, ltt_threshold  # noqa: E402
from abstain_layer.t3_evaluate import PROCESS_PREFIXES, boot_ci_auroc, cv_scores, load  # noqa: E402

PAIR = {"medqa": "runs/medqa/qwen3-8b/exp_t1_full/full_medqa.jsonl",
        "mmlupro": "runs/mmlu-pro/qwen3-8b/exp_t1_full/full_mmlupro.jsonl"}
HEDGE = re.compile(r"\b(however|but|although|unsure|possibly|may|might|unclear|uncertain)\b", re.I)


def _cv_auc(X, y, seed=42):
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        m = LogisticRegression(max_iter=2000).fit((X[tr]-mu)/sd, y[tr])
        oof[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, oof), oof


def h2():
    """h2: shallow behavioral baselines (Mind-the-Gap tier), zero LLM cost."""
    for ds, path in PAIR.items():
        recs = [json.loads(l) for l in open(path)]
        y = np.array([int(r["correct"]) for r in recs])
        feats, names = [], ["total_chars", "mean_rat_len", "mean_sents", "hedge_per_rat", "r1_len_std"]
        for r in recs:
            rats = [c["rationale"] for rd in r["rounds"] for c in rd["candidates"]]
            r1 = [len(c["rationale"]) for c in r["rounds"][0]["candidates"]]
            feats.append([
                sum(len(x) for x in rats),
                np.mean([len(x) for x in rats]),
                np.mean([x.count(".") for x in rats]),
                np.mean([len(HEDGE.findall(x)) for x in rats]),
                np.std(r1),
            ])
        X = np.array(feats)
        print(f"== h2 {ds} (n={len(y)}) ==")
        for j, nm in enumerate(names):
            print(f"  {nm:14s} AUROC={roc_auc_score(y, X[:, j]):.4f}")
        auc, _ = _cv_auc(X, y)
        lo, hi = boot_ci_auroc(y, np.zeros(0) if False else _cv_auc(X, y)[1])
        print(f"  COMBINED(5f-CV) AUROC={auc:.4f} CI=[{lo:.4f},{hi:.4f}]")


def _s1_scores():
    probe = {json.loads(l)["qid"]: json.loads(l) for l in open("results/probe_pilot_medqa.jsonl")
             if "error" not in json.loads(l)}
    soft = {r["qid"]: float(r["f3_5_mean_bidir"]) for r in csv.DictReader(open("results/soft_f3_medqa_ua.csv"))}
    y, s = [], []
    for q, r in probe.items():
        sq = f"medqa_{q}"
        if sq in soft:
            y.append(int(r["consensus_correct"]))
            s.append((r["keep_rate"], soft[sq]))
    y = np.array(y)
    s = np.array(s)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    return z(s[:, 0]) + z(s[:, 1]), y


def h3(R=50):
    """h3: repeated-split certification analysis for the S1 layer (R random 50/50 splits), mean±SE."""
    comb, y = _s1_scores()
    idx = np.arange(len(y))
    print(f"== h3 repeated splits R={R} on MedQA S1 (n={len(y)}) ==")
    for alpha in (0.10, 0.15, 0.20):
        covs, risks = [], []
        for seed in range(R):
            cal, te = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
            thr, _ = ltt_threshold(comb[cal], y[cal], alpha, delta=0.025)
            cov, risk = evaluate(comb[te], y[te], thr)
            covs.append(cov)
            risks.append(risk if cov > 0 else np.nan)
        covs = np.array(covs)
        risks = np.array(risks)
        n_pos = (covs > 0).sum()
        print(f"  α={alpha:.2f}: coverage={covs.mean():.3f}±{covs.std()/np.sqrt(R):.3f} "
              f"(nonzero in {n_pos}/{R}) risk|answered={np.nanmean(risks):.4f}±{np.nanstd(risks)/max(np.sqrt(n_pos),1):.4f}")


def h4():
    """h4: difficulty-proxy correlation matrix (Spearman). Proxies: stem length, option count, retrieval rounds."""
    from scipy.stats import spearmanr
    for ds, path in PAIR.items():
        X, y, names, raw = load(f"results/features_{'medqa' if ds=='medqa' else 'mmlupro'}.csv")
        recs = {json.loads(l)["qid"]: json.loads(l) for l in open(path)}
        qlen, ret = [], []
        for r in raw:
            rec = recs[r["qid"]]
            qlen.append(len(rec["rounds"][0]["candidates"][0]["rationale"]) * 0 +
                        len(json.dumps(rec["rounds"][0]["candidates"][0].get("prompt", ""))) * 0 +
                        0)
        # stem length needs the vendor json; use retrieval-round count and stop round as proxies here
        proxies = {"n_ret_rounds": [sum(1 for rd in recs[r["qid"]]["rounds"] if rd["retrieval"]) for r in raw],
                   "t_stop": [recs[r["qid"]]["stopped_at_round"] for r in raw]}
        core = ["f0_1_r1_agreement", "f0_5_agreement_auc", "f3_2_bidir_entropy",
                "f2_1_minority_persistence", "f4_2_doc_repeat_mean", "bl_terminal_agreement"]
        print(f"== h4 {ds}: Spearman(feature, proxy) ==")
        header = "  " + " ".join(f"{p:>13s}" for p in proxies)
        print(f"{'feature':32s}{header}")
        for f in core:
            col = X[:, names.index(f)]
            row = []
            for p, v in proxies.items():
                m = ~np.isnan(col)
                rho = spearmanr(col[m], np.array(v)[m]).statistic
                row.append(f"{rho:13.3f}")
            print(f"{f:32s}  " + " ".join(row))


def h5():
    """h5: smoke-50 sensitivity — MedQA combined AUROC with/without the first 50 questions."""
    X, y, names, raw = load("results/features_medqa.csv")
    proc = [j for j, n in enumerate(names) if n.startswith(PROCESS_PREFIXES)]
    keep_mask = np.array([int(r["qid"].split("_")[-1]) >= 50 for r in raw])
    for label, mask in (("with smoke-50", np.ones(len(y), bool)), ("without smoke-50", keep_mask)):
        oof = cv_scores(X[mask], y[mask], proc, "logreg")
        auc = roc_auc_score(y[mask], oof)
        r1 = X[mask, names.index("f0_1_r1_agreement")]
        print(f"== h5 {label}: n={mask.sum()} combined={auc:.4f} r1={roc_auc_score(y[mask], r1):.4f}")


if __name__ == "__main__":
    {"h2": h2, "h3": h3, "h4": h4, "h5": h5}[sys.argv[1]]()
