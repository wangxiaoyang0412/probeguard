"""Pre-registered adjudication of the probe control arms (plan/review_triage.md D2).

Compares the counter-evidence probe against three control arms on the same MedQA S1
questions: nodocs (instruction only), randdocs (instruction + irrelevant documents),
neutral (same documents, neutral instruction). The bet is one-directional: the flip
asymmetry Delta = flip(wrong) - flip(right) of the counter-evidence arm exceeds both
content-free arms (nodocs, randdocs) with a paired-bootstrap 95% CI excluding zero.

Verdicts (fixed before the control data were inspected):
  A  both content-free arms significantly below the counter-evidence arm -> paper
  B  direction right, any CI crosses zero -> internal record only
  C  any content-free arm's Delta at or above the counter-evidence arm -> wording plan
Usage: .venv/bin/python analysis/probe_ctrl_eval.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

B = 2000
SEED = 42
ARMS = {
    "counter": "results/probe_pilot_medqa.jsonl",
    "nodocs": "results/probe_ctrl_nodocs.jsonl",
    "randdocs": "results/probe_ctrl_randdocs.jsonl",
    "neutral": "results/probe_ctrl_neutral.jsonl",
}


def load_arm(path: str) -> dict:
    recs = {}
    if not Path(path).exists():
        return recs
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            recs[r["qid"]] = r
    return recs


def delta(flip: np.ndarray, correct: np.ndarray) -> float:
    return flip[correct == 0].mean() - flip[correct == 1].mean()


def main() -> None:
    arms = {name: load_arm(p) for name, p in ARMS.items()}
    for name, recs in arms.items():
        print(f"{name}: n={len(recs)}")
    arms = {n: r for n, r in arms.items() if r}
    missing = [n for n in ARMS if n not in arms]
    qids = sorted(set.intersection(*(set(r) for r in arms.values())))
    if missing or len(qids) < len(arms["counter"]):
        print(f"WARNING: partial data (missing arms: {missing or 'none'}, "
              f"intersection {len(qids)}); final verdict requires complete arms")
    correct = np.array([int(arms["counter"][q]["consensus_correct"]) for q in qids])
    flip = {n: np.array([int(arms[n][q]["flipped"]) for q in qids]) for n in arms}
    keep = {n: np.array([arms[n][q]["keep_rate"] for q in qids]) for n in arms}

    print(f"\naligned n={len(qids)} (wrong={int((correct == 0).sum())})")
    for n in arms:
        fw = flip[n][correct == 0].mean()
        fr = flip[n][correct == 1].mean()
        auc = roc_auc_score(correct, keep[n])
        print(f"  {n:9s} flip(wrong)={100*fw:5.1f}%  flip(right)={100*fr:4.1f}%  "
              f"Delta={100*(fw-fr):5.1f}pp  keepAUROC={auc:.3f}")

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(qids))
    verdict = "A"
    for arm in ("nodocs", "randdocs"):
        if arm not in arms:
            print(f"  counter - {arm}: arm missing, no verdict yet")
            verdict = "incomplete"
            continue
        point = delta(flip["counter"], correct) - delta(flip[arm], correct)
        diffs = []
        for _ in range(B):
            s = rng.choice(idx, len(idx), replace=True)
            diffs.append(delta(flip["counter"][s], correct[s])
                         - delta(flip[arm][s], correct[s]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  counter - {arm}: Delta diff={100*point:5.1f}pp  "
              f"CI=[{100*lo:5.1f}, {100*hi:5.1f}]")
        if point <= 0:
            verdict = "C"
        elif lo <= 0 and verdict != "C":
            verdict = "B"
    print(f"\nVERDICT: {verdict}  "
          "(A=paper, B=internal only, C=wording downgrade plan)")


if __name__ == "__main__":
    main()
