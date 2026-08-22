"""Soft-aggregation extension of family 3: pairwise continuous NLI probabilities -> f3_5/f3_6/f3_7.

soft_family3 is a pure function: it takes a directed-pair probability dict
{(i,j): (p_contra, p_neutral, p_entail)} and returns three features with frozen definitions.
Probability production (SoftNLIProbs) is separated from consumption for easy test injection.
"""
import hashlib
import json
from itertools import permutations
from typing import Dict, List, Tuple

Probs = Dict[Tuple[int, int], Tuple[float, float, float]]  # (i,j) -> (contra, neutral, entail)


def soft_family3(n: int, probs: Probs) -> Dict[str, float]:
    """Definitions frozen before evaluation data were inspected. n = number of claims; probs holds all n*(n-1) directed pairs."""
    if n < 2:
        return {"f3_5_mean_bidir": 1.0, "f3_6_mean_contradiction": 0.0,
                "f3_7_max_bidir": 1.0}
    bidir = []
    for i in range(n):
        for j in range(i + 1, n):
            e_ij = probs[(i, j)][2]
            e_ji = probs[(j, i)][2]
            bidir.append(min(e_ij, e_ji))
    contra = [probs[p][0] for p in permutations(range(n), 2)]
    return {
        "f3_5_mean_bidir": sum(bidir) / len(bidir),
        "f3_6_mean_contradiction": sum(contra) / len(contra),
        "f3_7_max_bidir": max(bidir),
    }


class SoftNLIProbs:
    """Same DeBERTa-large-MNLI checkpoint, returning the full softmax (no binarization).
    Label order: 0=contradiction 1=neutral 2=entailment (matches DebertaNLIJudge)."""

    def __init__(self, device: str = "cuda:0", batch_size: int = 8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        name = "microsoft/deberta-large-mnli"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    def pair_probs(self, claims: List[str]) -> Probs:
        pairs = list(permutations(range(len(claims)), 2))
        texts = [(claims[i], claims[j]) for i, j in pairs]
        out: Probs = {}
        with self.torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                batch = texts[s:s + self.batch_size]
                enc = self.tok([a for a, _ in batch], [b for _, b in batch],
                               truncation=True, max_length=256, padding=True,
                               return_tensors="pt").to(self.device)
                pr = self.torch.softmax(self.model(**enc).logits, dim=-1).tolist()
                for k, p in enumerate(pr):
                    i, j = pairs[s + k]
                    out[(i, j)] = (float(p[0]), float(p[1]), float(p[2]))
        return out


def probs_checksum(all_probs: Dict[str, Probs]) -> str:
    payload = json.dumps(
        {q: {f"{i},{j}": [round(x, 6) for x in v] for (i, j), v in sorted(p.items())}
         for q, p in sorted(all_probs.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    import argparse
    import csv

    import numpy as np

    from abstain_layer.rationale_features import extract_claims

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    judge = SoftNLIProbs(args.device, args.batch)
    rows = []
    store = {}
    n_done = 0
    for line in open(args.jsonl, encoding="utf-8"):
        rec = json.loads(line)
        if rec["rounds"][0]["agreement_rate"] != 1.0:
            continue  # scope: unanimous layer only
        claims = [extract_claims(c["rationale"])
                  for c in rec["rounds"][-1]["candidates"]]
        probs = judge.pair_probs(claims)
        feats = soft_family3(len(claims), probs)
        rows.append({"qid": rec["qid"], "correct": int(rec["correct"]), **feats})
        store[rec["qid"]] = probs
        n_done += 1
        if n_done % 100 == 0:
            print(f"processed {n_done}")

    np.savez_compressed(args.out_npz, payload=json.dumps(
        {q: {f"{i},{j}": v for (i, j), v in p.items()} for q, p in store.items()}))
    checksum = probs_checksum(store)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} unanimous-layer rows -> {args.out_csv}")
    print(f"probs npz -> {args.out_npz}  sha256[:16]={checksum}")


if __name__ == "__main__":
    main()
