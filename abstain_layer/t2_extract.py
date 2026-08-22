"""Batch feature extraction: converted JSONL -> one feature row per question (CSV).

Usage (server, after the full run completes):
  python -m abstain_layer.t2_extract --jsonl runs/.../full_medqa.jsonl \
      --out results/features_medqa.csv [--nli-device cuda:0]
Without --nli-device the family-3 NLI features are skipped (all others run on CPU in seconds).
"""
import argparse
import csv
import json
from typing import List, Tuple

from abstain_layer.features import extract_features
from abstain_layer.rationale_features import family3_nli


class DebertaNLIJudge:
    """Batched DeBERTa-large-MNLI judge (pre-specified NLI checkpoint)."""

    def __init__(self, device: str = "cuda:0", batch_size: int = 8, fp16: bool = False):
        # fp16=False: deberta's disentangled attention has float32 ops; naive half precision hits dtype conflicts
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        name = "microsoft/deberta-large-mnli"
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        # deberta-large-mnli label order: 0=contradiction 1=neutral 2=entailment
        self.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __call__(self, pairs: List[Tuple[str, str]]) -> List[str]:
        labels: List[str] = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                enc = self.tok([a for a, _ in batch], [b for _, b in batch],
                               truncation=True, max_length=256, padding=True,
                               return_tensors="pt").to(self.device)
                pred = self.model(**enc).logits.argmax(dim=-1).tolist()
                labels += [self.id2label[p] for p in pred]
        return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nli-device", default=None)
    parser.add_argument("--nli-batch", type=int, default=8)
    parser.add_argument("--t-max", type=int, default=8)
    args = parser.parse_args()

    judge = DebertaNLIJudge(args.nli_device, args.nli_batch) if args.nli_device else None
    rows = []
    for line in open(args.jsonl, encoding="utf-8"):
        rec = json.loads(line)
        row = {"qid": rec["qid"], "dataset": rec["dataset"],
               "correct": int(rec["correct"])}
        row.update(extract_features(rec, t_max=args.t_max))
        if judge is not None:
            row.update(family3_nli(rec, judge))
        rows.append(row)

    keys = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows x {len(keys)} cols -> {args.out}")


if __name__ == "__main__":
    main()
