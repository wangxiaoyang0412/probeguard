"""Aggregate per-question MA-RAG logs (question_{id}/round_{k}.jsonl + round_{k}_retrieval.json)
into the unified JSONL schema. The `correct` field replicates the base pipeline's
utils.calculate_accuracy strict-majority rule (last-round mode == gold with a strict
lead over the runner-up; ties count as wrong)."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterator, Optional


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _round_files(question_dir: Path):
    files = [p for p in question_dir.glob("round_*.jsonl")]
    return sorted(files, key=lambda p: int(p.stem.split("_")[1]))


def _load_retrieval(question_dir: Path, k: int) -> Optional[Dict[str, Any]]:
    path = question_dir / f"round_{k}_retrieval.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Schema keeps only queries/doc_ids; scores stay in the raw files for probe / family-4 use
    return {"queries": list(raw.get("queries", [])),
            "doc_ids": [str(d) for d in raw.get("doc_ids", [])]}


def convert_question_dir(question_dir: Path, dataset: str) -> Dict[str, Any]:
    question_dir = Path(question_dir)
    qid_num = question_dir.name.split("_")[-1]
    round_paths = _round_files(question_dir)
    if not round_paths:
        raise ValueError(f"no round files under {question_dir}")

    rounds = []
    gold = None
    for path in round_paths:
        k = int(path.stem.split("_")[1])
        entries = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").strip().split("\n")]
        gold = entries[0]["answer"]
        answers = [e["prediction"] for e in entries]
        counts = Counter(answers)
        majority, majority_n = counts.most_common(1)[0]
        candidates = [{
            "answer": e["prediction"],
            "rationale": e["response"],
            "text_hash": text_hash(e["response"]),
            "seq_logprob": float(fmean(e["logprobs"])) if e.get("logprobs") else None,
            "verifier_score": e.get("verifier_score"),  # always None for the intrinsic variant
        } for e in entries]
        rounds.append({
            "t": k,
            "candidates": candidates,
            "agreement_rate": majority_n / len(answers),
            "n_distinct_answers": len(counts),
            "majority_answer": majority,
            "retrieval": _load_retrieval(question_dir, k),
        })

    last_answers = [c["answer"] for c in rounds[-1]["candidates"]]
    most_common = Counter(last_answers).most_common(2)
    final_answer = most_common[0][0]
    # Base-pipeline calculate_accuracy rule: mode == gold with a strict majority (ties count as wrong)
    correct = (final_answer == gold and
               (len(most_common) == 1 or most_common[0][1] > most_common[1][1]))
    converged = len(set(last_answers)) == 1

    return {
        "qid": f"{dataset.lower()}_{qid_num}",
        "dataset": dataset,
        "gold": gold,
        "final_answer": final_answer,
        "correct": correct,
        "stopped_at_round": rounds[-1]["t"],
        "stop_reason": "converged" if converged else "max_rounds",
        "rounds": rounds,
        "probe": None,
    }


def convert_run(evaluations_dir: Path, dataset: str) -> Iterator[Dict[str, Any]]:
    """Walk the base pipeline's evaluations/ dir, yielding unified-schema records by ascending question id."""
    question_dirs = sorted(
        Path(evaluations_dir).glob("question_*"),
        key=lambda p: int(p.name.split("_")[-1]))
    for qdir in question_dirs:
        yield convert_question_dir(qdir, dataset=dataset)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from abstain_layer.schema import validate_line
    n_ok, n_bad = 0, 0
    with open(args.out, "w", encoding="utf-8") as fp:
        for rec in convert_run(Path(args.evaluations_dir), args.dataset):
            problems = validate_line(rec)
            if problems:
                n_bad += 1
                print(f"[INVALID] {rec['qid']}: {problems}")
            else:
                n_ok += 1
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {n_ok + n_bad} records ({n_ok} valid, {n_bad} invalid) -> {args.out}")


if __name__ == "__main__":
    main()
