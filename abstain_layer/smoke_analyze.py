"""Smoke-run acceptance analysis: schema validation, accuracy gate, process stats, spot-check printout."""
import argparse
import json
from collections import Counter
from pathlib import Path

from abstain_layer.schema import validate_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gate", type=float, default=0.62)
    args = parser.parse_args()

    recs = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    n = len(recs)
    invalid = {r["qid"]: validate_line(r) for r in recs}
    invalid = {k: v for k, v in invalid.items() if v}
    acc = sum(r["correct"] for r in recs) / n

    r1_unanimous = sum(1 for r in recs if r["rounds"][0]["agreement_rate"] == 1.0)
    stop = Counter(r["stop_reason"] for r in recs)
    rounds_hist = Counter(r["stopped_at_round"] for r in recs)
    with_retrieval = sum(1 for r in recs if any(rd["retrieval"] for rd in r["rounds"]))
    empty_rationales = sum(1 for r in recs for rd in r["rounds"]
                           for c in rd["candidates"] if not c["rationale"].strip())
    null_logprob = sum(1 for r in recs for rd in r["rounds"]
                       for c in rd["candidates"] if c["seq_logprob"] is None)
    # Blind-spot view: accuracy on the round-1 unanimous subset vs the disagreement subset
    ua = [r for r in recs if r["rounds"][0]["agreement_rate"] == 1.0]
    da = [r for r in recs if r["rounds"][0]["agreement_rate"] < 1.0]
    acc_ua = sum(r["correct"] for r in ua) / len(ua) if ua else float("nan")
    acc_da = sum(r["correct"] for r in da) / len(da) if da else float("nan")

    print(f"n={n}")
    print(f"schema_invalid={len(invalid)} {list(invalid.items())[:3]}")
    print(f"accuracy={acc:.4f} gate={args.gate} -> {'PASS' if acc >= args.gate else 'FAIL'}")
    print(f"r1_unanimous={r1_unanimous}/{n} ({r1_unanimous/n:.0%})  acc|unanimous={acc_ua:.3f}  acc|disagree={acc_da:.3f}")
    print(f"stop_reason={dict(stop)}")
    print(f"rounds_hist={dict(sorted(rounds_hist.items()))}")
    print(f"questions_with_retrieval={with_retrieval}")
    print(f"empty_rationales={empty_rationales}  null_seq_logprob={null_logprob}")

    print("--- spot check: 8 records ---")
    for r in recs[::max(1, n // 8)][:8]:
        rd0 = r["rounds"][0]
        print(f"{r['qid']} gold={r['gold']} final={r['final_answer']} correct={r['correct']} "
              f"stop={r['stopped_at_round']}/{r['stop_reason']} r1_agree={rd0['agreement_rate']:.2f} "
              f"rationale_len={len(rd0['candidates'][0]['rationale'])}")
    print("--- one full round-1 candidate rationale (first record, truncated 400 chars) ---")
    print(recs[0]["rounds"][0]["candidates"][0]["rationale"][:400])

    print("--- manifest ---")
    print(Path(args.manifest).read_text())


if __name__ == "__main__":
    main()
