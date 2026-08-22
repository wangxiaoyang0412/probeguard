import hashlib
import json

from abstain_layer.convert_logs import convert_question_dir
from abstain_layer.schema import validate_line


def _mk_round(qdir, k, preds, gold="C", logprobs=(-1.0, -2.0)):
    lines = []
    for i, p in enumerate(preds, start=1):
        lines.append({
            "round": k, "answer_id": i, "answer": gold, "prediction": p,
            "is_correct": p == gold, "prompt": "P", "reasoning": "",
            "response": f"analysis {i} -> <answer>{p}</answer>",
            "logprobs": list(logprobs), "token_entropies": [0.1, 0.2],
        })
    (qdir / f"round_{k}.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def test_converged_correct(tmp_path):
    q = tmp_path / "question_7"
    q.mkdir()
    _mk_round(q, 1, ["A", "C", "C", "C"])
    (q / "round_1_retrieval.json").write_text(json.dumps(
        {"queries": ["q1"], "doc_ids": ["d1", "d2"], "scores": [1.2, 0.8]}))
    _mk_round(q, 2, ["C", "C", "C", "C"])

    rec = convert_question_dir(q, dataset="MedQA")

    assert rec["qid"] == "medqa_7"
    assert rec["gold"] == "C"
    assert rec["final_answer"] == "C"
    assert rec["correct"] is True
    assert rec["stopped_at_round"] == 2
    assert rec["stop_reason"] == "converged"
    assert rec["rounds"][0]["agreement_rate"] == 0.75
    assert rec["rounds"][0]["n_distinct_answers"] == 2
    assert rec["rounds"][0]["retrieval"] == {"queries": ["q1"], "doc_ids": ["d1", "d2"]}
    assert rec["rounds"][1]["retrieval"] is None
    cand = rec["rounds"][0]["candidates"][0]
    assert cand["answer"] == "A"
    assert cand["rationale"] == "analysis 1 -> <answer>A</answer>"
    assert cand["seq_logprob"] == -1.5
    assert cand["verifier_score"] is None
    expected_hash = hashlib.sha256(
        "analysis 1 -> <answer>A</answer>".encode("utf-8")).hexdigest()[:16]
    assert cand["text_hash"] == expected_hash
    assert rec["probe"] is None


def test_tie_counts_wrong_and_not_converged(tmp_path):
    q = tmp_path / "question_3"
    q.mkdir()
    _mk_round(q, 1, ["C", "C", "B", "B"])

    rec = convert_question_dir(q, dataset="MedQA")

    assert rec["correct"] is False          # replicates the base pipeline's strict-majority rule: ties count as wrong
    assert rec["stop_reason"] == "max_rounds"
    assert rec["stopped_at_round"] == 1


def test_plurality_wrong_answer(tmp_path):
    q = tmp_path / "question_5"
    q.mkdir()
    _mk_round(q, 1, ["B", "B", "B", "C"], gold="C")

    rec = convert_question_dir(q, dataset="MedXpertQA")

    assert rec["final_answer"] == "B"
    assert rec["correct"] is False


def test_output_passes_schema(tmp_path):
    q = tmp_path / "question_1"
    q.mkdir()
    _mk_round(q, 1, ["B", "B"])

    rec = convert_question_dir(q, dataset="MedXpertQA")

    assert validate_line(rec) == []
