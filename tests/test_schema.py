from abstain_layer.schema import validate_line


def make_valid_line():
    return {
        "qid": "medqa_1", "dataset": "MedQA", "gold": "C",
        "final_answer": "C", "correct": True,
        "stopped_at_round": 2, "stop_reason": "converged",
        "rounds": [
            {
                "t": 1,
                "candidates": [
                    {"answer": "C", "rationale": "Because of X. <answer>C</answer>",
                     "text_hash": "ab12cd34ef56ab12", "seq_logprob": -3.2,
                     "verifier_score": None},
                    {"answer": "B", "rationale": "Because of Y. <answer>B</answer>",
                     "text_hash": "ab12cd34ef56ab13", "seq_logprob": -4.0,
                     "verifier_score": None},
                ],
                "agreement_rate": 0.5, "n_distinct_answers": 2,
                "majority_answer": "C",
                "retrieval": {"queries": ["q1"], "doc_ids": ["d1"]},
            },
            {
                "t": 2,
                "candidates": [
                    {"answer": "C", "rationale": "Given doc d1. <answer>C</answer>",
                     "text_hash": "ab12cd34ef56ab14", "seq_logprob": -2.0,
                     "verifier_score": None},
                    {"answer": "C", "rationale": "Doc d1 says so. <answer>C</answer>",
                     "text_hash": "ab12cd34ef56ab15", "seq_logprob": -2.5,
                     "verifier_score": None},
                ],
                "agreement_rate": 1.0, "n_distinct_answers": 1,
                "majority_answer": "C",
                "retrieval": None,
            },
        ],
        "probe": None,
    }


def test_valid_line_passes():
    assert validate_line(make_valid_line()) == []


def test_missing_rationale_fails():
    line = make_valid_line()
    del line["rounds"][0]["candidates"][0]["rationale"]
    assert any("rationale" in e for e in validate_line(line))


def test_agreement_rate_must_match_candidates():
    line = make_valid_line()
    line["rounds"][1]["agreement_rate"] = 0.5
    assert any("agreement_rate" in e for e in validate_line(line))


def test_bad_stop_reason_fails():
    line = make_valid_line()
    line["stop_reason"] = "gave_up"
    assert any("stop_reason" in e for e in validate_line(line))


def test_n_distinct_mismatch_fails():
    line = make_valid_line()
    line["rounds"][0]["n_distinct_answers"] = 3
    assert any("n_distinct" in e for e in validate_line(line))
