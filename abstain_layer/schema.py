"""Unified JSONL schema validation (field names are frozen; do not rename)."""
from collections import Counter
from typing import Any, Dict, List

import jsonschema

# 1.0.0 = original schema; 1.1.0 = two approved revisions (full candidates.rationale text + top-level probe block)
SCHEMA_VERSION = "1.1.0"

_CANDIDATE_SCHEMA = {
    "type": "object",
    "required": ["answer", "rationale", "text_hash", "seq_logprob", "verifier_score"],
    "properties": {
        "answer": {"type": "string"},
        "rationale": {"type": "string"},
        "text_hash": {"type": "string", "minLength": 8},
        "seq_logprob": {"type": ["number", "null"]},
        "verifier_score": {"type": ["number", "null"]},
    },
}

_ROUND_SCHEMA = {
    "type": "object",
    "required": ["t", "candidates", "agreement_rate", "n_distinct_answers",
                 "majority_answer", "retrieval"],
    "properties": {
        "t": {"type": "integer", "minimum": 1},
        "candidates": {"type": "array", "minItems": 1, "items": _CANDIDATE_SCHEMA},
        "agreement_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "n_distinct_answers": {"type": "integer", "minimum": 1},
        "majority_answer": {"type": "string"},
        "retrieval": {
            "type": ["object", "null"],
            "required": ["queries", "doc_ids"],
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
                "doc_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

_PROBE_SCHEMA = {
    "type": ["object", "null"],
    "required": ["triggered", "counter_queries", "counter_docs", "survival_trajectory"],
    "properties": {
        "triggered": {"type": "boolean"},
        "counter_queries": {"type": "array", "items": {"type": "string"}},
        "counter_docs": {"type": "array", "items": {"type": "string"}},
        "survival_trajectory": {"type": "array", "items": _ROUND_SCHEMA},
    },
}

QUESTION_LOG_SCHEMA = {
    "type": "object",
    "required": ["qid", "dataset", "gold", "final_answer", "correct",
                 "stopped_at_round", "stop_reason", "rounds", "probe"],
    "properties": {
        "qid": {"type": "string"},
        "dataset": {"type": "string"},
        "gold": {"type": "string"},
        "final_answer": {"type": "string"},
        "correct": {"type": "boolean"},
        "stopped_at_round": {"type": "integer", "minimum": 1},
        "stop_reason": {"type": "string", "enum": ["converged", "max_rounds"]},
        "rounds": {"type": "array", "minItems": 1, "items": _ROUND_SCHEMA},
        "probe": _PROBE_SCHEMA,
    },
}

_TOL = 1e-6


def validate_line(obj: Dict[str, Any]) -> List[str]:
    """Return a list of violation messages; empty = valid. Structural + semantic-consistency checks."""
    validator = jsonschema.Draft7Validator(QUESTION_LOG_SCHEMA)
    errors = [f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
              for e in validator.iter_errors(obj)]
    if errors:
        return errors

    for r in obj["rounds"]:
        answers = [c["answer"] for c in r["candidates"]]
        counts = Counter(answers)
        majority, majority_n = counts.most_common(1)[0]
        expected_rate = majority_n / len(answers)
        if abs(r["agreement_rate"] - expected_rate) > _TOL:
            errors.append(
                f"round {r['t']}: agreement_rate={r['agreement_rate']} "
                f"but candidates imply {expected_rate}")
        if r["n_distinct_answers"] != len(counts):
            errors.append(
                f"round {r['t']}: n_distinct_answers={r['n_distinct_answers']} "
                f"but candidates imply {len(counts)}")
        if counts[r["majority_answer"]] != majority_n:
            errors.append(
                f"round {r['t']}: majority_answer={r['majority_answer']} "
                f"is not a modal answer of candidates")
    return errors
