"""Unit tests for ptrue_conf scoring functions (pure functions, no network)."""
from abstain_layer.ptrue_conf import score_ptrue, truefalse_mass
import math


def _pos(tokens_logprobs):
    return {"top_logprobs": [{"token": t, "logprob": lp} for t, lp in tokens_logprobs]}


def test_truefalse_mass_variants():
    p_t, p_f = truefalse_mass([{"token": "True", "logprob": math.log(0.7)},
                               {"token": " false", "logprob": math.log(0.2)},
                               {"token": "Maybe", "logprob": math.log(0.1)}])
    assert abs(p_t - 0.7) < 1e-9 and abs(p_f - 0.2) < 1e-9


def test_score_ptrue_normalizes():
    ptrue, p_t, p_f = score_ptrue([_pos([("True", math.log(0.6)), ("False", math.log(0.3))])])
    assert abs(ptrue - 0.6 / 0.9) < 1e-9


def test_score_ptrue_skips_empty_position():
    # scan forward when the first token is whitespace / carries no True/False mass
    ptrue, _, _ = score_ptrue([_pos([("\n", math.log(0.9))]),
                               _pos([("False", math.log(0.8))])])
    assert ptrue == 0.0


def test_score_ptrue_none_when_absent():
    ptrue, p_t, p_f = score_ptrue([_pos([("Yes", math.log(0.9))])] * 4)
    assert ptrue is None and p_t == 0.0 and p_f == 0.0
