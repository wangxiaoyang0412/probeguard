from itertools import permutations

from abstain_layer.rationale_features import extract_claims  # noqa: F401 (frozen dep)
from abstain_layer.soft_nli import probs_checksum, soft_family3


def uniform_probs(n, contra, neutral, entail):
    return {(i, j): (contra, neutral, entail) for i, j in permutations(range(n), 2)}


def test_definitions_on_uniform_matrix():
    # 8 rationales, pairwise entail prob 0.45 ("just short of entailment")
    f = soft_family3(8, uniform_probs(8, 0.05, 0.50, 0.45))
    assert abs(f["f3_5_mean_bidir"] - 0.45) < 1e-9
    assert abs(f["f3_6_mean_contradiction"] - 0.05) < 1e-9
    assert abs(f["f3_7_max_bidir"] - 0.45) < 1e-9


def test_plateau_breaking_regression():
    """Plateau-breaking regression test: two questions tied under hard clustering
    (both 8 clusters, hard entropy = ln 8) must be separable by the soft scores.
    Question A: pairwise entail=0.45 (near entailment, below the line); question B:
    pairwise entail=0.02 (genuinely unrelated rationales).
    Under the hard criterion (argmax; 0.45<0.50 is not entailment) both yield 8
    singleton clusters -> tied; under the soft one A's mean_bidir is far above B's."""
    fa = soft_family3(8, uniform_probs(8, 0.05, 0.50, 0.45))
    fb = soft_family3(8, uniform_probs(8, 0.60, 0.38, 0.02))
    assert fa["f3_5_mean_bidir"] > fb["f3_5_mean_bidir"] + 0.4
    assert fb["f3_6_mean_contradiction"] > fa["f3_6_mean_contradiction"] + 0.5


def test_min_symmetrization_and_max():
    # asymmetric pair: e_01=0.9, e_10=0.2 -> min=0.2; another pair 0.7/0.8 -> min=0.7 (the max_bidir)
    probs = {
        (0, 1): (0.0, 0.1, 0.9), (1, 0): (0.0, 0.8, 0.2),
        (0, 2): (0.0, 0.3, 0.7), (2, 0): (0.0, 0.2, 0.8),
        (1, 2): (0.5, 0.4, 0.1), (2, 1): (0.5, 0.4, 0.1),
    }
    f = soft_family3(3, probs)
    assert abs(f["f3_7_max_bidir"] - 0.7) < 1e-9
    assert abs(f["f3_5_mean_bidir"] - (0.2 + 0.7 + 0.1) / 3) < 1e-9


def test_checksum_deterministic():
    p = {"q1": uniform_probs(2, 0.1, 0.2, 0.7)}
    assert probs_checksum(p) == probs_checksum(
        {"q1": uniform_probs(2, 0.1, 0.2, 0.7)})
    assert probs_checksum(p) != probs_checksum(
        {"q1": uniform_probs(2, 0.1, 0.3, 0.6)})
