from abstain_layer.rationale_features import extract_claims, family3_nli

from tests.test_rationale_features import _rec, _round


def test_extract_claims_strips_tag_and_picks_cue_sentences():
    rationale = (
        "The patient presents with chest pain. ECG shows ST elevation. "
        "Troponin is elevated. This rules out stable angina. "
        "Therefore the most likely diagnosis is myocardial infarction. "
        "The correct answer is B. <answer>B</answer>")
    claims = extract_claims(rationale)
    assert "<answer>" not in claims
    assert "Therefore the most likely diagnosis" in claims
    assert "correct answer is B" in claims
    assert "chest pain" not in claims          # non-conclusion sentences are excluded


def test_extract_claims_fallback_last_sentence():
    claims = extract_claims("Step one. Step two. Step three final conclusion here.")
    assert claims == "Step three final conclusion here."


def make_fake_judge(contradict_pairs=()):
    """Fake NLI: identical claims -> bidirectional entailment; pairs in contradict_pairs -> contradiction; otherwise neutral."""
    def judge(pairs):
        labels = []
        for a, b in pairs:
            if a == b:
                labels.append("entailment")
            elif (a, b) in contradict_pairs or (b, a) in contradict_pairs:
                labels.append("contradiction")
            else:
                labels.append("neutral")
        return labels
    return judge


def test_true_consensus_one_cluster_both_criteria():
    r = "Therefore the answer is C because the mechanism is X."
    rec = _rec([_round(["C"] * 4, [r] * 4)])
    f = family3_nli(rec, make_fake_judge())
    assert f["f3_1_bidir_cluster_count"] == 1.0
    assert f["f3_2_bidir_entropy"] == 0.0
    assert f["f3_1_noncon_cluster_count"] == 1.0


def test_false_consensus_split_under_bidirectional_merged_under_noncon():
    r1 = "Therefore the answer is C via mechanism X."
    r2 = "Therefore the answer is C via pathway Y."
    rec = _rec([_round(["C"] * 4, [r1, r1, r2, r2])])
    f = family3_nli(rec, make_fake_judge())
    assert f["f3_1_bidir_cluster_count"] == 2.0     # primary criterion: two clusters (no mutual entailment)
    assert f["f3_2_bidir_entropy"] > 0.6
    assert f["f3_1_noncon_cluster_count"] == 1.0    # backup criterion: no contradiction -> merged (lenient)


def test_contradictory_claims_split_under_both():
    r1 = "Therefore the answer is C because X inhibits Y."
    r2 = "Therefore the answer is C because X activates Y."
    rec = _rec([_round(["C"] * 2, [r1, r2])])
    f = family3_nli(rec, make_fake_judge(contradict_pairs=(
        (extract_claims(r1), extract_claims(r2)),)))
    assert f["f3_1_noncon_cluster_count"] == 2.0    # contradiction -> backup criterion splits too
