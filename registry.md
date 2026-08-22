# Frozen feature registry

All definitions below were fixed before the corresponding evaluation data were inspected
(versioned registry; v1 defined families 0–4, the probe protocol, and the Mondrian
stratification; v1.1 froze the NLI configuration and claim-extraction rule; v1.2 added the
soft rationale aggregates, frozen before computation). The stratification and probe trigger
predate any calibration data.

Notation: s_t(a) is the share of candidates voting option a at round t; m_t the modal
answer; T_stop the stopping round; D_t the documents retrieved after round t.

## Family 0 — agreement dynamics (10 features)
first-round agreement s_1(m_1); convergence round; majority flips; agreement slope;
trajectory-averaged agreement (area); post-retrieval agreement delta (mean, last);
oscillation index; consistency-bonus flip; consistency-bonus final match.

## Family 1 — answer-distribution dynamics (8 features)
entropy of the candidate answer distribution (first, last, slope, monotone fraction);
Jensen–Shannon divergence between adjacent rounds (mean, last, max); JS drift to the
final distribution (slope).

## Family 2 — minority dynamics (4 features)
minority persistence (max over non-final options of the fraction of rounds with at least
one supporter); maximum dissent run length; minority turnover; runner-up share slope.

## Family 3 — rationale signals (5 hard + 3 soft)
Concluding claims are extracted by the frozen deterministic rule (paper Appendix C), then
scored pairwise by DeBERTa-large-MNLI. Hard: bidirectional-entailment cluster count and
entropy; non-contradiction cluster count and entropy (pre-registered backup criterion);
coherence slope. Soft (v1.2): mean over unordered pairs of min(e_ij, e_ji) — the paper's
RSE; mean directed contradiction probability; max bidirectional pair.

## Family 4 — retrieval dynamics (4 features)
query diversity; document-repeat rate (mean, last); new-document saturation.

## Probe protocol (pre-registered)
Trigger: first-round unanimity (s_1(m_1) = 1). Budget per probed question: one
counter-query generation call (temperature 0), one retrieval round with the substrate's
own stack, N concurrent devil's-advocate re-answers. Statistic: keep rate = fraction of
re-answers retaining the consensus. Reporting: flip-rate curves for correct and incorrect
consensus separately.

## Mondrian stratification (pre-registered before any calibration data)
S1: first-round unanimous — scored by z(keep rate) + z(RSE).
S2: contested but converged — scored by the combined process score.
S3: never converged — abstains as a whole.
