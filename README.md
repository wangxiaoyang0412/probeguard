# ProbeGuard

Certified abstention for multi-round medical consensus systems: process features from the
consensus-formation trajectory, rationale semantic entropy and an active counter-evidence
probe for unanimous votes, and stratified Learn-then-Test calibration with a distribution-free
selective-risk guarantee.

This repository accompanies the paper *Unanimously Wrong: Certified Abstention from How
Medical LLM Consensus Forms*.

## Layout

- `abstain_layer/` — instrumentation wrapper around the substrate, feature extraction
  (`features.py`, `rationale_features.py`, `t2_extract.py`), evaluation (`t3_evaluate.py`),
  counter-evidence probe (`probe_pilot.py`), self-report baselines (`verbalized_conf.py`,
  `ptrue_conf.py`), and calibration (`phase1_conformal.py`, `phase1_mondrian.py`).
- `analysis/` — table/figure generators and analysis scripts used in the paper.
- `patches/` — diffs applied to the substrate checkout for environment compatibility and
  response-format robustness across backbones; none touch the consensus decision logic.
- `results/` — per-question probe records, feature tables, per-signal AUROC/AURC tables,
  and calibration outputs backing every number in the paper.
- `registry.md` — the frozen feature registry (definitions fixed before evaluation data
  were inspected).
- `tests/` — unit tests for schema validation, feature extraction, and the claim-extraction rule.

## Setup

1. Check out the substrate (MA-RAG) at the commit pinned in the paper and apply `patches/`.
2. `pip install -r requirements.txt`; serve backbones with vLLM; start the substrate's
   retrieval service over MedCorp.
3. Run the pipeline: `run_instrumented.py` (generation with logging) →
   `convert_logs.py` → `t2_extract.py` (features) → `t3_evaluate.py` (discrimination) →
   `probe_pilot.py --all-unanimous` (probe) → `phase1_mondrian.py` (certification).

Prompts are reproduced verbatim in the paper's Appendix A; hyperparameters in Appendix K.
