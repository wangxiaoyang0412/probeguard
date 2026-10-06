# Stored records

Every file here is an output of the pipeline on the datasets and backbones of the paper. Dataset stems: `medqa` (1,273 questions), `mmlupro` (MMLU-Pro medicine, 818), `medmcqa` (4,183), `medxpertqa` (490, hard-frontier reference). Backbone suffixes: none for Qwen3-8B, `_huatuo` for HuatuoGPT-o1-8B, `_llama` for Llama-3.1-8B-Instruct. The random seed is 42 throughout.

| File | Content | Produced by |
|---|---|---|
| `features_<dataset>[_backbone].csv` | One row per question: the process features of the frozen registry (families 0 to 4), the two always-available baseline signals and the correctness label | `abstain_layer/t2_extract.py` |
| `auroc_table_<dataset>[_backbone].csv` | AUROC with bootstrap intervals for every signal and the combined process score | `abstain_layer/t3_evaluate.py` |
| `risk_coverage_<dataset>[_backbone].csv` | Risk-coverage curve of the combined process score | `abstain_layer/t3_evaluate.py` |
| `aurc_table_<dataset>[_backbone].csv` | Area under the risk-coverage curve per signal | `analysis/aurc_table.py` |
| `probe_pilot_medqa.jsonl` | Counter-evidence probe on the MedQA unanimous layer (Qwen3-8B, 746 questions): counter-queries, re-answer votes, keep rate, flip flags | `abstain_layer/probe_pilot.py` |
| `probe_<dataset>_ua[_backbone].jsonl` | The same probe on the other unanimous layers | `abstain_layer/probe_pilot.py` |
| `probe_ctrl_{nodocs,randdocs,neutral}.jsonl` | Control arms of the probe on the same MedQA questions | `abstain_layer/probe_control.py` |
| `soft_f3_<dataset>[_backbone]_ua.csv` | Pairwise NLI aggregates of the rationales on the unanimous layer, including rationale semantic entropy (`f3_5_mean_bidir`) | `abstain_layer/soft_nli.py` |
| `probe_fuel_gbaker.jsonl`, `soft_f3_fuel_ua.csv`, `fuel_provenance.json` | The in-domain calibration questions: 1,500 questions sampled with seed 42 from the `GBaker/MedQA-USMLE-4-options` training split (no overlap with the test set), of which the 875 with a unanimous first round enter calibration; probe records, rationale aggregates and provenance | `abstain_layer/probe_pilot.py`, `abstain_layer/soft_nli.py` |
| `verbalized_<dataset>.jsonl`, `ptrue_<dataset>.jsonl` | Self-report baselines: verbalized confidence and P(True) | `abstain_layer/verbalized_conf.py`, `abstain_layer/ptrue_conf.py` |
| `formation_medqa.json` | Per-round agreement and error statistics behind Figure 2 | derived from the generation logs |
| `medqa_specialty_labels.json` | Specialty labels for 899 MedQA test questions | S-MedQA labels, matched by question |
| `c9_specialty.csv`, `specialty_cert_medqa.csv` | Per-specialty coverage and realized risk of the certified rule (Table 8) | `analysis/c9_specialty.py`, `analysis/specialty_table.py` |
| `phase1_conformal*.csv` | Global (unstratified) calibration runs on the full question stream, kept for the comparison in Section 4.4 | `abstain_layer/phase1_conformal.py` |

The generation logs themselves (every round's candidates, rationales and retrieval events) are not stored here; the feature tables and probe records above are derived from them and are sufficient for every table and figure in the paper.
