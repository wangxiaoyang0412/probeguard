"""Generate the LaTeX macros in paper/numbers.tex from results/ and runs/ — the single source for every number in the paper.
Hard rule: no hand-written numbers in the text; writing uses macros only. Re-running this script updates the whole paper.
Usage (run where both runs/ and results/ are present):
  python analysis/gen_numbers.py --results results --runs runs --out results/numbers.tex
v2 (2026-08-21): AUROC macros now parsed programmatically from auroc_table_*.csv (incl. cross-backbone tables);
verbalized/ptrue/probe macros computed from jsonl; dataset stats extended to two new backbones; missing files skipped with a comment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def ds_stats(out: list, tex: str, path: Path) -> None:
    if not path.exists():
        out.append(f"% {tex}: run jsonl missing, stats not emitted")
        return
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    ua = [r for r in recs if r["rounds"][0]["agreement_rate"] == 1.0]
    uaerr = sum(1 for r in ua if not r["correct"])
    out += [
        macro(f"N{tex}", n),
        macro(f"Acc{tex}", f"{acc*100:.1f}"),
        macro(f"UaRate{tex}", f"{len(ua)/n*100:.0f}"),
        macro(f"UaN{tex}", len(ua)),
        macro(f"UaErrN{tex}", uaerr),
        macro(f"UaErrRate{tex}", f"{uaerr/max(len(ua),1)*100:.1f}"),
    ]


def auroc_macros(out: list, tex: str, path: Path) -> None:
    if not path.exists():
        out.append(f"% {tex}: auroc table missing, macros not emitted")
        return
    rows = {r[0]: float(r[1]) for r in csv.reader(open(path)) if r and r[0] != "feature"}
    out += [
        macro(f"Combined{tex}", f"{rows['COMBINED_logreg']:.3f}"),
        macro(f"Terminal{tex}", f"{rows['bl_terminal_agreement']:.3f}"),
        macro(f"ROne{tex}", f"{rows['bl_r1_agreement(strict-ref)']:.3f}"),
        macro(f"SeqLogprob{tex}", f"{rows['bl_seq_logprob_mean']:.3f}"),
        # answer-distribution entropy, orientation-corrected (higher entropy -> wrong)
        macro(f"AnsEntropy{tex}", f"{1 - rows['f1_1_entropy_first']:.3f}"),
    ]


def conf_auroc(out: list, tex: str, path: Path, key: str) -> None:
    if not path.exists():
        out.append(f"% {tex}: {path.name} missing")
        return
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if r.get(key) is not None and "error" not in r]
    y = np.array([int(r["correct"]) for r in ok])
    s = np.array([r[key] for r in ok], float)
    out.append(macro(tex, f"{roc_auc_score(y, s):.3f}"))


def probe_macros(out: list, tex: str, path: Path) -> None:
    if not path.exists():
        out.append(f"% probe {tex}: missing")
        return
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if "error" not in r]
    wrong = [r for r in ok if not r["consensus_correct"]]
    right = [r for r in ok if r["consensus_correct"]]
    y = np.array([int(r["consensus_correct"]) for r in ok])
    keep = np.array([r["keep_rate"] for r in ok], float)
    out += [
        macro(f"FlipWrong{tex}", f"{np.mean([r['flipped'] for r in wrong])*100:.1f}"),
        macro(f"FlipRight{tex}", f"{np.mean([r['flipped'] for r in right])*100:.1f}"),
        macro(f"Keep{tex}", f"{roc_auc_score(y, keep):.3f}"),
        macro(f"FlipToGoldN{tex}", sum(1 for r in wrong if r.get("flip_to_gold"))),
        macro(f"ProbeWrongN{tex}", len(wrong)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="paper/numbers.tex")
    args = parser.parse_args()
    R, RUNS = Path(args.results), Path(args.runs)
    out = []

    # ---- dataset stats (per backbone run) ----
    for tex, sub in (
        ("MedQA", "medqa/qwen3-8b"), ("MMLUPro", "mmlu-pro/qwen3-8b"),
        ("MedBullets", "medbullets/qwen3-8b"), ("MedXpert", "medxpertqa/qwen3-8b"),
        ("MedMCQA", "medmcqa/qwen3-8b"),
        ("MedQAHuatuo", "medqa/huatuogpt-o1-8b"), ("MMLUProHuatuo", "mmlu-pro/huatuogpt-o1-8b"),
        ("MedQALlama", "medqa/llama-3.1-8b"), ("MMLUProLlama", "mmlu-pro/llama-3.1-8b"),
    ):
        stem = sub.split("/")[0].replace("-", "")
        ds_stats(out, tex, RUNS / sub / "exp_t1_full" / f"full_{stem}.jsonl")

    # ---- AUROC tables (t3 outputs, incl. cross-backbone) ----
    out.append("% ---- discrimination (parsed from auroc_table_*.csv) ----")
    for tex, fn in (
        ("MedQA", "auroc_table_medqa.csv"), ("MMLUPro", "auroc_table_mmlupro.csv"),
        ("MedMCQA", "auroc_table_medmcqa.csv"),
        ("MedQAHuatuo", "auroc_table_medqa_huatuo.csv"),
        ("MMLUProHuatuo", "auroc_table_mmlupro_huatuo.csv"),
        ("MedQALlama", "auroc_table_medqa_llama.csv"),
        ("MMLUProLlama", "auroc_table_mmlupro_llama.csv"),
    ):
        auroc_macros(out, tex, R / fn)

    # ---- AURC (parsed from aurc_table_*.csv; analysis/aurc_table.py) ----
    out.append("% ---- AURC ----")
    namemap = {"terminal": "AurcTerminal", "r1": "AurcROne", "seq_logprob": "AurcSeqLogprob",
               "ans_entropy": "AurcAnsEntropy", "combined": "AurcCombined",
               "verbalized": "AurcVerbalized", "ptrue": "AurcPTrue"}
    for tex, fn in (("MedQA", "aurc_table_medqa.csv"), ("MMLUPro", "aurc_table_mmlupro.csv"),
                    ("MedMCQA", "aurc_table_medmcqa.csv"),
                    ("MedQAHuatuo", "aurc_table_medqa_huatuo.csv"),
                    ("MMLUProHuatuo", "aurc_table_mmlupro_huatuo.csv"),
                    ("MedQALlama", "aurc_table_medqa_llama.csv"),
                    ("MMLUProLlama", "aurc_table_mmlupro_llama.csv")):
        p = R / fn
        if not p.exists():
            out.append(f"% aurc {tex}: missing")
            continue
        rows = {r[0]: r[1] for r in csv.reader(open(p)) if r and r[0] != "signal"}
        out += [macro(f"{pre}{tex}", f"{float(rows[k]):.3f}")
                for k, pre in namemap.items() if k in rows]

    # ---- self-report baselines (computed from jsonl) ----
    out.append("% ---- self-report baselines ----")
    conf_auroc(out, "VerbalizedMedQA", R / "verbalized_medqa.jsonl", "verbalized_conf")
    conf_auroc(out, "VerbalizedMMLUPro", R / "verbalized_mmlupro.jsonl", "verbalized_conf")
    conf_auroc(out, "PTrueMedQA", R / "ptrue_medqa.jsonl", "ptrue")
    conf_auroc(out, "PTrueMMLUPro", R / "ptrue_mmlupro.jsonl", "ptrue")

    # ---- probes (computed from jsonl; qwen MedQA full-layer kept frozen below) ----
    out.append("% ---- probes ----")
    probe_macros(out, "MedMCQA", R / "probe_medmcqa_ua.jsonl")
    probe_macros(out, "MMLUPro", R / "probe_mmlupro_ua.jsonl")
    probe_macros(out, "MedQAHuatuo", R / "probe_medqa_ua_huatuo.jsonl")
    probe_macros(out, "MMLUProHuatuo", R / "probe_mmlupro_ua_huatuo.jsonl")
    probe_macros(out, "MedQALlama", R / "probe_medqa_ua_llama.jsonl")
    probe_macros(out, "MMLUProLlama", R / "probe_mmlupro_ua_llama.jsonl")

    # ---- frozen analysis outputs (not derivable from files above) ----
    frozen = {
        # certification narrative (phase1 v1/v3 + repeated-split analyses)
        "NaiveViolation": "15.7", "CertGlobalShare": "35",
        # ablation/sensitivity (attribution.py, soft_eval, holes.py h4/h5)
        "AblDropFThree": "0.682", "HardFThreeMedQA": "0.578", "HardFThreeMMLUPro": "0.593",
        "RhoTrajMin": "0.93", "RhoRseMax": "0.39", "SmokeDelta": "0.007",
        "TransferBullets": "0.555",
        "FThreeMedQA": "0.587", "FThreeMMLUPro": "0.613",
        "KeepMedQA": "0.642", "ComboUaMedQA": "0.685", "ComboUaMMLUPro": "0.688",
        "FlipWrongMedQA": "14.0", "FlipRightMedQA": "4.2",
        "CertAlpha": "0.15", "CertCovMean": "0.598", "CertCovSE": "0.029",
        "CertRisk": "8.9", "CertNonzero": "46", "CertRepeats": "50",
        # fuel-augmented certification (analysis/c8_fuel_cert.py, R=50, zero violations)
        "FuelN": "875", "FuelCertCovTen": "0.236", "FuelCertCovTenSE": "0.023",
        "FuelCertRiskTen": "6.3", "FuelCertCovFifteen": "0.909",
        "FuelCertCovFifteenSE": "0.003", "FuelCertRiskFifteen": "11.8",
        # cross-backbone S1 certification (analysis/fig_cert.py arms, R=50)
        "CertCovTenHuatuo": "0.425", "CertCovFifteenHuatuo": "0.997",
        "CertCovFifteenLlama": "0.869",
        "ProbeMedianSecMedQA": "114", "ProbeMedianSecMMLUPro": "100",
        "ProbeMedianTok": "4381", "GenMedianSecMedQA": "72",
    }
    out.append("% ---- frozen analysis outputs (update via upstream scripts) ----")
    out += [macro(k, v) for k, v in frozen.items()]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        "% AUTO-GENERATED by analysis/gen_numbers.py — DO NOT EDIT BY HAND\n"
        + "\n".join(out) + "\n")
    print(f"wrote {len(out)} lines -> {args.out}")


if __name__ == "__main__":
    main()
