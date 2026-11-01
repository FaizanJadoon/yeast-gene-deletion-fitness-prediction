
"""
Faizan Paper 3 — Master Reproduction Runner

Usage:
    python run_all.py --mode standard
    python run_all.py --mode full

Standard:
    Reuse existing ESM-2 embeddings and rerun research analyses.

Full:
    Back up existing embeddings, regenerate them, and rerun analyses.

Run from the project's Python virtual environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
EMBEDDINGS = RESULTS / "embeddings"
LOGS = ROOT / "logs"
RUNS = LOGS / "reproduction"

MODELS = ("ESM2_8M", "ESM2_35M", "ESM2_150M", "ESM2_650M")

STAGES = [
    ("Fermentation data preparation", "prepare_fermentation_fitness.py"),
    ("Data source audit", "audit_data_sources.py"),
    ("Baseline and ESM-2 experiments", "run_experiments.py"),
    ("Experiment analysis", "analyze_experiments.py"),
    ("Generalization evaluation", "evaluate_generalization.py"),
    ("Homology-controlled validation", "validate_homology.py"),
    ("Chemical-response experiments", "final_experiments.py"),
    ("Cross-study transfer", "cross_study_transfer.py"),
    ("Final publication figures", "final_figures.py"),
]

EXPECTED = [
    "data/processed/fermentation_fitness.parquet",
    "data/processed/research_cohort.parquet",
    "results/tables/model_results.csv",
    "results/tables/homology_results.csv",
    "results/tables/homology_predictions.parquet",
    "results/final_experiments/chemical_prediction_results.csv",
    "results/final_experiments/chemical_predictions.csv.gz",
    "results/final_experiments/cross_study_transfer_results.csv",
    "results/final_experiments/cross_study_transfer_bootstrap.csv",
    "figures/final/figure_01_fermentation.pdf",
    "figures/final/figure_02_chemical.pdf",
    "figures/final/figure_03_scaling.pdf",
    "figures/final/figure_04_transfer.pdf",
]


def timestamp():
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_details():
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "transformers",
        "biopython",
        "pyarrow",
        "matplotlib",
        "mmseqs",
    ]

    details = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            p: installed_version(p) for p in packages
        },
    }

    try:
        import torch

        details["cuda_available"] = torch.cuda.is_available()
        details["torch_cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            details["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        details["torch_check_error"] = str(exc)

    return details


def validate_inputs():
    required = [
        ROOT / "data/processed/yeast_gene_protein_mapping.csv",
        ROOT / "data/raw/GSE46145_series_matrix.txt.gz",
        ROOT / "data/processed/research_cohort.parquet",
        ROOT / "src/run_experiments.py",
    ]

    chemical_candidates = [
        ROOT / "data/raw/HOP_scores_1.txt",
        ROOT / "data/raw/HOP_scores.txt",
    ]

    missing = [str(p) for p in required if not p.is_file()]

    if not any(p.is_file() for p in chemical_candidates):
        missing.append(
            "data/raw/HOP_scores_1.txt or HOP_scores.txt"
        )

    for _, script in STAGES:
        path = SRC / script
        if not path.is_file():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" + "\n".join(missing)
        )


def backup_embeddings(run_dir):
    backup = run_dir / "embedding_backup"
    backup.mkdir(parents=True, exist_ok=True)

    moved = []

    for model in MODELS:
        for suffix in (".npy", "_genes.json"):
            source = EMBEDDINGS / f"{model}{suffix}"

            if source.exists():
                destination = backup / source.name
                shutil.move(str(source), str(destination))
                moved.append((source, destination))

    return moved


def restore_embeddings(moved):
    for original, backup in moved:
        if backup.exists() and not original.exists():
            shutil.move(str(backup), str(original))


def run_stage(label, script, log_file):
    command = [
        sys.executable,
        str(SRC / script),
    ]

    print(f"\n{'=' * 65}", flush=True)
    print(f"STAGE: {label}", flush=True)
    print(f"COMMAND: {' '.join(command)}", flush=True)
    print(f"{'=' * 65}", flush=True)

    started = time.monotonic()

    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"Stage: {label}\n")
        log.write(f"Command: {command}\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)

        return_code = process.wait()

    elapsed = round(time.monotonic() - started, 2)

    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}. "
            f"See {log_file}"
        )

    print(f"COMPLETED: {label} ({elapsed}s)", flush=True)

    return {
        "stage": label,
        "script": script,
        "seconds": elapsed,
        "status": "PASS",
    }


def verify_outputs():
    checks = []

    for relative in EXPECTED:
        path = ROOT / relative
        passed = path.is_file() and path.stat().st_size > 0

        checks.append({
            "file": relative,
            "status": "PASS" if passed else "FAIL",
            "size_bytes": path.stat().st_size
            if path.exists() else 0,
        })

    for model in MODELS:
        for suffix in (".npy", "_genes.json"):
            relative = f"results/embeddings/{model}{suffix}"
            path = ROOT / relative

            passed = path.is_file() and path.stat().st_size > 0

            checks.append({
                "file": relative,
                "status": "PASS" if passed else "FAIL",
                "size_bytes": path.stat().st_size
                if path.exists() else 0,
            })

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("standard", "full"),
        default="standard",
    )
    args = parser.parse_args()

    run_id = timestamp()
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "Faizan Paper 3",
        "run_id": run_id,
        "mode": args.mode,
        "status": "RUNNING",
        "environment": environment_details(),
        "stages": [],
        "output_checks": [],
    }

    report_path = run_dir / "reproduction_report.json"
    save_json(report_path, report)

    moved_embeddings = []

    try:
        validate_inputs()

        if args.mode == "full":
            print(
                "\nFULL MODE: Backing up existing embeddings.",
                flush=True,
            )
            moved_embeddings = backup_embeddings(run_dir)

        for index, (label, script) in enumerate(STAGES, 1):
            log_file = run_dir / f"{index:02d}_{script}.log"

            result = run_stage(label, script, log_file)
            report["stages"].append(result)
            save_json(report_path, report)

        report["output_checks"] = verify_outputs()

        failed = [
            item for item in report["output_checks"]
            if item["status"] != "PASS"
        ]

        if failed:
            raise RuntimeError(
                "Missing or empty outputs:\n"
                + "\n".join(item["file"] for item in failed)
            )

        report["status"] = "PASS"

        print("\nREPRODUCTION RUN COMPLETED", flush=True)
        print("Mode:", args.mode, flush=True)
        print("Verified outputs:", len(report["output_checks"]))
        print("Report:", report_path)

    except Exception:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()

        print("\nREPRODUCTION RUN FAILED", flush=True)
        print(report["error"], flush=True)

        if args.mode == "full":
            restore_embeddings(moved_embeddings)

        raise

    finally:
        report["completed_utc"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
        save_json(report_path, report)


if __name__ == "__main__":
    main()
