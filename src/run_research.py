
from pathlib import Path
import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

GENE_PATTERN = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_proteins():
    print("\n[1/4] Preparing protein mapping")

    mapping_path = PROCESSED / "yeast_gene_protein_mapping.csv"
    df = pd.read_csv(mapping_path)

    required = {
        "gene_id", "uniprot_accession", "sequence", "length"
    }
    if not required.issubset(df.columns):
        raise ValueError("Protein mapping columns are incomplete")

    df = df.dropna(subset=["gene_id", "sequence"]).copy()
    df = df[df["gene_id"].astype(str).str.fullmatch(
        GENE_PATTERN.pattern
    )]

    df["sequence"] = df["sequence"].astype(str).str.upper()
    df["length"] = df["sequence"].str.len()

    duplicate_count = int(df["gene_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Duplicate gene mappings detected: {duplicate_count}"
        )

    df["sequence_sha256"] = df["sequence"].apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()
    )

    output = PROCESSED / "validated_protein_mapping.parquet"
    df.to_parquet(output, index=False)

    print("Mapped genes:", len(df))
    print("Median protein length:", df["length"].median())
    print("Proteins longer than 1022 aa:",
          int((df["length"] > 1022).sum()))

    return df



def inspect_hop():
    print("\n[2/4] Auditing chemical perturbation scores")

    path = RAW / "HOP_scores.txt"

    with path.open("r", encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="\t"))

    adjusted = [
        col for col in header
        if col.startswith("Ad. scores for Exp.")
    ]

    if not adjusted:
        raise ValueError("No adjusted HOP score columns found")

    valid_genes = set()
    standard_gene_rows = 0
    measured_gene_rows = 0

    # Process in chunks to keep memory usage manageable.
    reader = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig",
        usecols=[header[0]] + adjusted,
        chunksize=100,
        low_memory=False
    )

    for chunk in reader:
        genes = chunk.iloc[:, 0].astype(str)
        standard = genes.str.fullmatch(GENE_PATTERN.pattern)

        standard_gene_rows += int(standard.sum())

        scores = chunk[adjusted].apply(
            pd.to_numeric, errors="coerce"
        )

        measured = scores.notna().any(axis=1)
        usable = standard & measured

        measured_gene_rows += int(usable.sum())
        valid_genes.update(genes[usable])

    summary = {
        "standard_gene_rows": standard_gene_rows,
        "measured_gene_rows": measured_gene_rows,
        "unique_measured_genes": len(valid_genes),
        "adjusted_experiment_columns": len(adjusted),
        "score_semantics_validated": False
    }

    save_json(
        summary,
        RESULTS / "tables" / "hop_audit.json"
    )

    print("Standard gene rows:", standard_gene_rows)
    print("Measured gene rows:", measured_gene_rows)
    print("Unique measured genes:", len(valid_genes))
    print("Adjusted experiments:", len(adjusted))

    return valid_genes, summary

    print("\n[2/4] Auditing chemical perturbation scores")

    path = RAW / "HOP_scores.txt"

    with path.open("r", encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="\t"))

    adjusted = [
        col for col in header
        if col.startswith("Ad. scores for Exp.")
    ]

    if not adjusted:
        raise ValueError("No adjusted HOP score columns found")

    # Read only gene identifiers to avoid loading the 482 MB
    # matrix into memory at this stage.
    gene_ids = pd.read_csv(
        path,
        sep="\t",
        usecols=[0],
        encoding="utf-8-sig"
    ).iloc[:, 0].astype(str)

    valid = gene_ids.str.fullmatch(GENE_PATTERN.pattern)

    summary = {
        "total_rows": len(gene_ids),
        "standard_gene_rows": int(valid.sum()),
        "adjusted_experiment_columns": len(adjusted),
        "first_experiment_columns": adjusted[:5],
        "note": (
            "Adjusted scores are not yet interpreted as "
            "absolute fitness. Their meaning and direction "
            "must be validated against source documentation."
        )
    }

    save_json(summary, RESULTS / "tables" / "hop_audit.json")

    print("Standard gene rows:", summary["standard_gene_rows"])
    print("Adjusted experiments:", len(adjusted))

    return set(gene_ids[valid]), summary


def inspect_fermentation():
    print("\n[3/4] Auditing fermentation barcode intensities")

    path = RAW / "GSE46145_series_matrix.txt.gz"

    genes = set()
    barcode_rows = 0
    sample_count = None
    example_rows = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        in_table = False

        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if not in_table:
                continue

            if sample_count is None:
                sample_count = max(
                    len(next(csv.reader([line], delimiter="\t"))) - 1,
                    0
                )
                continue

            barcode_rows += 1

            match = re.search(
                r"(Y[A-P][LR]\d{3}[WC])::",
                line
            )
            if match:
                genes.add(match.group(1))

            if len(example_rows) < 3:
                example_rows.append(line[:160].strip())

    summary = {
        "barcode_rows": barcode_rows,
        "sample_count": sample_count,
        "systematic_genes": len(genes),
        "example_rows": example_rows,
        "label_status": "NOT_DERIVED",
        "warning": (
            "Microarray intensities are not fitness scores. "
            "Initial pools, HIP/HOP, controls, replicate "
            "normalization and barcode aggregation must "
            "be resolved before model training."
        )
    }

    save_json(
        summary,
        RESULTS / "tables" / "fermentation_audit.json"
    )

    print("Barcode rows:", barcode_rows)
    print("Samples:", sample_count)
    print("Genes:", len(genes))

    return genes, summary


def main():
    for directory in [
        PROCESSED, METADATA, RESULTS / "tables",
        RESULTS / "figures", LOGS
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FAIZAN PAPER 3 — REPRODUCIBLE RESEARCH PIPELINE")
    print("=" * 65)

    proteins = prepare_proteins()
    hop_genes, hop_summary = inspect_hop()
    fermentation_genes, fermentation_summary = inspect_fermentation()

    print("\n[4/4] Building research cohort")

    shared = hop_genes & fermentation_genes
    mapped = shared & set(proteins["gene_id"])

    cohort = proteins[
        proteins["gene_id"].isin(mapped)
    ].copy().sort_values("gene_id")

    cohort_path = PROCESSED / "research_cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Faizan Paper 3",
        "total_protein_mappings": len(proteins),
        "shared_experimental_genes": len(shared),
        "shared_genes_with_protein_sequences": len(mapped),
        "mapping_coverage_percent": round(
            100 * len(mapped) / len(shared), 2
        ) if shared else 0,
        "hop_adjusted_experiments":
            hop_summary["adjusted_experiment_columns"],
        "fermentation_samples":
            fermentation_summary["sample_count"],
        "fermentation_fitness_labels_validated": False,
        "chemical_score_semantics_validated": False,
        "model_training_authorized": False,
        "protein_mapping_sha256": file_sha256(
            PROCESSED / "yeast_gene_protein_mapping.csv"
        ),
        "cohort_output": str(cohort_path.relative_to(ROOT))
    }

    save_json(
        manifest,
        RESULTS / "tables" / "research_manifest.json"
    )

    print("\n" + "=" * 65)
    print("COHORT PREPARATION COMPLETE")
    print("=" * 65)
    print("Shared experimental genes:", len(shared))
    print("Genes with protein sequences:", len(mapped))
    print("Coverage:", manifest["mapping_coverage_percent"], "%")
    print("Saved:", cohort_path)
    print("\nSTATUS: Data preparation complete.")
    print("Fitness-label validation required before training.")


if __name__ == "__main__":
    main()
