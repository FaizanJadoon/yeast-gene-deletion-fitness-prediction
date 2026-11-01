
from pathlib import Path
import gzip
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/GSE46145_series_matrix.txt.gz"
OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "results/tables"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Zero-based sample indices in the GEO series matrix.
# Pool assignments are based on the GEO sample titles.
#
# IMPORTANT: Sample 6/7/9/10 and sample 8/11 share pools.
# The remaining endpoints have individual pools.

POOL_MAP = {
    5: 0, 6: 0, 7: 1,
    8: 0, 9: 0, 10: 1,
    11: 2, 12: 3, 13: 4,

    19: 14, 20: 15, 21: 15,
    22: 14, 23: 15, 24: 15,
    25: 16, 26: 17, 27: 18,
}

CONDITIONS = {
    "HIP": {
        "generations": 20,
        "YPD": [5, 6, 7],
        "Phase_I": [8, 9, 10],
        "Phase_II": [11, 12, 13],
    },
    "HOP": {
        "generations": 10,
        "YPD": [19, 20, 21],
        "Phase_I": [22, 23, 24],
        "Phase_II": [25, 26, 27],
    },
}

GENE_RE = re.compile(r"(Y[A-P][LR]\d{3}[WC])::")


def load_matrix():
    print("[1/5] Loading GEO barcode matrix")

    with gzip.open(SOURCE, "rt", encoding="utf-8") as f:
        lines = []
        inside = False

        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if inside:
                lines.append(line)

    from io import StringIO

    matrix = pd.read_csv(
        StringIO("".join(lines)),
        sep="\t",
        index_col=0,
    )

    if matrix.shape[1] != 28:
        raise ValueError(
            f"Expected 28 samples, found {matrix.shape[1]}"
        )

    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    print("Barcode rows:", matrix.shape[0])
    print("Samples:", matrix.shape[1])
    return matrix


def prepare_barcode_table(matrix):
    print("[2/5] Extracting gene and barcode identifiers")

    genes = matrix.index.astype(str).str.extract(
        GENE_RE.pattern, expand=False
    )

    valid = genes.notna()
    matrix = matrix.loc[valid].copy()
    genes = genes[valid]

    matrix.insert(0, "gene_id", genes.to_numpy())

    print("Usable barcode rows:", len(matrix))
    print("Unique genes:", matrix["gene_id"].nunique())
    return matrix


def calculate_relative_scores(data):
    print("[3/5] Calculating matched-pool relative scores")

    sample_columns = list(data.columns[1:])
    intensity = data[sample_columns].to_numpy(dtype=np.float64)

    # Microarray signals are positive. Clip to avoid undefined logs.
    intensity = np.maximum(intensity, 1e-6)

    # A log2 change per generation, relative to the matched pool.
    scores = {}

    for collection, settings in CONDITIONS.items():
        generations = settings["generations"]

        for condition, indices in settings.items():
            if condition == "generations":
                continue

            for replicate, endpoint in enumerate(indices, start=1):
                pool = POOL_MAP[endpoint]

                change = (
                    np.log2(intensity[:, endpoint])
                    - np.log2(intensity[:, pool])
                ) / generations

                # Center the barcode distribution per sample.
                change -= np.nanmedian(change)

                scores[
                    f"{collection}_{condition}_rep{replicate}"
                ] = change

    result = pd.DataFrame(scores)
    result.insert(0, "gene_id", data["gene_id"].to_numpy())

    return result


def aggregate_and_normalize(barcode_scores):
    print("[4/5] Aggregating and normalizing fitness")

    # Median combines multiple barcodes for the same gene.
    gene_scores = barcode_scores.groupby(
        "gene_id", sort=True
    ).median(numeric_only=True)

    output = pd.DataFrame(index=gene_scores.index)

    for collection in CONDITIONS:
        control = [
            f"{collection}_YPD_rep{i}" for i in range(1, 4)
        ]

        control_mean = gene_scores[control].mean(axis=1)

        for condition in ["Phase_I", "Phase_II"]:
            cols = [
                f"{collection}_{condition}_rep{i}"
                for i in range(1, 4)
            ]

            # Each condition is compared to the mean YPD control.
            relative = gene_scores[cols].sub(
                control_mean, axis=0
            )

            prefix = f"{collection}_{condition}"

            for i, col in enumerate(relative.columns, start=1):
                output[f"{prefix}_rep{i}"] = relative[col]

            output[f"{prefix}_mean"] = relative.mean(axis=1)
            output[f"{prefix}_sd"] = relative.std(axis=1)
            output[f"{prefix}_n"] = relative.notna().sum(axis=1)

    output.index.name = "gene_id"
    return gene_scores, output


def evaluate_replicates(gene_scores, fitness):
    print("[5/5] Evaluating replicate agreement")

    quality = {}

    for collection in CONDITIONS:
        for condition in ["YPD", "Phase_I", "Phase_II"]:
            cols = [
                f"{collection}_{condition}_rep{i}"
                for i in range(1, 4)
            ]

            correlations = []

            for i in range(3):
                for j in range(i + 1, 3):
                    a = gene_scores[cols[i]]
                    b = gene_scores[cols[j]]

                    mask = a.notna() & b.notna()

                    if mask.sum() >= 10:
                        rho = spearmanr(
                            a[mask], b[mask]
                        ).statistic
                        correlations.append(float(rho))

            quality[f"{collection}_{condition}"] = {
                "replicate_spearman": correlations,
                "median_replicate_spearman": (
                    float(np.median(correlations))
                    if correlations else None
                ),
            }

    for collection in CONDITIONS:
        for condition in ["Phase_I", "Phase_II"]:
            prefix = f"{collection}_{condition}"
            quality[prefix]["genes_with_3_replicates"] = int(
                (fitness[f"{prefix}_n"] == 3).sum()
            )

    return quality


def main():
    print("=" * 64)
    print("FAIZAN PAPER 3 — FERMENTATION FITNESS PREPARATION")
    print("=" * 64)

    matrix = load_matrix()
    barcodes = prepare_barcode_table(matrix)
    barcode_scores = calculate_relative_scores(barcodes)

    gene_scores, fitness = aggregate_and_normalize(
        barcode_scores
    )

    quality = evaluate_replicates(gene_scores, fitness)

    gene_scores.to_parquet(
        OUTPUT / "fermentation_gene_scores.parquet"
    )
    fitness.reset_index().to_parquet(
        OUTPUT / "fermentation_fitness.parquet",
        index=False
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GSE46145",
        "normalization": (
            "Matched-pool log2 intensity change per generation, "
            "barcode-wise median centering, gene-wise median "
            "aggregation, and subtraction of mean YPD controls"
        ),
        "fitness_type": "relative_barcode_derived_proxy",
        "biological_validation_status": "PENDING",
        "pool_map_zero_based": POOL_MAP,
        "replicate_quality": quality,
        "gene_count": len(fitness),
    }

    (REPORTS / "fermentation_quality.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nGenes:", len(fitness))
    for name, values in quality.items():
        print(
            name,
            "median Spearman:",
            values["median_replicate_spearman"]
        )

    print("\nSaved:")
    print("data/processed/fermentation_gene_scores.parquet")
    print("data/processed/fermentation_fitness.parquet")
    print("results/tables/fermentation_quality.json")
    print("\nSTATUS: Fitness proxies prepared; validation pending.")


if __name__ == "__main__":
    main()
