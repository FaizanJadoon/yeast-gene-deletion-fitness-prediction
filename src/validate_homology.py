
from pathlib import Path
import csv
import json
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"
EMBEDDINGS = ROOT / "results" / "embeddings"
WORK = ROOT / "results" / "homology"

for folder in (TABLES, WORK):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
REPEATS = 5
TEST_SIZE = 0.20

MIN_IDENTITY = 0.30
MIN_COVERAGE = 0.80

TARGETS = [
    "HOP_Phase_I_mean",
    "HOP_Phase_II_mean",
]

MODELS = [
    "Sequence_Composition",
    "ESM2_8M",
    "ESM2_35M",
    "ESM2_150M",
    "ESM2_650M",
]

ALPHAS = np.logspace(-2, 5, 12)

AA = "ACDEFGHIKLMNPQRSTVWY"


def log(message):
    print(message, flush=True)


def load_cohort():
    cohort = pd.read_parquet(
        PROCESSED / "research_cohort.parquet"
    )

    fitness = pd.read_parquet(
        PROCESSED / "fermentation_fitness.parquet"
    )

    df = cohort.merge(
        fitness,
        on="gene_id",
        how="inner",
        validate="one_to_one",
    )

    df = df.dropna(
        subset=["sequence"] + TARGETS
    ).copy()

    df["sequence"] = (
        df["sequence"].astype(str).str.upper()
    )

    df = df.sort_values("gene_id").reset_index(drop=True)

    if len(df) != 4491:
        raise ValueError(
            f"Expected 4491 genes, found {len(df)}"
        )

    if df["gene_id"].duplicated().any():
        raise ValueError("Duplicate genes detected")

    return df


def write_fasta(df):
    fasta = WORK / "cohort_proteins.fasta"

    with fasta.open("w", encoding="utf-8") as handle:
        for row in df.itertuples():
            handle.write(
                f">{row.gene_id}\n{row.sequence}\n"
            )

    return fasta


def find_mmseqs():
    executable = shutil.which("mmseqs")

    if executable is None:
        raise RuntimeError(
            "MMseqs2 executable not found on PATH. "
            "Install the free MMseqs2 binary before "
            "running this script. No paid services needed."
        )

    return executable


def run_mmseqs(executable, fasta):
    prefix = WORK / "clusters"
    tsv = WORK / "clusters_cluster.tsv"

    if tsv.exists():
        log("Existing MMseqs2 clusters found")
        return tsv

    command = [
        executable,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(WORK / "tmp"),
        "--min-seq-id",
        str(MIN_IDENTITY),
        "-c",
        str(MIN_COVERAGE),
        "--cov-mode",
        "5",
        "--cluster-mode",
        "0",
        "--threads",
        "6",
    ]

    log("Running MMseqs2 protein clustering")
    log(" ".join(command))

    subprocess.run(command, check=True)

    if not tsv.exists():
        raise FileNotFoundError(
            f"MMseqs2 output not found: {tsv}"
        )

    return tsv


def load_clusters(df, tsv):
    clusters = pd.read_csv(
        tsv,
        sep="\t",
        header=None,
        names=["representative", "gene_id"],
    )

    if clusters["gene_id"].duplicated().any():
        raise ValueError(
            "Duplicate members in MMseqs2 clusters"
        )

    assignments = df[["gene_id"]].merge(
        clusters,
        on="gene_id",
        how="left",
        validate="one_to_one",
    )

    if assignments["representative"].isna().any():
        missing = assignments[
            assignments["representative"].isna()
        ]

        raise ValueError(
            f"Missing cluster assignments: {len(missing)}"
        )

    assignments["group"] = pd.factorize(
        assignments["representative"]
    )[0]

    assignments.to_csv(
        PROCESSED / "mmseqs_homology_groups.csv",
        index=False,
    )

    sizes = assignments["group"].value_counts()

    log(f"Protein sequences: {len(df)}")
    log(f"Clusters: {len(sizes)}")
    log(f"Largest cluster: {sizes.max()}")
    log(f"Singleton clusters: {(sizes == 1).sum()}")

    return assignments["group"].to_numpy(), sizes


def sequence_features(sequences):
    rows = []

    for sequence in sequences:
        length = max(len(sequence), 1)

        features = [
            np.log1p(length),
            float(length > 1022),
        ]

        features.extend([
            sequence.count(aa) / length
            for aa in AA
        ])

        features.extend([
            sum(sequence.count(a) for a in "AILMFWV")
            / length,
            sum(sequence.count(a) for a in "DE")
            / length,
            sum(sequence.count(a) for a in "KRH")
            / length,
            sum(sequence.count(a) for a in "STNQ")
            / length,
            sum(sequence.count(a) for a in "GP")
            / length,
        ])

        rows.append(features)

    return np.asarray(rows, dtype=np.float64)


def load_features(df):
    genes = df["gene_id"].tolist()

    features = {
        "Sequence_Composition": sequence_features(
            df["sequence"].tolist()
        )
    }

    for model in MODELS[1:]:
        array = np.load(
            EMBEDDINGS / f"{model}.npy"
        ).astype(np.float64)

        saved_genes = json.loads(
            (
                EMBEDDINGS / f"{model}_genes.json"
            ).read_text(encoding="utf-8")
        )

        if saved_genes != genes:
            raise ValueError(
                f"Gene ordering mismatch for {model}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"Nonfinite embeddings for {model}"
            )

        features[model] = array

    return features


def select_alpha(X, y, groups):
    unique_groups = np.unique(groups)

    if len(unique_groups) < 3:
        raise ValueError(
            "Insufficient groups for inner CV"
        )

    folds = GroupKFold(
        n_splits=min(3, len(unique_groups))
    )

    best_alpha = None
    best_error = float("inf")

    for alpha in ALPHAS:
        errors = []

        for train, valid in folds.split(
            X, y, groups
        ):
            estimator = make_pipeline(
                StandardScaler(),
                Ridge(
                    alpha=float(alpha),
                    solver="lsqr",
                    max_iter=5000,
                ),
            )

            estimator.fit(X[train], y[train])

            predicted = estimator.predict(X[valid])

            errors.append(
                float(
                    np.mean(
                        (y[valid] - predicted) ** 2
                    )
                )
            )

        mean_error = float(np.mean(errors))

        if mean_error < best_error:
            best_error = mean_error
            best_alpha = float(alpha)

    return best_alpha


def evaluate(df, features, groups):
    splitter = GroupShuffleSplit(
        n_splits=REPEATS,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    indices = np.arange(len(df))

    rows = []
    predictions = []
    split_records = []

    for repeat, (train, test) in enumerate(
        splitter.split(indices, groups=groups)
    ):
        if set(groups[train]) & set(groups[test]):
            raise ValueError(
                "Train/test cluster overlap detected"
            )

        log(
            f"\nHomology split {repeat + 1}/{REPEATS}: "
            f"train={len(train)}, test={len(test)}"
        )

        for index in train:
            split_records.append({
                "repeat": repeat,
                "gene_id": df.iloc[index]["gene_id"],
                "split": "train",
            })

        for index in test:
            split_records.append({
                "repeat": repeat,
                "gene_id": df.iloc[index]["gene_id"],
                "split": "test",
            })

        for target in TARGETS:
            y = df[target].to_numpy(dtype=float)

            for model_name in MODELS:
                X = features[model_name]

                alpha = select_alpha(
                    X[train],
                    y[train],
                    groups[train],
                )

                estimator = make_pipeline(
                    StandardScaler(),
                    Ridge(
                        alpha=alpha,
                        solver="lsqr",
                        max_iter=5000,
                    ),
                )

                estimator.fit(
                    X[train],
                    y[train],
                )

                predicted = estimator.predict(
                    X[test]
                )

                if (
                    np.std(y[test]) == 0
                    or np.std(predicted) == 0
                ):
                    correlation = np.nan
                else:
                    correlation = float(
                        spearmanr(
                            y[test], predicted
                        ).statistic
                    )

                rows.append({
                    "repeat": repeat,
                    "target": target,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "alpha": alpha,
                    "spearman": correlation,
                    "r2": float(
                        r2_score(y[test], predicted)
                    ),
                    "mae": float(
                        mean_absolute_error(
                            y[test], predicted
                        )
                    ),
                })

                for index, value in zip(
                    test, predicted
                ):
                    predictions.append({
                        "repeat": repeat,
                        "target": target,
                        "model": model_name,
                        "gene_id": df.iloc[index]["gene_id"],
                        "observed": float(y[index]),
                        "predicted": float(value),
                    })

                log(
                    f"  {target} | {model_name}: "
                    f"Spearman={correlation:.4f}"
                )

        pd.DataFrame(rows).to_csv(
            TABLES / "homology_results_partial.csv",
            index=False,
        )

    pd.DataFrame(split_records).to_csv(
        PROCESSED / "homology_splits.csv",
        index=False,
    )

    pd.DataFrame(predictions).to_parquet(
        TABLES / "homology_predictions.parquet",
        index=False,
    )

    return pd.DataFrame(rows)


def main():
    started = time.time()

    log("=" * 65)
    log("FAIZAN PAPER 3 — HOMOLOGY VALIDATION")
    log("=" * 65)

    df = load_cohort()
    fasta = write_fasta(df)

    mmseqs = find_mmseqs()

    cluster_tsv = run_mmseqs(mmseqs, fasta)

    groups, sizes = load_clusters(
        df, cluster_tsv
    )

    features = load_features(df)

    manifest = {
        "seed": SEED,
        "repeats": REPEATS,
        "test_fraction": TEST_SIZE,
        "minimum_sequence_identity": MIN_IDENTITY,
        "minimum_alignment_coverage": MIN_COVERAGE,
        "coverage_mode": 5,
        "cluster_mode": 0,
        "inner_cv": "GroupKFold",
        "models": MODELS,
        "targets": TARGETS,
        "mmseqs_executable": mmseqs,
        "python_version": platform.python_version(),
        "limitations": [
            "MMseqs2 clustering does not guarantee every "
            "cross-split homologous pair is excluded",
            "Thirty-percent identity is an operational "
            "threshold rather than an evolutionary boundary",
            "Frozen embeddings truncate proteins at 1022 "
            "residues",
            "Fitness labels are provisional",
        ],
    }

    (TABLES / "homology_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    results = evaluate(
        df, features, groups
    )

    results.to_csv(
        TABLES / "homology_results.csv",
        index=False,
    )

    summary = (
        results.groupby(
            ["target", "model"]
        )
        .agg(
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_r2=("r2", "mean"),
            mean_mae=("mae", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES / "homology_summary.csv",
        index=False,
    )

    log("\nHOMOLOGY VALIDATION SUMMARY")
    log(summary.round(4).to_string(index=False))

    log(
        f"\nCompleted in "
        f"{(time.time() - started) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
