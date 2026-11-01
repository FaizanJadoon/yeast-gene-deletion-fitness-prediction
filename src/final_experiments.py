
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import validate_homology as homology


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "HOP_scores_1.txt"
OUT = ROOT / "results" / "final_experiments"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_SPLITS = 5
TEST_SIZE = 0.20
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN = 100
MIN_TEST = 50
MAX_CONDITIONS = 60

PATTERN = re.compile(
    r"^Ad\. scores for Exp\. ([^_]+)_([^_]+)_(.+)$"
)


def log(message):
    print(message, flush=True)


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=str),
        encoding="utf-8",
    )


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    cohort = homology.load_cohort()
    groups, _ = homology.load_clusters(
        cohort,
        homology.WORK / "clusters_cluster.tsv",
    )
    features = homology.load_features(cohort)

    if isinstance(features, tuple):
        raise TypeError(
            "Unexpected feature structure. Inspect "
            "validate_homology.load_features before continuing."
        )

    genes = cohort["gene_id"].astype(str).tolist()

    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate cohort genes")

    if len(groups) != len(genes):
        raise ValueError("Cluster assignment length mismatch")

    for name, matrix in features.items():
        if len(matrix) != len(genes):
            raise ValueError(f"Feature alignment mismatch: {name}")

    return cohort, np.asarray(groups), features, genes


def select_conditions(genes):
    log("Reading chemical-response matrix...")

    header = pd.read_csv(RAW, sep="\t", nrows=0).columns
    valid = [c for c in header[1:] if PATTERN.match(c)]

    if len(valid) != 5808:
        raise ValueError(
            f"Expected 5808 adjusted HOP columns; found {len(valid)}"
        )

    # Deterministic, evenly spaced condition selection.
    # This is a computational pilot, not a random sample of compounds.
    indices = np.linspace(
        0, len(valid) - 1, MAX_CONDITIONS, dtype=int
    )
    chosen = [valid[i] for i in sorted(set(indices))]

    frame = pd.read_csv(
        RAW,
        sep="\t",
        index_col=0,
        usecols=[header[0]] + chosen,
        low_memory=False,
    )

    frame.index = frame.index.astype(str)
    frame = frame.reindex(genes)

    if frame.index.tolist() != genes:
        raise ValueError("Chemical-response gene alignment failed")

    metadata = []
    for name in chosen:
        match = PATTERN.match(name)
        metadata.append(
            {
                "column": name,
                "experiment_id": match.group(1),
                "concentration": match.group(2),
                "assay_id": match.group(3),
                "n_measured": int(frame[name].notna().sum()),
            }
        )

    pd.DataFrame(metadata).to_csv(
        OUT / "selected_conditions.csv", index=False
    )

    log(f"Selected chemical conditions: {len(chosen)}")
    return frame, chosen


def rank_correlation(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred).statistic)


def ridge_predict(x, y, train, test, groups):
    # Hyperparameter selection uses training genes only.
    inner = GroupKFold(n_splits=3)
    best_alpha = None
    best_score = -np.inf

    for alpha in ALPHAS:
        scores = []

        for inner_train, inner_val in inner.split(
            x[train], y[train], groups[train]
        ):
            tr = train[inner_train]
            va = train[inner_val]

            scaler = StandardScaler()
            xtr = scaler.fit_transform(x[tr])
            xva = scaler.transform(x[va])

            model = Ridge(alpha=alpha)
            model.fit(xtr, y[tr])
            prediction = model.predict(xva)

            score = rank_correlation(y[va], prediction)
            if np.isfinite(score):
                scores.append(score)

        score = float(np.mean(scores)) if scores else -np.inf

        if score > best_score:
            best_score = score
            best_alpha = alpha

    if best_alpha is None:
        best_alpha = 100.0

    scaler = StandardScaler()
    xtr = scaler.fit_transform(x[train])
    xte = scaler.transform(x[test])

    model = Ridge(alpha=best_alpha)
    model.fit(xtr, y[train])

    return model.predict(xte), best_alpha


def evaluate_chemical(chemical, conditions, features, groups, genes):
    log("Running chemical-response prediction...")

    splits = list(
        GroupShuffleSplit(
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            random_state=SEED,
        ).split(np.arange(len(genes)), groups=groups)
    )

    results = []
    predictions = []

    for condition_number, condition in enumerate(conditions, 1):
        y = pd.to_numeric(
            chemical[condition], errors="coerce"
        ).to_numpy(dtype=np.float64)

        for split_number, (train_all, test_all) in enumerate(
            splits, 1
        ):
            train = train_all[np.isfinite(y[train_all])]
            test = test_all[np.isfinite(y[test_all])]

            if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
                continue

            for model_name, matrix in features.items():
                x = np.asarray(matrix, dtype=np.float32)

                predicted, alpha = ridge_predict(
                    x, y, train, test, groups
                )

                score = rank_correlation(y[test], predicted)

                results.append(
                    {
                        "condition": condition,
                        "split": split_number,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "spearman": score,
                        "alpha": alpha,
                    }
                )

                for index, value in zip(test, predicted):
                    predictions.append(
                        {
                            "condition": condition,
                            "split": split_number,
                            "model": model_name,
                            "gene_id": genes[index],
                            "observed": float(y[index]),
                            "predicted": float(value),
                        }
                    )

        log(
            f"Chemical condition {condition_number}/"
            f"{len(conditions)} completed"
        )

    result = pd.DataFrame(results)
    result.to_csv(
        OUT / "chemical_prediction_results.csv", index=False
    )

    pd.DataFrame(predictions).to_csv(
        OUT / "chemical_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = (
        result.groupby("model", as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            sd_spearman=("spearman", "std"),
            evaluations=("spearman", "count"),
        )
        .sort_values("mean_spearman", ascending=False)
    )

    summary.to_csv(
        OUT / "chemical_prediction_summary.csv", index=False
    )

    log("\nCHEMICAL-RESPONSE PREDICTION")
    log(summary.round(4).to_string(index=False))


def evaluate_transfer(chemical, conditions, genes):
    log("\nRunning fermentation-to-chemical association analysis...")

    fitness_path = ROOT / "data" / "processed" / "fermentation_fitness.parquet"
    fitness = pd.read_parquet(fitness_path)

    if "gene_id" not in fitness.columns:
        raise ValueError("Missing gene_id in fermentation targets")

    fitness = fitness.set_index("gene_id").reindex(genes)

    target_names = [
        "HOP_Phase_I_mean",
        "HOP_Phase_II_mean",
    ]

    results = []

    for target in target_names:
        if target not in fitness.columns:
            raise ValueError(f"Missing fermentation target: {target}")

        fermentation = pd.to_numeric(
            fitness[target], errors="coerce"
        ).to_numpy(dtype=float)

        for condition in conditions:
            chemical_scores = pd.to_numeric(
                chemical[condition], errors="coerce"
            ).to_numpy(dtype=float)

            valid = (
                np.isfinite(fermentation)
                & np.isfinite(chemical_scores)
            )

            if valid.sum() < 100:
                continue

            results.append(
                {
                    "fermentation_target": target,
                    "chemical_condition": condition,
                    "n_genes": int(valid.sum()),
                    "spearman": rank_correlation(
                        fermentation[valid],
                        chemical_scores[valid],
                    ),
                }
            )

    result = pd.DataFrame(results)
    result.to_csv(
        OUT / "cross_study_target_associations.csv",
        index=False,
    )

    summary = (
        result.groupby("fermentation_target", as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            conditions=("spearman", "count"),
        )
    )

    summary.to_csv(
        OUT / "cross_study_association_summary.csv",
        index=False,
    )

    log("\nCROSS-STUDY TARGET ASSOCIATIONS")
    log(summary.round(4).to_string(index=False))


def main():
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "test_size": TEST_SIZE,
        "alphas": ALPHAS,
        "max_conditions": MAX_CONDITIONS,
        "split_method": "MMseqs2 homology-cluster GroupShuffleSplit",
        "model_selection": "inner GroupKFold",
        "chemical_data": str(RAW),
        "chemical_sha256": fingerprint(RAW),
        "note": (
            "Chemical prediction is a 60-condition pilot. "
            "Cross-study target associations are descriptive "
            "and are not fermentation-trained model transfer."
        ),
    }

    save_json(OUT / "experiment_config.json", config)

    cohort, groups, features, genes = load_inputs()
    chemical, conditions = select_conditions(genes)

    evaluate_chemical(
        chemical, conditions, features, groups, genes
    )
    evaluate_transfer(chemical, conditions, genes)

    log("\nFINAL EXPERIMENT PILOT COMPLETED")
    log(f"Results: {OUT}")


if __name__ == "__main__":
    main()
