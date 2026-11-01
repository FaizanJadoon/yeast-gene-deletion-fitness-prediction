
from pathlib import Path
import json
import hashlib
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    KFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    normalize,
)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EMBEDDINGS = ROOT / "results" / "embeddings"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRACTION = 0.20
REPEATS = 5

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

# Approximate similarity analysis only.
# This is not a validated homology threshold.
KMER_SIZE = 3
NEIGHBORS = 20
COSINE_THRESHOLD = 0.65


def log(message):
    print(message, flush=True)


def sequence_features(sequences):
    rows = []

    for sequence in sequences:
        length = max(len(sequence), 1)

        values = [
            np.log1p(length),
            float(length > 1022),
        ]

        values.extend([
            sequence.count(aa) / length
            for aa in AA
        ])

        values.extend([
            sum(sequence.count(a) for a in "AILMFWV") / length,
            sum(sequence.count(a) for a in "DE") / length,
            sum(sequence.count(a) for a in "KRH") / length,
            sum(sequence.count(a) for a in "STNQ") / length,
            sum(sequence.count(a) for a in "GP") / length,
        ])

        rows.append(values)

    return np.asarray(rows, dtype=np.float64)


def load_data():
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

    if df["gene_id"].duplicated().any():
        raise ValueError("Duplicate gene IDs")

    return df


def load_features(df):
    features = {
        "Sequence_Composition": sequence_features(
            df["sequence"].tolist()
        )
    }

    gene_ids = df["gene_id"].tolist()

    for model in MODELS[1:]:
        vector_path = EMBEDDINGS / f"{model}.npy"
        gene_path = EMBEDDINGS / f"{model}_genes.json"

        if not vector_path.exists() or not gene_path.exists():
            raise FileNotFoundError(
                f"Missing cached embeddings for {model}"
            )

        saved_ids = json.loads(
            gene_path.read_text(encoding="utf-8")
        )

        if saved_ids != gene_ids:
            raise ValueError(
                f"Gene order mismatch for {model}"
            )

        vectors = np.load(vector_path).astype(
            np.float64
        )

        if len(vectors) != len(df):
            raise ValueError(
                f"Embedding count mismatch: {model}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                f"Nonfinite embeddings: {model}"
            )

        features[model] = vectors

        log(f"Loaded {model}: {vectors.shape}")

    return features


def build_similarity_groups(df):
    path = PROCESSED / "approx_sequence_groups.csv"
    metadata_path = TABLES / "sequence_group_manifest.json"

    sequences = df["sequence"].tolist()
    genes = df["gene_id"].tolist()

    sequence_hash = hashlib.sha256(
        "\n".join(
            f"{g}:{s}" for g, s in zip(genes, sequences)
        ).encode("utf-8")
    ).hexdigest()

    if path.exists() and metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        saved = pd.read_csv(path)

        if (
            metadata.get("sequence_hash") == sequence_hash
            and metadata.get("kmer_size") == KMER_SIZE
            and metadata.get("neighbors") == NEIGHBORS
            and metadata.get("cosine_threshold")
                == COSINE_THRESHOLD
            and saved["gene_id"].tolist() == genes
        ):
            log("Reusing saved approximate sequence groups")
            return saved["group"].to_numpy()

    log("Building 3-mer TF-IDF sequence representations")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(KMER_SIZE, KMER_SIZE),
        lowercase=False,
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(sequences)

    X = normalize(X, norm="l2")

    log("Finding approximate sequence neighbors")

    nn = NearestNeighbors(
        n_neighbors=min(NEIGHBORS + 1, len(df)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    rows = []
    cols = []

    for i in range(len(df)):
        for distance, j in zip(
            distances[i], indices[i]
        ):
            if i == j:
                continue

            similarity = 1.0 - distance

            if similarity >= COSINE_THRESHOLD:
                rows.append(i)
                cols.append(int(j))

    adjacency = csr_matrix(
        (
            np.ones(len(rows), dtype=np.int8),
            (rows, cols),
        ),
        shape=(len(df), len(df)),
    )

    adjacency = adjacency.maximum(adjacency.T)

    n_groups, labels = connected_components(
        adjacency,
        directed=False,
    )

    result = pd.DataFrame({
        "gene_id": genes,
        "group": labels,
    })

    result.to_csv(path, index=False)

    sizes = pd.Series(labels).value_counts()

    metadata = {
        "sequence_hash": sequence_hash,
        "kmer_size": KMER_SIZE,
        "neighbors": NEIGHBORS,
        "cosine_threshold": COSINE_THRESHOLD,
        "n_groups": int(n_groups),
        "largest_group": int(sizes.max()),
        "singleton_groups": int((sizes == 1).sum()),
        "method": "TF-IDF 3-mer cosine nearest-neighbor graph",
        "warning": (
            "This is an approximate sequence-composition "
            "similarity sensitivity analysis. It does not "
            "establish absence of protein homology."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    log(
        f"Sequence groups: {n_groups}; "
        f"largest: {sizes.max()}; "
        f"singletons: {(sizes == 1).sum()}"
    )

    return labels


def make_splits(df, groups):
    splits = []
    indices = np.arange(len(df))

    for repeat in range(REPEATS):
        train, test = train_test_split(
            indices,
            test_size=TEST_FRACTION,
            random_state=SEED + repeat,
        )

        splits.append((
            "random_gene",
            repeat,
            train,
            test,
        ))

    group_splitter = GroupShuffleSplit(
        n_splits=REPEATS,
        test_size=TEST_FRACTION,
        random_state=SEED,
    )

    for repeat, (train, test) in enumerate(
        group_splitter.split(indices, groups=groups)
    ):
        if len(np.unique(groups[train])) == 0:
            raise ValueError("Empty training groups")

        if set(groups[train]) & set(groups[test]):
            raise ValueError("Sequence group leakage")

        splits.append((
            "approx_similarity_group",
            repeat,
            train,
            test,
        ))

    return splits


def tune_ridge(X, y, seed):
    # Train-only inner CV. The test set is never used
    # for selecting the regularization strength.
    inner = KFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )

    best_alpha = None
    best_score = float("inf")

    for alpha in ALPHAS:
        errors = []

        for inner_train, inner_valid in inner.split(X):
            estimator = make_pipeline(
                StandardScaler(),
                Ridge(
                    alpha=float(alpha),
                    solver="lsqr",
                    max_iter=5000,
                ),
            )

            estimator.fit(
                X[inner_train],
                y[inner_train],
            )

            prediction = estimator.predict(
                X[inner_valid]
            )

            errors.append(
                np.mean(
                    (y[inner_valid] - prediction) ** 2
                )
            )

        score = float(np.mean(errors))

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha


def score_predictions(y_true, predicted):
    if (
        np.std(y_true) == 0
        or np.std(predicted) == 0
    ):
        correlation = np.nan
    else:
        correlation = float(
            spearmanr(y_true, predicted).statistic
        )

    return {
        "spearman": correlation,
        "r2": float(r2_score(y_true, predicted)),
        "mae": float(
            mean_absolute_error(y_true, predicted)
        ),
    }


def evaluate_all(df, features, splits):
    rows = []
    prediction_rows = []
    split_rows = []

    for split_type, repeat, train, test in splits:
        log(
            f"\nSplit: {split_type}, repeat {repeat + 1}, "
            f"train={len(train)}, test={len(test)}"
        )

        split_rows.extend([
            {
                "split_type": split_type,
                "repeat": repeat,
                "gene_id": df.iloc[i]["gene_id"],
                "split": "train",
            }
            for i in train
        ])

        split_rows.extend([
            {
                "split_type": split_type,
                "repeat": repeat,
                "gene_id": df.iloc[i]["gene_id"],
                "split": "test",
            }
            for i in test
        ])

        for target in TARGETS:
            y = df[target].to_numpy(dtype=float)

            for model_name, X in features.items():
                alpha = tune_ridge(
                    X[train],
                    y[train],
                    SEED + repeat,
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

                metrics = score_predictions(
                    y[test],
                    predicted,
                )

                rows.append({
                    "split_type": split_type,
                    "repeat": repeat,
                    "target": target,
                    "model": model_name,
                    "alpha": alpha,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                })

                for index, prediction in zip(
                    test, predicted
                ):
                    prediction_rows.append({
                        "split_type": split_type,
                        "repeat": repeat,
                        "target": target,
                        "model": model_name,
                        "gene_id": df.iloc[index]["gene_id"],
                        "observed": y[index],
                        "predicted": float(prediction),
                    })

                log(
                    f"  {target} | {model_name}: "
                    f"Spearman={metrics['spearman']:.4f}"
                )

        # Checkpoint after every completed split.
        pd.DataFrame(rows).to_csv(
            TABLES / "generalization_results_partial.csv",
            index=False,
        )

    pd.DataFrame(split_rows).to_csv(
        PROCESSED / "generalization_splits.csv",
        index=False,
    )

    pd.DataFrame(prediction_rows).to_parquet(
        TABLES / "generalization_predictions.parquet",
        index=False,
    )

    return pd.DataFrame(rows)


def summarize(results):
    summary = (
        results.groupby(
            ["split_type", "target", "model"]
        )
        .agg(
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_r2=("r2", "mean"),
            mean_mae=("mae", "mean"),
            mean_test_genes=("n_test", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES / "generalization_summary.csv",
        index=False,
    )

    for target in TARGETS:
        subset = summary[
            summary["target"] == target
        ]

        fig, ax = plt.subplots(figsize=(11, 5))

        for split_type in subset["split_type"].unique():
            part = subset[
                subset["split_type"] == split_type
            ].set_index("model").reindex(MODELS)

            ax.plot(
                MODELS,
                part["mean_spearman"],
                marker="o",
                label=split_type,
            )

        ax.set_title(
            f"Generalization sensitivity: {target}"
        )
        ax.set_ylabel("Mean test Spearman")
        ax.set_xlabel("Model")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()

        fig.savefig(
            FIGURES / f"generalization_{target}.png",
            dpi=300,
        )

        plt.close(fig)

    return summary


def main():
    log("=" * 65)
    log("FAIZAN PAPER 3 — GENERALIZATION EVALUATION")
    log("=" * 65)

    df = load_data()

    log(f"Genes: {len(df)}")

    features = load_features(df)

    groups = build_similarity_groups(df)

    splits = make_splits(df, groups)

    log(f"Total splits: {len(splits)}")

    manifest = {
        "seed": SEED,
        "repeats": REPEATS,
        "test_fraction": TEST_FRACTION,
        "models": MODELS,
        "targets": TARGETS,
        "sequence_group_method": (
            "3-mer TF-IDF cosine graph, approximate only"
        ),
        "sequence_similarity_threshold": COSINE_THRESHOLD,
        "ridge_solver": "lsqr",
        "ridge_alpha_selection": (
            "training-only 3-fold CV"
        ),
        "limitations": [
            "Sequence groups are not validated homology clusters",
            "Nearest-neighbor search may miss homologous pairs",
            "Inner CV is not group-aware",
            "Only five repeated splits are evaluated",
            "Saved ESM embeddings truncate long proteins",
            "Experimental fitness labels remain provisional",
        ],
        "python_version": platform.python_version(),
    }

    (TABLES / "generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    results = evaluate_all(
        df, features, splits
    )

    results.to_csv(
        TABLES / "generalization_results.csv",
        index=False,
    )

    summary = summarize(results)

    print("\nGENERALIZATION SUMMARY")
    print(
        summary.round(4).to_string(index=False)
    )

    log("\nGENERALIZATION EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
