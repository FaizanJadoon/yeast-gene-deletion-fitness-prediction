
from pathlib import Path
import os
import gc
import json
import random
import hashlib
import platform
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
EMBEDDINGS = RESULTS / "embeddings"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"

for directory in [
    TABLES, FIGURES, EMBEDDINGS, LOGS, MODELS
]:
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20
MAX_RESIDUES = 1022
ALPHAS = np.logspace(-3, 5, 15)

MODEL_LIST = {
    "ESM2_8M": "facebook/esm2_t6_8M_UR50D",
    "ESM2_35M": "facebook/esm2_t12_35M_UR50D",
    "ESM2_150M": "facebook/esm2_t30_150M_UR50D",
    "ESM2_650M": "facebook/esm2_t33_650M_UR50D",
}

TARGETS = [
    "HOP_Phase_I_mean",
    "HOP_Phase_II_mean",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HOME"] = str(MODELS / "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def save_json(data, path):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    with (LOGS / "experiments.log").open(
        "a", encoding="utf-8"
    ) as f:
        f.write(line + "\n")


# ============================================================
# DATA
# ============================================================

def load_research_data():
    log("Loading validated research cohort")

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
        validate="one_to_one"
    )

    df = df.dropna(
        subset=["sequence"] + TARGETS
    ).copy()

    df["sequence"] = (
        df["sequence"].astype(str).str.upper()
    )

    df["length"] = df["sequence"].str.len()

    if df["gene_id"].duplicated().any():
        raise ValueError("Duplicate gene IDs detected")

    if not np.isfinite(df[TARGETS].to_numpy()).all():
        raise ValueError("Nonfinite fitness targets detected")

    df = df.sort_values("gene_id").reset_index(drop=True)

    log(f"Research genes: {len(df)}")

    return df


def create_splits(df):
    path = PROCESSED / "experiment_gene_splits.csv"

    if path.exists():
        splits = pd.read_csv(path)

        if set(splits["gene_id"]) != set(df["gene_id"]):
            raise ValueError(
                "Saved split does not match current cohort"
            )

        log("Reusing saved gene-held-out split")
        return splits

    train_ids, test_ids = train_test_split(
        df["gene_id"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    train_set = set(train_ids)

    splits = pd.DataFrame({
        "gene_id": df["gene_id"],
        "split": [
            "train" if g in train_set else "test"
            for g in df["gene_id"]
        ]
    })

    splits.to_csv(path, index=False)

    log(
        f"Created split: {len(train_ids)} train, "
        f"{len(test_ids)} test"
    )

    return splits


# ============================================================
# SEQUENCE BASELINES
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sequence_features(sequences):
    rows = []

    for sequence in sequences:
        length = max(len(sequence), 1)

        features = [
            np.log1p(length),
            float(length > MAX_RESIDUES),
        ]

        for aa in AMINO_ACIDS:
            features.append(sequence.count(aa) / length)

        # Simple physicochemical composition proxies.
        features.extend([
            sum(sequence.count(a) for a in "AILMFWV") / length,
            sum(sequence.count(a) for a in "DE") / length,
            sum(sequence.count(a) for a in "KRH") / length,
            sum(sequence.count(a) for a in "STNQ") / length,
            sum(sequence.count(a) for a in "GP") / length,
        ])

        rows.append(features)

    return np.asarray(rows, dtype=np.float32)


# ============================================================
# ESM-2 EMBEDDINGS
# ============================================================

def embedding_paths(model_name):
    return (
        EMBEDDINGS / f"{model_name}.npy",
        EMBEDDINGS / f"{model_name}_genes.json"
    )


def calculate_embeddings(df, model_name, model_id):
    embedding_file, gene_file = embedding_paths(model_name)

    gene_ids = df["gene_id"].tolist()

    if embedding_file.exists() and gene_file.exists():
        saved_ids = json.loads(
            gene_file.read_text(encoding="utf-8")
        )

        if saved_ids == gene_ids:
            vectors = np.load(embedding_file)

            if (
                vectors.ndim == 2
                and vectors.shape[0] == len(gene_ids)
                and np.isfinite(vectors).all()
            ):
                log(f"Using cached embeddings: {model_name}")
                return vectors

        raise ValueError(
            f"Embedding cache mismatch: {model_name}"
        )

    log(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        add_pooling_layer=False
    )

    model.eval()
    model.to(DEVICE)

    if DEVICE == "cuda":
        model.half()

    # One protein per batch minimizes memory pressure on
    # RTX 3080 10 GB. Increase only after benchmarking.
    vectors = []
    total = len(df)

    for index, sequence in enumerate(df["sequence"]):
        # The first full-length window is used consistently.
        # Truncation is a documented limitation, not an
        # equivalent representation of the full protein.
        sequence = sequence[:MAX_RESIDUES]

        tokens = tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        tokens = {
            key: value.to(DEVICE)
            for key, value in tokens.items()
        }

        with torch.inference_mode():
            output = model(**tokens)

            hidden = output.last_hidden_state[0]

            # ESM uses special beginning/end tokens.
            # Exclude them and all padding from mean pooling.
            residue_count = min(
                len(sequence),
                hidden.shape[0] - 2
            )

            if residue_count <= 0:
                raise ValueError(
                    f"No residues for gene {gene_ids[index]}"
                )

            pooled = hidden[
                1:1 + residue_count
            ].float().mean(dim=0)

            vector = pooled.cpu().numpy()

        vectors.append(vector)

        if (index + 1) % 100 == 0:
            log(
                f"{model_name}: "
                f"{index + 1}/{total} proteins"
            )

    vectors = np.asarray(vectors, dtype=np.float32)

    np.save(embedding_file, vectors)
    gene_file.write_text(
        json.dumps(gene_ids),
        encoding="utf-8"
    )

    log(f"Saved embeddings: {model_name} {vectors.shape}")

    del model, tokenizer

    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return vectors


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.isfinite(y_pred).all():
        raise ValueError("Model produced nonfinite predictions")

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        spearman = np.nan
        pearson = np.nan
    else:
        spearman = float(
            spearmanr(y_true, y_pred).statistic
        )
        pearson = float(
            pearsonr(y_true, y_pred).statistic
        )

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": spearman,
        "Pearson": pearson,
    }


def run_model(
    X,
    y,
    train_mask,
    test_mask,
    model_name,
    target_name,
    gene_ids,
):
    log(f"Training {model_name} on {target_name}")

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    if model_name == "Dummy_Mean":
        estimator = DummyRegressor(strategy="mean")
    else:
        # Alpha selection uses only training data.
        estimator = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=ALPHAS, cv=5)
        )

    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    metrics = evaluate(y_test, predictions)

    metrics.update({
        "model": model_name,
        "target": target_name,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seed": SEED,
        "split_type": "random_gene_held_out",
    })

    predictions_df = pd.DataFrame({
        "gene_id": gene_ids[test_mask],
        "target": target_name,
        "model": model_name,
        "observed": y_test,
        "predicted": predictions,
    })

    prediction_path = (
        TABLES /
        f"predictions_{model_name}_{target_name}.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    log(
        f"{model_name} | {target_name} | "
        f"Spearman={metrics['Spearman']:.4f} | "
        f"R2={metrics['R2']:.4f}"
    )

    return metrics


# ============================================================
# FIGURES
# ============================================================

def create_figures(results):
    for target in TARGETS:
        subset = results[
            results["target"] == target
        ].copy()

        subset = subset.sort_values(
            "Spearman", ascending=False
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            subset["model"],
            subset["Spearman"]
        )

        ax.set_title(
            f"Gene-held-out fitness prediction: {target}"
        )
        ax.set_ylabel("Test Spearman correlation")
        ax.set_xlabel("Model")

        ax.tick_params(axis="x", rotation=35)

        fig.tight_layout()

        fig.savefig(
            FIGURES / f"spearman_{target}.png",
            dpi=300
        )

        plt.close(fig)


# ============================================================
# MASTER EXECUTION
# ============================================================

def main():
    log("=" * 65)
    log("FAIZAN PAPER 3 — FULL MODELING EXPERIMENT")
    log("=" * 65)

    log(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    df = load_research_data()

    splits = create_splits(df)

    df = df.merge(
        splits,
        on="gene_id",
        validate="one_to_one"
    )

    train_mask = (
        df["split"].to_numpy() == "train"
    )

    test_mask = (
        df["split"].to_numpy() == "test"
    )

    gene_ids = df["gene_id"].to_numpy()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "models": MODEL_LIST,
        "targets": TARGETS,
        "max_residues": MAX_RESIDUES,
        "long_sequence_handling": "first_1022_residues",
        "split_type": "random_gene_held_out",
        "label_status": "provisional_barcode_derived",
        "limitations": [
            "Random gene split does not control protein homology",
            "Long protein sequences are truncated",
            "Fitness labels require biological validation",
            "Chemical perturbation transfer not evaluated here",
            "Only frozen pretrained embeddings are evaluated"
        ]
    }

    save_json(
        manifest,
        TABLES / "experiment_manifest.json"
    )

    # Baselines.
    features = sequence_features(
        df["sequence"].tolist()
    )

    feature_sets = {
        "Dummy_Mean": np.zeros(
            (len(df), 1), dtype=np.float32
        ),
        "Sequence_Composition": features,
        "Protein_Length": features[:, :1],
    }

    results = []

    for name, X in feature_sets.items():
        for target in TARGETS:
            result = run_model(
                X,
                df[target].to_numpy(dtype=float),
                train_mask,
                test_mask,
                name,
                target,
                gene_ids,
            )
            results.append(result)

    # Save baseline results before expensive GPU inference.
    pd.DataFrame(results).to_csv(
        TABLES / "model_results_partial.csv",
        index=False
    )

    for model_name, model_id in MODEL_LIST.items():
        vectors = calculate_embeddings(
            df,
            model_name,
            model_id
        )

        for target in TARGETS:
            result = run_model(
                vectors,
                df[target].to_numpy(dtype=float),
                train_mask,
                test_mask,
                model_name,
                target,
                gene_ids,
            )

            results.append(result)

        pd.DataFrame(results).to_csv(
            TABLES / "model_results_partial.csv",
            index=False
        )

        del vectors
        gc.collect()

    final_results = pd.DataFrame(results)

    final_results.to_csv(
        TABLES / "model_results.csv",
        index=False
    )

    create_figures(final_results)

    log("=" * 65)
    log("MODEL EXPERIMENTS COMPLETED")
    log("=" * 65)

    print(
        final_results[
            ["model", "target", "Spearman", "R2", "MAE"]
        ].to_string(index=False)
    )

    log("Results saved to results/tables/")
    log("Figures saved to results/figures/")


if __name__ == "__main__":
    main()
