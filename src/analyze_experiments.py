
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 5000
CONFIDENCE = 0.95

MODELS = [
    "Sequence_Composition",
    "ESM2_8M",
    "ESM2_35M",
    "ESM2_150M",
    "ESM2_650M",
]

TARGETS = [
    "HOP_Phase_I_mean",
    "HOP_Phase_II_mean",
]

COMPARISONS = [
    ("ESM2_8M", "Sequence_Composition"),
    ("ESM2_35M", "Sequence_Composition"),
    ("ESM2_150M", "Sequence_Composition"),
    ("ESM2_650M", "Sequence_Composition"),
    ("ESM2_150M", "ESM2_35M"),
    ("ESM2_650M", "ESM2_150M"),
]


def spearman_fast(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(
        np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    )


def load_predictions(target):
    frames = []

    for model in MODELS:
        path = TABLES / f"predictions_{model}_{target}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        required = {"gene_id", "observed", "predicted"}

        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {path}")

        if df["gene_id"].duplicated().any():
            raise ValueError(f"Duplicate gene IDs in {path}")

        df = df[["gene_id", "observed", "predicted"]]
        df = df.rename(columns={
            "observed": f"observed_{model}",
            "predicted": model,
        })

        frames.append(df)

    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on="gene_id",
            how="inner",
            validate="one_to_one",
        )

    if len(merged) != 899:
        raise ValueError(
            f"Expected 899 test genes, found {len(merged)}"
        )

    reference = merged[f"observed_{MODELS[0]}"].to_numpy()

    for model in MODELS[1:]:
        observed = merged[f"observed_{model}"].to_numpy()

        if not np.allclose(
            reference, observed, rtol=0, atol=1e-10
        ):
            raise ValueError(
                f"Observed targets differ for {model}"
            )

    columns = ["gene_id", f"observed_{MODELS[0]}"] + MODELS
    merged = merged[columns].sort_values("gene_id")

    if not np.isfinite(
        merged.drop(columns=["gene_id"]).to_numpy()
    ).all():
        raise ValueError("Nonfinite predictions or labels")

    return merged.reset_index(drop=True)


def bootstrap_target(df, target):
    print(f"\nAnalyzing {target}")

    y = df[f"observed_{MODELS[0]}"].to_numpy()
    predictions = {
        model: df[model].to_numpy()
        for model in MODELS
    }

    observed_scores = {
        model: spearman_fast(y, predictions[model])
        for model in MODELS
    }

    rng = np.random.default_rng(SEED)
    n = len(y)

    samples = {
        model: np.empty(N_BOOTSTRAP, dtype=float)
        for model in MODELS
    }

    for iteration in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_sample = y[idx]

        for model in MODELS:
            samples[model][iteration] = spearman_fast(
                y_sample, predictions[model][idx]
            )

        if (iteration + 1) % 1000 == 0:
            print(
                f"  Bootstrap: {iteration + 1}/{N_BOOTSTRAP}"
            )

    alpha = (1 - CONFIDENCE) / 2

    score_rows = []

    for model in MODELS:
        values = samples[model]
        values = values[np.isfinite(values)]

        if len(values) < N_BOOTSTRAP * 0.95:
            raise ValueError(
                f"Too many invalid bootstrap values: {model}"
            )

        low, high = np.quantile(
            values, [alpha, 1 - alpha]
        )

        score_rows.append({
            "target": target,
            "model": model,
            "spearman": observed_scores[model],
            "ci_low": float(low),
            "ci_high": float(high),
            "n_test": n,
            "bootstrap_repetitions": N_BOOTSTRAP,
        })

    comparison_rows = []

    for model_a, model_b in COMPARISONS:
        differences = (
            samples[model_a] - samples[model_b]
        )

        differences = differences[
            np.isfinite(differences)
        ]

        low, high = np.quantile(
            differences, [alpha, 1 - alpha]
        )

        observed_difference = (
            observed_scores[model_a]
            - observed_scores[model_b]
        )

        # Descriptive bootstrap tail proportion.
        # This is NOT a permutation-test p-value.
        fraction_nonpositive = np.mean(differences <= 0)
        fraction_nonnegative = np.mean(differences >= 0)

        comparison_rows.append({
            "target": target,
            "model_a": model_a,
            "model_b": model_b,
            "delta_spearman": observed_difference,
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_two_sided_tail_fraction": float(
                min(
                    1.0,
                    2 * min(
                        fraction_nonpositive,
                        fraction_nonnegative,
                    )
                )
            ),
            "ci_excludes_zero": bool(
                low > 0 or high < 0
            ),
            "n_test": n,
        })

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(comparison_rows),
    )


def plot_results(scores):
    for target in TARGETS:
        subset = scores[
            scores["target"] == target
        ].copy()

        subset = subset.set_index("model").loc[MODELS]
        subset = subset.reset_index()

        x = np.arange(len(subset))
        y = subset["spearman"].to_numpy()

        lower = y - subset["ci_low"].to_numpy()
        upper = subset["ci_high"].to_numpy() - y

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=5,
        )

        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["model"],
            rotation=30,
            ha="right",
        )

        ax.set_ylabel("Spearman correlation")
        ax.set_title(
            f"{target}: 95% bootstrap confidence intervals"
        )

        fig.tight_layout()

        path = FIGURES / f"bootstrap_ci_{target}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)


def main():
    print("=" * 65)
    print("FAIZAN PAPER 3 — STATISTICAL VALIDATION")
    print("=" * 65)

    all_scores = []
    all_comparisons = []

    for target in TARGETS:
        df = load_predictions(target)

        scores, comparisons = bootstrap_target(
            df, target
        )

        all_scores.append(scores)
        all_comparisons.append(comparisons)

    scores = pd.concat(
        all_scores, ignore_index=True
    )

    comparisons = pd.concat(
        all_comparisons, ignore_index=True
    )

    scores.to_csv(
        TABLES / "bootstrap_model_scores.csv",
        index=False,
    )

    comparisons.to_csv(
        TABLES / "bootstrap_model_comparisons.csv",
        index=False,
    )

    plot_results(scores)

    manifest = {
        "seed": SEED,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "confidence_level": CONFIDENCE,
        "resampling_unit": "held_out_gene",
        "method": "paired_percentile_bootstrap",
        "targets": TARGETS,
        "models": MODELS,
        "limitations": [
            "Random gene split does not control sequence homology",
            "Confidence intervals are conditional on the saved split",
            "Only one training split and seed are evaluated",
            "Bootstrap genes may not be biologically independent",
            "No multiple-comparison adjustment is performed",
            "Bootstrap tail fractions are not formal p-values",
            "Fitness targets remain provisional",
        ],
    }

    (TABLES / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL CONFIDENCE INTERVALS")
    print(
        scores[
            ["target", "model", "spearman",
             "ci_low", "ci_high"]
        ].round(4).to_string(index=False)
    )

    print("\nPAIRED MODEL COMPARISONS")
    print(
        comparisons[
            ["target", "model_a", "model_b",
             "delta_spearman", "ci_low",
             "ci_high", "ci_excludes_zero"]
        ].round(4).to_string(index=False)
    )

    print("\nSaved statistical tables and figures.")
    print("STATUS: Statistical analysis completed.")


if __name__ == "__main__":
    main()
