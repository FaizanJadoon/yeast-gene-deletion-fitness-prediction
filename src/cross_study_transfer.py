
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "final_experiments"
OUT.mkdir(parents=True, exist_ok=True)

FERMENTATION_FILE = RESULTS / "tables" / "homology_predictions.parquet"
CHEMICAL_FILE = OUT / "chemical_predictions.csv.gz"
CONDITIONS_FILE = OUT / "selected_conditions.csv"

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


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 50:
        return np.nan

    x = x[valid]
    y = y[valid]

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(spearmanr(x, y).statistic)


def main():
    print("Loading existing predictions...", flush=True)

    fermentation = pd.read_parquet(FERMENTATION_FILE)
    chemical = pd.read_csv(
        CHEMICAL_FILE,
        usecols=["condition", "gene_id", "observed"],
    )

    conditions = pd.read_csv(CONDITIONS_FILE)["column"].tolist()

    # Chemical observations are repeated across model/split rows.
    # Deduplicate to one measured score per gene and condition.
    chemical = chemical.drop_duplicates(
        subset=["condition", "gene_id"]
    )

    chemical = chemical[
        chemical["condition"].isin(conditions)
    ]

    fermentation = fermentation[
        fermentation["model"].isin(MODELS)
        & fermentation["target"].isin(TARGETS)
    ].copy()

    # Keep the original held-out predictions from each homology split.
    fermentation["repeat"] = fermentation["repeat"].astype(int)

    merged = fermentation.merge(
        chemical,
        on="gene_id",
        how="inner",
        suffixes=("_fermentation", "_chemical"),
    )

    print(
        "Aligned prediction/measurement rows:",
        len(merged),
        flush=True,
    )

    rows = []

    for (target, model, repeat, condition), group in merged.groupby(
        ["target", "model", "repeat", "condition"],
        sort=False,
    ):
        if len(group) < 50:
            continue

        prediction = group["predicted"].to_numpy(dtype=float)
        chemical_observed = group["observed_chemical"].to_numpy(
            dtype=float
        )

        score = correlation(prediction, chemical_observed)

        rows.append(
            {
                "target": target,
                "model": model,
                "repeat": repeat,
                "condition": condition,
                "n_genes": len(group),
                "transfer_spearman": score,
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No valid transfer comparisons")

    results.to_csv(
        OUT / "cross_study_transfer_results.csv",
        index=False,
    )

    summary = (
        results.groupby(["target", "model"], as_index=False)
        .agg(
            mean_transfer_spearman=("transfer_spearman", "mean"),
            median_transfer_spearman=("transfer_spearman", "median"),
            sd_transfer_spearman=("transfer_spearman", "std"),
            evaluations=("transfer_spearman", "count"),
        )
        .sort_values(["target", "mean_transfer_spearman"],
                     ascending=[True, False])
    )

    summary.to_csv(
        OUT / "cross_study_transfer_summary.csv",
        index=False,
    )

    # Bootstrap whole conditions rather than treating overlapping
    # held-out gene splits as independent observations.
    condition_means = (
        results.groupby(
            ["target", "model", "condition"],
            as_index=False,
        )["transfer_spearman"]
        .mean()
    )

    rng = np.random.default_rng(2026)
    intervals = []

    for (target, model), group in condition_means.groupby(
        ["target", "model"]
    ):
        values = group["transfer_spearman"].dropna().to_numpy()

        if not len(values):
            continue

        samples = rng.choice(
            values,
            size=(5000, len(values)),
            replace=True,
        ).mean(axis=1)

        low, high = np.quantile(samples, [0.025, 0.975])

        intervals.append(
            {
                "target": target,
                "model": model,
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "conditions": len(values),
            }
        )

    ci = pd.DataFrame(intervals)
    ci.to_csv(
        OUT / "cross_study_transfer_bootstrap.csv",
        index=False,
    )

    config = {
        "seed": 2026,
        "bootstrap_resamples": 5000,
        "analysis": "Out-of-sample fermentation prediction versus observed chemical response",
        "fermentation_predictions": str(FERMENTATION_FILE),
        "chemical_observations": str(CHEMICAL_FILE),
        "conditions": len(conditions),
        "models": MODELS,
        "targets": TARGETS,
        "limitations": [
            "Chemical conditions may not be statistically independent.",
            "Only the previously selected 60-condition pilot is evaluated.",
            "Fermentation and chemical response have different endpoint definitions.",
            "No model is trained on chemical outcomes in this transfer analysis.",
        ],
    }

    (OUT / "cross_study_transfer_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nCROSS-STUDY TRANSFER RESULTS")
    print(summary.round(4).to_string(index=False))

    print("\nBOOTSTRAP CONFIDENCE INTERVALS")
    print(ci.round(4).to_string(index=False))

    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
