
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FINAL = RESULTS / "final_experiments"
FIGURES = ROOT / "figures" / "final"

FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Sequence_Composition",
    "ESM2_8M",
    "ESM2_35M",
    "ESM2_150M",
    "ESM2_650M",
]

LABELS = {
    "Sequence_Composition": "Composition",
    "ESM2_8M": "ESM-2 8M",
    "ESM2_35M": "ESM-2 35M",
    "ESM2_150M": "ESM-2 150M",
    "ESM2_650M": "ESM-2 650M",
}

COLORS = {
    "Sequence_Composition": "#7F8C8D",
    "ESM2_8M": "#3498DB",
    "ESM2_35M": "#2E86C1",
    "ESM2_150M": "#1F618D",
    "ESM2_650M": "#154360",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save(name):
    for extension in ("png", "pdf"):
        plt.savefig(
            FIGURES / f"{name}.{extension}",
            bbox_inches="tight",
        )
    plt.close()
    print("Saved:", name, flush=True)


def plot_fermentation():
    df = pd.read_csv(TABLES / "homology_results.csv")

    print("Fermentation columns:", df.columns.tolist())

    target_col = "target"
    model_col = "model"
    metric_col = "spearman"

    required = {target_col, model_col, metric_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing fermentation columns: {required - set(df.columns)}"
        )

    df = df[df[model_col].isin(MODELS)]

    grouped = (
        df.groupby([target_col, model_col])[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    targets = [
        "HOP_Phase_I_mean",
        "HOP_Phase_II_mean",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(targets))
    width = 0.15

    for i, model in enumerate(MODELS):
        means = []
        stds = []

        for target in targets:
            sub = grouped[
                (grouped[target_col] == target)
                & (grouped[model_col] == model)
            ]

            means.append(
                sub["mean"].iloc[0] if len(sub) else np.nan
            )
            stds.append(
                sub["std"].iloc[0] if len(sub) else np.nan
            )

        ax.bar(
            x + (i - 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=LABELS[model],
            color=COLORS[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Fermentation Phase I", "Fermentation Phase II"])
    ax.set_ylabel("Held-out Spearman correlation")
    ax.set_title("Homology-controlled fermentation prediction")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)

    save("figure_01_fermentation")


def plot_chemical():
    df = pd.read_csv(
        FINAL / "chemical_prediction_results.csv"
    )

    required = {"model", "spearman", "condition"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing chemical columns: {required - set(df.columns)}"
        )

    df = df[df["model"].isin(MODELS)]

    condition_means = (
        df.groupby(["condition", "model"])["spearman"]
        .mean()
        .unstack("model")
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    values = [
        condition_means[m].dropna().values
        for m in MODELS
    ]

    bp = ax.boxplot(
        values,
        tick_labels=[LABELS[m] for m in MODELS],
        patch_artist=True,
        showfliers=False,
    )

    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Spearman correlation")
    ax.set_title(
        "Chemical-response prediction across 60 selected conditions"
    )
    ax.tick_params(axis="x", rotation=20)

    save("figure_02_chemical")


def plot_scaling():
    fermentation = pd.read_csv(
        TABLES / "homology_results.csv"
    )
    chemical = pd.read_csv(
        FINAL / "chemical_prediction_results.csv"
    )

    sizes = [8, 35, 150, 650]
    names = [f"ESM2_{n}M" for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))

    for target, label, marker in [
        ("HOP_Phase_I_mean", "Fermentation I", "o"),
        ("HOP_Phase_II_mean", "Fermentation II", "s"),
    ]:
        means = (
            fermentation[
                fermentation["target"] == target
            ]
            .groupby("model")["spearman"]
            .mean()
        )

        ax.plot(
            sizes,
            [means.get(m, np.nan) for m in names],
            marker=marker,
            linewidth=2,
            label=label,
        )

    chemical_means = (
        chemical.groupby("model")["spearman"].mean()
    )

    ax.plot(
        sizes,
        [chemical_means.get(m, np.nan) for m in names],
        marker="^",
        linewidth=2,
        label="Chemical pilot",
    )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}M" for n in sizes])
    ax.set_xlabel("ESM-2 model parameters")
    ax.set_ylabel("Mean held-out Spearman correlation")
    ax.set_title("Model scaling across prediction tasks")
    ax.legend(frameon=False)

    save("figure_03_scaling")


def plot_transfer():
    df = pd.read_csv(
        FINAL / "cross_study_transfer_bootstrap.csv"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )

    targets = [
        ("HOP_Phase_I_mean", "Fermentation Phase I"),
        ("HOP_Phase_II_mean", "Fermentation Phase II"),
    ]

    for ax, (target, label) in zip(axes, targets):
        subset = (
            df[df["target"] == target]
            .set_index("model")
            .reindex(MODELS)
        )

        x = np.arange(len(MODELS))
        means = subset["mean"].to_numpy(dtype=float)
        lower = means - subset["ci_low"].to_numpy(dtype=float)
        upper = subset["ci_high"].to_numpy(dtype=float) - means

        ax.bar(
            x,
            means,
            color=[COLORS[m] for m in MODELS],
            width=0.65,
        )

        ax.errorbar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color="black",
            capsize=3,
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[m] for m in MODELS],
            rotation=35,
            ha="right",
        )
        ax.set_title(label)
        ax.set_ylabel("Transfer Spearman correlation")

    fig.suptitle(
        "Fermentation-trained predictions versus chemical responses"
    )
    fig.tight_layout()

    save("figure_04_transfer")


def main():
    plot_fermentation()
    plot_chemical()
    plot_scaling()
    plot_transfer()

    manifest = {
        "figures": [
            "figure_01_fermentation",
            "figure_02_chemical",
            "figure_03_scaling",
            "figure_04_transfer",
        ],
        "formats": ["png", "pdf"],
        "source_files": [
            str(TABLES / "homology_results.csv"),
            str(FINAL / "chemical_prediction_results.csv"),
            str(FINAL / "cross_study_transfer_bootstrap.csv"),
        ],
        "notes": [
            "Fermentation error bars show across-split standard deviations.",
            "Chemical boxplots summarize condition-level means.",
            "Transfer confidence intervals use condition-level bootstrap.",
            "Transfer results represent associations, not chemical-model calibration.",
            "The selected chemical conditions may be dependent.",
        ],
    }

    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nAll four figures generated successfully.")


if __name__ == "__main__":
    main()
