# When Do Pretrained Protein Models Improve Yeast Gene-Deletion Fitness Prediction?

### A Cross-Study Evaluation Across Fermentation and Chemical Perturbations

Research code and experimental results evaluating pretrained
protein foundation models for yeast gene-deletion fitness prediction.

## Research objective

This study examines when pretrained protein representations
improve prediction of yeast gene-deletion fitness relative to
simpler baseline approaches.

The experiments evaluate predictive performance,
generalization, homology-controlled validation,
model scaling, chemical perturbations, and cross-study transfer.

## Pretrained protein models

The study evaluates four ESM-2 models:

| Model | Hugging Face identifier |
|---|---|
| ESM-2 8M | facebook/esm2_t6_8M_UR50D |
| ESM-2 35M | facebook/esm2_t12_35M_UR50D |
| ESM-2 150M | facebook/esm2_t30_150M_UR50D |
| ESM-2 650M | facebook/esm2_t33_650M_UR50D |

Frozen pretrained protein embeddings are evaluated against
simple baseline models.

## Research datasets

The study uses public yeast fitness and protein-sequence
resources, including:

- Yeast Phenome
- NCBI GEO GSE46145
- UniProt yeast protein sequences
- Chemical perturbation response datasets

See `data/README.md` for dataset provenance and checksums.

## Experimental evaluation

The repository includes:

1. Data source auditing and preprocessing
2. Protein mapping and research cohort preparation
3. Baseline and pretrained ESM-2 experiments
4. Statistical analysis and confidence intervals
5. Generalization evaluation
6. Homology-controlled validation
7. Chemical-response experiments
8. Cross-study transfer analysis
9. Publication figure generation

## Original computing environment

- Python 3.12.4
- PyTorch 2.9.1+cu128
- CUDA-enabled NVIDIA GeForce RTX 3080
- Transformers 4.57.6
- NumPy 2.5.4
- scikit-learn 1.9.1

## Installation

Create a Python 3.12 virtual environment and install
dependencies from `requirements.txt`.

CUDA-enabled PyTorch may require installation through
the appropriate official PyTorch wheel index.

MMseqs2 is used for homology-related analyses and
must be installed separately where required.

## Reproduction

The main entry point is:

    python run_all.py --mode standard

This mode reuses existing ESM-2 embeddings and reruns
the research analyses.

To regenerate embeddings and rerun analyses:

    python run_all.py --mode full

Full mode may require substantial computation and
the original local research datasets.

The pipeline checks required inputs, records execution
logs, and verifies expected output files.

IMPORTANT: A complete independent reproduction from
a fresh machine has not yet been verified.

The repository excludes raw datasets, processed
intermediates, model weights, and large embeddings.

Therefore, cloning the repository alone is not sufficient
to run the full pipeline.

## Repository structure

- `src/` — Research and evaluation scripts
- `run_all.py` — Master reproduction runner
- `data/README.md` — Dataset documentation
- `results/tables/` — Evaluation results
- `results/final_experiments/` — Chemical and transfer analyses
- `figures/final/` — Publication figures
- `manuscript/` — Markdown manuscript
- `requirements.txt` — Python dependencies

## Scientific limitations

- Some biological fitness labels remain provisional.
- Long protein sequences are truncated for ESM-2 inference.
- Random gene splits do not control protein homology.
- Homology-aware evaluation uses MMseqs2 clusters.
- Chemical-response and fermentation outcomes have
  different biological endpoint definitions.
- Cross-study associations should not be interpreted
  as proof of causal biological transfer.
- Independent reproduction remains to be tested.

## Research status

This repository provides the code, saved experimental
summaries, publication figures, and documented limitations
for the research project.

It is intended to support transparent scientific review
and reproducibility.
