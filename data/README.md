# Research datasets

This repository studies yeast gene-deletion fitness prediction
using fermentation and chemical perturbation datasets.

Raw datasets and intermediate processed datasets are not
included in the GitHub repository.

## Primary data sources

Yeast Phenome:
https://www.yeastphenome.org/downloads/

NCBI GEO GSE46145:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46145

UniProt yeast protein sequences:
https://www.uniprot.org/

Additional data sources are documented in:
src/audit_data_sources.py

## Original data checksums

Yeast Phenome archive:
e1df66f7a250b6b66a26d6da305d989d168ff214b48d8406f64e0b351d9caa5a

GSE46145 series matrix:
6c5ea178cbec282ba356bf94a768449d30648260ded28d839c5eadbe503b629d

UniProt yeast FASTA:
1aaa81860c8bda68e81e4290703d871c7c8bd12a928038772fbc91b7e17b342a

Processed research cohort:
b2220568c1cd36cbcf0395aef68eb73c13031b24b466a7a95af0032e4a21bfe5

Processed fermentation fitness:
0ffd4952a380344eef344c616464b411cb3f5a0f52951ecda11c364c43a76eba

## Important limitations

The research uses experimental fitness and chemical-response
measurements with dataset-specific endpoint definitions.

Some source labels and their biological interpretations
require further validation.

The held-out splits and homology-control methods are
documented in the experiment manifests and source code.

Reproduction requires the original datasets and their
preprocessing steps.
