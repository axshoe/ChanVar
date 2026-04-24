# ChanVar

**In Silico Functional Annotation of Rare CACNA1A Variants in Familial Hemiplegic Migraine**

Part of the **MiSOF** (Migraine Stratification Outcomes Framework) series.  
[The Xiu Lab](https://thexiulab.org) · [GitHub: axshoe](https://github.com/axshoe/ChanVar)

---

## Overview

ChanVar is a Python pipeline for structural and evolutionary annotation of rare missense variants in *CACNA1A*, the gene encoding the α1A subunit of P/Q-type voltage-gated calcium channels. It produces a Composite Pathogenicity Score (CPS) per variant by integrating eight biophysical and population-genetic features, assigns domain-aware classification, and generates publication-quality reports and figures.

ChanVar is the foundational annotation layer for the MiSOF pipeline, providing variant-level functional context that downstream projects (TraitStrata, SigVigil, NeuroTrack) can reference when modeling phenotypic heterogeneity and treatment outcomes in channelopathy-associated migraine.

## Features

- gnomAD v4 allele frequency retrieval (GraphQL + VCF batch mode)
- ClinVar pathogenicity training set construction (4-star minimum configurable)
- AlphaFold2 / cryo-EM structure download and domain mapping (all 24 transmembrane segments, EEEE selectivity filter, I-II/II-III/III-IV linkers)
- FoldX / EvoEF2 ΔΔG thermodynamic stability estimation with TM-domain reliability flags
- Local structural RMSD computation (±10 residue Cα window, BioPython Superimposer)
- Jensen-Shannon divergence conservation scoring from 87-vertebrate CACNA1A alignment
- ClinVar ACMG evidence integration (PS3/BS3, PM2, PP3/BP4 evidence codes)
- CADD/AlphaMissense/REVEL integration for ensemble meta-score features
- Bootstrap confidence intervals for CPS (1000 iterations, feature-specific noise models)
- Reports: JSON (full feature dump), Markdown (table + citations), HTML (visual CPS bar + contribution plots)
- Five publication-quality figures (domain architecture, domain-stratified CPS, ROC curves, forest plot, interactive 3D structure viewer)
- CLI: `chanvar annotate`, `chanvar batch`, `chanvar train`, `chanvar validate`

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Single variant
chanvar annotate --variant "R192Q" --output results/

# Batch mode (TSV: one HGVS per line)
chanvar batch --input variants.tsv --output results/ --format html

# Train logistic weights on ClinVar gold standard
chanvar train --min-stars 2 --output weights.json

# Validate against held-out ClinVar set
chanvar validate --weights weights.json --output validation/
```

See [SETUP.md](SETUP.md) for complete installation, optional dependency configuration (FoldX, EvoEF2, MUSCLE/MAFFT), and environment setup.

## Project Structure

```
chanvar/
├── data/               # gnomAD, ClinVar, AlphaFold clients; Grantham matrix; conservation TSV
├── structure/          # ΔΔG stability, local RMSD
├── evolution/          # Conservation scoring, MSA alignment wrappers
├── scoring/            # Feature assembly, CPS computation, bootstrap CI
├── report/             # JSON/Markdown/HTML report generators
├── viz/                # Five publication figures + interactive 3D viewer
└── cli.py              # Entry-point commands
tests/
figures/                # Generated figure output directory
```

## Citation

If you use ChanVar in research, please cite:

> Xiu, A. (2026). *ChanVar: In Silico Functional Annotation of Rare CACNA1A Variants in Familial Hemiplegic Migraine*. The Xiu Lab. https://github.com/axshoe/ChanVar

And the primary methods papers listed in `chanvar/report/generator.py → CITATIONS`.

## License

MIT — see LICENSE.

---

*The Xiu Lab · thexiulab.org · github.com/axshoe*
