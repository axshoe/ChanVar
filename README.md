# ChanVar

**In Silico Functional Annotation of Rare CACNA1A Variants in Familial Hemiplegic Migraine**

Part of the **MiSOF** (Migraine Stratification Outcomes Framework) series.  
[The Xiu Lab](https://thexiulab.org) · [GitHub: axshoe](https://github.com/axshoe/ChanVar)

---

## Overview

ChanVar is a Python pipeline for structural and evolutionary annotation of rare missense variants in *CACNA1A*, the gene encoding the α1A subunit of P/Q-type voltage-gated calcium channels. It produces a Composite Pathogenicity Score (CPS) per variant by integrating eight biophysical and population-genetic features, assigns domain-aware classification, and generates publication-quality reports and figures.

ChanVar is the foundational annotation layer for the MiSOF pipeline, providing variant-level functional context that downstream projects (TraitStrata, SigVigil, NeuroTrack) can reference when modeling phenotypic heterogeneity and treatment outcomes in channelopathy-associated migraine.

## Benchmark Results

Three known pathogenic FHM1/EA2 variants, scored at 62.5% data completeness (5/8 features; FoldX ΔΔG and AlphaMissense not yet integrated):

| Variant | Domain | CPS | 95% CI | Classification |
|---------|--------|-----|--------|----------------|
| R192Q | Voltage sensor S4-I | 0.791 | 0.725–0.831 | Possibly Pathogenic |
| S218L | Voltage sensor S4-I | 0.831 | 0.765–0.873 | Possibly Pathogenic |
| T666M | Pore lining, domain II | 0.769 | 0.703–0.811 | Possibly Pathogenic |

CPS ordering (S218L > R192Q > T666M) matches documented clinical severity hierarchy in the FHM literature.

## Features

- gnomAD v4 allele frequency retrieval (GraphQL API + VCF batch mode)
- ClinVar pathogenicity training set construction (2-star minimum, configurable)
- AlphaFold2 structure download and domain mapping (all 24 transmembrane segments, EEEE selectivity filter, interdomain linkers)
- FoldX / EvoEF2 ΔΔG thermodynamic stability estimation with TM-domain reliability flags
- Local structural RMSD computation (±10 residue Cα window, BioPython Superimposer)
- Jensen-Shannon divergence conservation scoring from 87-vertebrate CACNA1A alignment
- ACMG evidence integration (PS3/BS3, PM2, PP3/BP4 evidence codes)
- CADD-PHRED integration for ensemble meta-score features
- Bootstrap confidence intervals for CPS (1,000 iterations, feature-specific noise models)
- Reports: JSON (full feature dump), Markdown (table + citations), HTML (visual CPS bar + contribution plots)
- Four publication figures (domain architecture, domain-stratified CPS, ROC curves, forest plot) + interactive 3D structure viewer
- CLI: `chanvar annotate`, `chanvar batch`, `chanvar train`, `chanvar validate`

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Single variant (offline, no external tools required)
chanvar annotate --variant R192Q --af 0.0 --cadd 32.5 --output results/

# Single variant (with live gnomAD lookup)
chanvar annotate --variant R192Q --output results/

# Batch mode (CSV with aa_change, af, cadd columns)
chanvar batch --input variants.csv --output results/

# Run test suite (58 tests)
python -m pytest tests/ -v
```

See [SETUP.md](SETUP.md) for complete installation instructions, optional dependency configuration (FoldX, EvoEF2, MUSCLE), and environment setup on Windows/macOS/Linux.

## Project Structure

```
chanvar/
├── data/               # gnomAD, ClinVar, AlphaFold clients; Grantham matrix; conservation TSV
├── structure/          # ΔΔG stability (FoldX/EvoEF2), local RMSD
├── evolution/          # JSD conservation scoring, MSA alignment wrappers
├── scoring/            # Feature assembly, CPS computation, bootstrap CI
├── report/             # JSON/Markdown/HTML report generators
└── viz/                # Figures 1–4 (matplotlib) + Figure 5 (3Dmol.js interactive viewer)
tests/                  # 58 unit tests (pytest)
figures/                # Generated figure output
```

## MiSOF Series

ChanVar is Project 1 of the MiSOF (Migraine Stratification Outcomes Framework) series:

| # | Project | Description |
|---|---------|-------------|
| 1 | **ChanVar** | CACNA1A variant functional annotation (this project) |
| 2 | TraitStrata | Genotype-phenotype stratification across channelopathy spectrum |
| 3 | SigVigil | FDA FAERS pharmacovigilance signal analysis for migraine preventives |
| 4 | NeuroTrack | Adverse effect trajectory modeling and treatment outcome prediction |

## Citation

If you use ChanVar in research, please cite:

> Xiu, A. (2026). *ChanVar: In Silico Functional Annotation of Rare CACNA1A Variants in Familial Hemiplegic Migraine*. The Xiu Lab. https://github.com/axshoe/ChanVar

And the primary methods papers listed in `chanvar/report/generator.py → CITATIONS`.

## License

MIT — see LICENSE.

---

*The Xiu Lab · thexiulab.org · github.com/axshoe*
