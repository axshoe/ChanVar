"""
ChanVar: In Silico Functional Annotation of Rare CACNA1A Variants in FHM
=========================================================================
A Xiu Lab Project | thexiulab.org | github.com/axshoe/chanvar

Pipeline for CACNA1A-specific pathogenicity scoring integrating:
  - Population frequency (gnomAD v4)
  - Structural stability (FoldX dDDG + RMSD)
  - Evolutionary conservation (rate4site + PhyloP)
  - Functional domain membership
  - Known variant comparison (ClinVar)
  - Ensemble predictor aggregation (AlphaMissense, CADD, REVEL)

Output: ChanVar Pathogenicity Score (CPS) with 95% bootstrap CI.
"""

__version__ = "1.0.0"
__author__ = "Angie Xiu"
__lab__ = "The Xiu Lab"
__url__ = "https://thexiulab.org"
__github__ = "https://github.com/axshoe/chanvar"

from chanvar.scoring.cps import compute_cps
from chanvar.scoring.features import build_feature_vector
from chanvar.report.generator import generate_report

__all__ = ["compute_cps", "build_feature_vector", "generate_report"]
