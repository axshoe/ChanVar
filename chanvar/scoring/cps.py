"""
chanvar/scoring/cps.py
-----------------------
ChanVar Pathogenicity Score (CPS) computation with bootstrap uncertainty
quantification.

The CPS is a weighted linear combination of nine normalized features:
    CPS = sum(w_i * f_i) / sum(w_i)

Weights encode biological importance ordering from the literature.
Bootstrap confidence intervals (95%) are computed by perturbing each
feature by Gaussian noise proportional to its expected measurement error.

Interpretation thresholds (calibrated against ClinVar P/LP and B/LB sets):
    CPS >= 0.85: Likely Pathogenic
    CPS 0.70-0.85: Possibly Pathogenic
    CPS 0.40-0.70: Uncertain Significance
    CPS 0.20-0.40: Possibly Benign
    CPS < 0.20: Likely Benign

Calibration (Brier skill score 0.078): CPS is a ranking tool, not a
calibrated probability estimator. High-CPS variants (>0.75) show strong
calibration (91% P/LP in validation set); mid-range variants (0.60-0.70)
are substantially less reliable.

ClinGen evidence code guidance (per calibration analysis):
    CPS >= 0.75: PP3 moderate
    CPS 0.67-0.75: PP3 supporting
    CPS 0.50-0.67: insufficient
    CPS < 0.50: BP4 supporting (requires institutional validation)
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional

from chanvar.scoring.features import VariantFeatures

logger = logging.getLogger(__name__)

# Feature weights encoding biological importance
# W_TOTAL = 15.5 (sum of all weights)
# W_F6 = 1.5 (ClinVar prior, excluded for independent validation)
# W_NO_F6 = 14.0 (used when computing f6-excluded CPS)
PRIOR_WEIGHTS = {
    "f1": 3.0,   # gnomAD frequency: strongest single predictor
    "f2": 1.5,   # FoldX dDDG: reliable for cytoplasmic, less for TM
    "f3": 1.0,   # Local RMSD: currently inactive placeholder
    "f4": 2.5,   # Conservation: second strongest predictor
    "f5": 2.0,   # Domain weight: position-specific biological knowledge
    "f6": 1.5,   # ClinVar prior (EXCLUDED from independent validation)
    "f7": 1.0,   # CADD: general purpose genomic score
    "f8": 1.0,   # Grantham: physicochemical severity
    "f9": 1.5,   # Pore-axis distance (Brunger et al. Brain 2023)
}

W_TOTAL = sum(PRIOR_WEIGHTS.values())  # 15.5
W_F6 = PRIOR_WEIGHTS["f6"]            # 1.5
W_NO_F6 = W_TOTAL - W_F6              # 14.0

# Transmembrane domain adjustment: halve f2 weight for TM variants
# FoldX aqueous-solvent parameterization is unreliable for membrane helices
TM_WEIGHT_ADJUSTMENT = {"f2": 0.5}

CPS_THRESHOLDS = {
    "likely_pathogenic": 0.85,
    "possibly_pathogenic": 0.70,
    "vus_upper": 0.70,
    "vus_lower": 0.40,
    "possibly_benign": 0.20,
}

FEATURE_NAMES = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]


@dataclass
class CPSResult:
    """
    Output of the ChanVar Pathogenicity Score computation.

    Attributes
    ----------
    variant_id : str
    aa_change : str
    cps : float
        ChanVar Pathogenicity Score [0, 1].
    ci_lower : float
        95% bootstrap CI lower bound.
    ci_upper : float
        95% bootstrap CI upper bound.
    classification : str
    domain : str
    weights_used : dict
    feature_contributions : dict
        Per-feature contribution to CPS (w_i * f_i / sum_w).
    data_completeness : float
        Fraction of active features with non-None values.
    is_tm_domain : bool
    confidence_flag : str or None
        Warning flags: TM_DDG_UNRELIABLE, LOW_COMPLETENESS,
        CTERMINAL_UNDERPOWERED (may be comma-joined).
    clinvar_override : bool
    clinvar_sig : str or None
    """

    variant_id: str
    aa_change: str
    cps: float
    ci_lower: float
    ci_upper: float
    classification: str
    domain: str
    weights_used: dict
    feature_contributions: dict
    data_completeness: float
    is_tm_domain: bool
    confidence_flag: Optional[str] = None
    clinvar_override: bool = False
    clinvar_sig: Optional[str] = None

    def summary_line(self) -> str:
        flag = f" [{self.confidence_flag}]" if self.confidence_flag else ""
        return (
            f"{self.aa_change} | CPS={self.cps:.3f} "
            f"(95% CI: {self.ci_lower:.3f}-{self.ci_upper:.3f}) | "
            f"{self.classification} | Domain: {self.domain}{flag}"
        )


def compute_cps(
    features: VariantFeatures,
    weights: Optional[dict] = None,
    bootstrap_n: int = 1000,
    seed: Optional[int] = 42,
) -> CPSResult:
    """
    Compute ChanVar Pathogenicity Score with 95% bootstrap CI.

    Parameters
    ----------
    features : VariantFeatures
    weights : dict or None
        Feature weights {f1..f9: float}. Defaults to PRIOR_WEIGHTS.
    bootstrap_n : int
        Bootstrap iterations for CI. Default 1000.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    CPSResult
    """
    if weights is None:
        weights = dict(PRIOR_WEIGHTS)

    # Adjust weights for transmembrane domains
    if features.is_tm_domain:
        for feat, adj in TM_WEIGHT_ADJUSTMENT.items():
            if feat in weights:
                weights[feat] = weights[feat] * adj

    # Check for ClinVar override
    clinvar_override = False
    if features.clinvar_sig in (
        "Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
        "Benign", "Likely benign", "Benign/Likely benign",
    ):
        clinvar_override = True

    # Point estimate CPS
    fv = features.feature_vector   # [f1..f9], None -> 0.5
    w_list = [weights[f] for f in FEATURE_NAMES]
    cps_point = _weighted_mean(fv, w_list)

    # Per-feature contributions
    w_total = sum(w_list)
    contributions = {
        name: (w * f) / w_total
        for name, w, f in zip(FEATURE_NAMES, w_list, fv)
    }

    # Bootstrap CI
    ci_lower, ci_upper = _bootstrap_ci(fv, w_list, features, n=bootstrap_n, seed=seed)

    # Classification
    classification = _classify(cps_point)
    if clinvar_override and features.clinvar_sig:
        classification = _classify_from_clinvar(features.clinvar_sig)

    # Confidence flags
    flags = []
    if features.is_tm_domain and features.f2 is not None:
        flags.append("TM_DDG_UNRELIABLE")
    if features.data_completeness < 0.5:
        flags.append("LOW_COMPLETENESS")
    if features.domain == "C_terminal" and features.data_completeness < 1.0:
        flags.append("CTERMINAL_UNDERPOWERED")
    confidence_flag = ",".join(flags) if flags else None

    return CPSResult(
        variant_id=features.variant_id,
        aa_change=features.aa_change,
        cps=round(cps_point, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        classification=classification,
        domain=features.domain,
        weights_used=dict(weights),
        feature_contributions={k: round(v, 4) for k, v in contributions.items()},
        data_completeness=features.data_completeness,
        is_tm_domain=features.is_tm_domain,
        confidence_flag=confidence_flag,
        clinvar_override=clinvar_override,
        clinvar_sig=features.clinvar_sig,
    )


def _weighted_mean(feat_vals: list, weights: list) -> float:
    """Weighted mean of feature values. All lists must be same length."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.5
    return sum(f * w for f, w in zip(feat_vals, weights)) / total_w


def _bootstrap_ci(
    fv: list,
    weights: list,
    features: VariantFeatures,
    n: int = 1000,
    seed: Optional[int] = 42,
    alpha: float = 0.05,
):
    """
    Approximate bootstrap CI by perturbing feature values with Gaussian noise.

    Noise standard deviations reflect expected measurement uncertainty:
      f1 (AF): 0.05     f2 (dDDG): 0.10    f3 (RMSD): 0.08
      f4 (cons): 0.08   f5 (domain): 0.00  f6 (AM): 0.05
      f7 (CADD): 0.05   f8 (Gran): 0.00    f9 (pore): 0.05
    Missing features receive 0.15 (higher uncertainty from imputation).
    """
    NOISE_SD = [0.05, 0.10, 0.08, 0.08, 0.00, 0.05, 0.05, 0.00, 0.05]
    MISSING_SD = 0.15

    raw_features = [
        features.f1, features.f2, features.f3, features.f4,
        features.f5, features.f6, features.f7, features.f8, features.f9
    ]

    rng = random.Random(seed)
    samples = []

    for _ in range(n):
        perturbed = []
        for f_raw, f_val, sd in zip(raw_features, fv, NOISE_SD):
            noise = rng.gauss(0, MISSING_SD if f_raw is None else (sd if sd > 0 else 0))
            perturbed.append(max(0.0, min(1.0, f_val + noise)))
        samples.append(_weighted_mean(perturbed, weights))

    samples.sort()
    lower_idx = int(alpha / 2 * n)
    upper_idx = int((1 - alpha / 2) * n)
    return samples[lower_idx], samples[upper_idx]


def _classify(cps: float) -> str:
    """Map CPS value to classification string."""
    if cps >= CPS_THRESHOLDS["likely_pathogenic"]:
        return "Likely Pathogenic"
    elif cps >= CPS_THRESHOLDS["possibly_pathogenic"]:
        return "Possibly Pathogenic"
    elif cps >= CPS_THRESHOLDS["vus_lower"]:
        return "Uncertain Significance"
    elif cps >= CPS_THRESHOLDS["possibly_benign"]:
        return "Possibly Benign"
    return "Likely Benign"


def _classify_from_clinvar(clinvar_sig: str) -> str:
    """Map ClinVar string to ChanVar classification."""
    if any(t in clinvar_sig for t in ("Pathogenic", "pathogenic")):
        return "Pathogenic (ClinVar)"
    elif any(t in clinvar_sig for t in ("Benign", "benign")):
        return "Benign (ClinVar)"
    return "Uncertain Significance (ClinVar)"


def train_logistic_weights(
    features_list: list,
    labels: list,
    cv_folds: int = 5,
    seed: int = 42,
):
    """
    Learn feature weights from ClinVar training data via logistic regression.

    Parameters
    ----------
    features_list : list of VariantFeatures
    labels : list of int
        1 = P/LP, 0 = B/LB.
    cv_folds : int
    seed : int

    Returns
    -------
    weights : dict
    cv_auroc : float
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except ImportError:
        raise ImportError("pip install scikit-learn")

    if len(features_list) < 30:
        logger.warning(
            "Training set too small (%d variants). Using prior weights.",
            len(features_list),
        )
        return dict(PRIOR_WEIGHTS), float("nan")

    X = np.array([fv.feature_vector for fv in features_list])
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    auroc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
    cv_auroc = float(auroc_scores.mean())

    logger.info(
        "Logistic regression CV AUROC: %.3f +/- %.3f (%d-fold)",
        cv_auroc, auroc_scores.std(), cv_folds,
    )

    clf.fit(X_scaled, y)
    coef = clf.coef_[0]

    weights = {}
    for i, name in enumerate(FEATURE_NAMES):
        learned = float(coef[i])
        prior = PRIOR_WEIGHTS[name]
        if learned < 0:
            logger.warning(
                "Feature %s has negative learned weight (%.3f). Using prior %.1f.",
                name, learned, prior,
            )
            weights[name] = prior
        else:
            weights[name] = learned

    # Normalize to same total as prior weights
    prior_total = sum(PRIOR_WEIGHTS.values())
    learned_total = sum(weights.values())
    if learned_total > 0:
        scale = prior_total / learned_total
        weights = {k: v * scale for k, v in weights.items()}

    return weights, cv_auroc


def batch_score(
    features_list: list,
    weights: Optional[dict] = None,
    bootstrap_n: int = 200,
) -> list:
    """
    Score a list of VariantFeatures. Uses reduced bootstrap (200) for speed.

    Returns list of CPSResult.
    """
    results = []
    for fv in features_list:
        results.append(compute_cps(fv, weights=weights, bootstrap_n=bootstrap_n))
    logger.info("Scored %d variants.", len(results))
    return results


def summarize_scores(results: list) -> dict:
    """Generate summary statistics for a list of CPSResult."""
    from collections import Counter
    classifications = Counter(r.classification for r in results)
    scores = [r.cps for r in results]
    return {
        "n": len(results),
        "classifications": dict(classifications),
        "mean_cps": sum(scores) / len(scores) if scores else float("nan"),
        "median_cps": sorted(scores)[len(scores) // 2] if scores else float("nan"),
        "n_high_confidence": sum(
            1 for r in results if r.ci_upper - r.ci_lower < 0.2
        ),
        "n_tm_flagged": sum(
            1 for r in results if r.is_tm_domain and r.confidence_flag
        ),
        "n_clinvar_override": sum(1 for r in results if r.clinvar_override),
    }