"""
chanvar/scoring/gof_lof.py
---------------------------
GOF/LOF binary classifier for CACNA1A missense variants.

Predicts whether a variant is likely Gain-of-Function (GOF, FHM1)
or Loss-of-Function (LOF, EA2) based on structural and evolutionary features.

Method: Logistic regression with L2 regularization, trained on labeled
FHM1 (GOF) and EA2 (LOF) variants from ClinVar + FHM1 literature.

AUROC target: ~0.75 on leave-one-out CV (Heyne et al. achieved 0.85
on a much larger cross-channel dataset).

Output: A probability P(GOF) in [0, 1] alongside the CPS.
  P(GOF) > 0.6 -> likely gain of function (FHM1 type)
  P(GOF) < 0.4 -> likely loss of function (EA2 type)
  0.4 <= P(GOF) <= 0.6 -> ambiguous (report as such)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class GOFLOFResult:
    p_gof: float           # Probability of gain-of-function [0, 1]
    prediction: str        # "GOF", "LOF", or "Ambiguous"
    confidence: str        # "High" (p > 0.7 or p < 0.3), "Moderate", "Low"
    features_used: dict    # Feature values that drove prediction


# ── Training data ─────────────────────────────────────────────────────────
# Known GOF (FHM1) variants with structural features
# Format: (aa_change, domain, grantham_dist, foldx_ddg, conservation_jsd, pore_f9)
GOF_VARIANTS = [
    # Voltage sensor S4-I
    ("R192Q", "voltage_sensor", 43,  1.60, 0.95, 0.82),
    ("S218L", "voltage_sensor", 145, -1.21, 0.88, 0.81),
    ("V215A", "voltage_sensor", 64,  0.80, 0.82, 0.83),
    ("P202L", "voltage_sensor", 98,  -0.42, 0.80, 0.78),
    # Pore lining -- GOF if activation is shifted
    ("D302H", "pore_lining",    65,  3.20, 0.90, 0.93),
    ("D302N", "pore_lining",    23,  0.99, 0.90, 0.93),
    ("G297R", "pore_lining",   125,  4.88, 0.89, 0.92),
    ("I711M", "pore_lining",    10,  0.41, 0.87, 0.89),
    ("I711V", "pore_lining",    29,  1.46, 0.87, 0.89),
    ("R1660H", "pore_lining",   29,  0.79, 0.86, 0.85),
    ("R1660C", "pore_lining",  180,  1.72, 0.86, 0.85),
    ("R1666W", "pore_lining",  101, -1.54, 0.84, 0.83),
    ("R1666P", "pore_lining",  103,  3.43, 0.84, 0.83),
    ("G293R", "selectivity_filter", 125, 16.36, 0.97, 0.97),
    ("G676R", "pore_lining",   125, 18.62, 0.88, 0.84),
    ("P1352L", "pore_lining",   98,  0.85, 0.83, 0.86),
    ("L1344P", "pore_lining",   98,  1.31, 0.84, 0.84),
]

# Known LOF (EA2) variants
LOF_VARIANTS = [
    ("T665M", "pore_lining",   81, -1.52, 0.88, 0.88),
    ("R1348Q", "pore_lining",  43,  0.84, 0.87, 0.87),
    ("R1351L", "pore_lining",  83, -1.41, 0.88, 0.88),
    ("R1351Q", "pore_lining",  43,  0.33, 0.88, 0.88),
    ("R1663Q", "selectivity_filter", 43, 2.22, 0.92, 0.97),
    ("E667K",  "pore_lining",  56, -2.39, 0.84, 0.83),
    ("T1355N", "pore_lining",  65, -0.13, 0.82, 0.84),
    ("A712T",  "pore_lining",  58,  0.80, 0.87, 0.85),
    ("V713M",  "pore_lining",  21, -1.11, 0.85, 0.86),
    ("S1798L", "interdomain_linker", 145, -1.40, 0.72, 0.44),
    ("S1798P", "interdomain_linker", 74,  3.10, 0.72, 0.44),
]

DOMAIN_ENCODING = {
    "voltage_sensor": 1.0,
    "selectivity_filter": 0.9,
    "pore_lining": 0.7,
    "interdomain_linker": 0.3,
    "N_terminal": 0.2,
    "C_terminal": 0.2,
}


def _build_features(variants_list: list) -> np.ndarray:
    """Convert variant tuples to feature matrix."""
    features = []
    for _, domain, grantham, ddg, cons, pore_f9 in variants_list:
        domain_enc = DOMAIN_ENCODING.get(domain, 0.3)
        # GOF signal features:
        # - S4 domain (voltage sensor) more likely GOF
        # - Moderate ddG (not extreme destabilization) more likely GOF
        # - High conservation (conserved positions tend to cause FHM1 GOF)
        # - Close to pore (high pore_f9) more likely GOF
        feat = [
            domain_enc,
            float(np.sign(ddg)),       # sign of ddG: positive=destabilizing
            float(min(abs(ddg), 5.0) / 5.0),  # magnitude, capped at 5
            float(cons),               # conservation score
            float(pore_f9),            # pore distance feature
            float(grantham / 215.0),   # physicochemical radicality
        ]
        features.append(feat)
    return np.array(features)


class GOFLOFClassifier:
    """
    Logistic regression classifier for GOF vs LOF prediction in CACNA1A variants.

    Train with fit(), then predict with predict_proba().
    """

    def __init__(self, C: float = 1.0):
        if not SKLEARN_AVAILABLE:
            raise ImportError("pip install scikit-learn")
        self.model = LogisticRegression(C=C, penalty="l2", random_state=42,
                                        max_iter=1000, class_weight="balanced")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_gof=None, X_lof=None):
        """
        Train on GOF and LOF variants.
        Uses built-in training data if X_gof/X_lof not provided.
        """
        if X_gof is None:
            X_gof = _build_features(GOF_VARIANTS)
        if X_lof is None:
            X_lof = _build_features(LOF_VARIANTS)

        X = np.vstack([X_gof, X_lof])
        y = np.array([1] * len(X_gof) + [0] * len(X_lof))

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

        # Report leave-one-out CV performance
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        probs = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X_scaled):
            m = LogisticRegression(C=1.0, penalty="l2", random_state=42,
                                   max_iter=1000, class_weight="balanced")
            m.fit(X_scaled[train_idx], y[train_idx])
            probs[test_idx] = m.predict_proba(X_scaled[test_idx])[:, 1]

        auc = roc_auc_score(y, probs)
        print(f"GOF/LOF classifier LOO-CV AUROC: {auc:.3f} (n={len(y)}: {len(X_gof)} GOF, {len(X_lof)} LOF)")
        return auc

    def predict(self,
                domain: str,
                foldx_ddg: float,
                conservation_jsd: float,
                pore_f9: float,
                grantham_dist: float) -> GOFLOFResult:
        """
        Predict GOF/LOF for a single variant.

        Args:
            domain: Domain name (e.g., "voltage_sensor", "pore_lining")
            foldx_ddg: FoldX delta-delta-G in kcal/mol
            conservation_jsd: JSD conservation score [0, 1]
            pore_f9: Pore distance feature [0, 1] (1 = close to pore)
            grantham_dist: Grantham physicochemical distance [0, 215]

        Returns:
            GOFLOFResult with P(GOF), prediction string, confidence
        """
        if not self.fitted:
            self.fit()

        domain_enc = DOMAIN_ENCODING.get(domain, 0.3)
        feat = np.array([[
            domain_enc,
            float(np.sign(foldx_ddg)),
            float(min(abs(foldx_ddg), 5.0) / 5.0),
            float(conservation_jsd),
            float(pore_f9),
            float(grantham_dist / 215.0),
        ]])
        feat_scaled = self.scaler.transform(feat)
        p_gof = float(self.model.predict_proba(feat_scaled)[0, 1])

        if p_gof >= 0.6:
            prediction = "GOF"
        elif p_gof <= 0.4:
            prediction = "LOF"
        else:
            prediction = "Ambiguous"

        if p_gof >= 0.7 or p_gof <= 0.3:
            confidence = "High"
        elif p_gof >= 0.6 or p_gof <= 0.4:
            confidence = "Moderate"
        else:
            confidence = "Low"

        return GOFLOFResult(
            p_gof=round(p_gof, 3),
            prediction=prediction,
            confidence=confidence,
            features_used={
                "domain": domain,
                "foldx_ddg": foldx_ddg,
                "conservation_jsd": conservation_jsd,
                "pore_f9": pore_f9,
                "grantham_dist": grantham_dist,
            }
        )


# Module-level singleton (lazy initialization)
_classifier: Optional[GOFLOFClassifier] = None


def get_classifier() -> GOFLOFClassifier:
    """Get or initialize the module-level GOF/LOF classifier."""
    global _classifier
    if _classifier is None:
        _classifier = GOFLOFClassifier()
        _classifier.fit()
    return _classifier


def predict_gof_lof(domain: str, foldx_ddg: float, conservation_jsd: float,
                    pore_f9: float, grantham_dist: float) -> GOFLOFResult:
    """Convenience function for GOF/LOF prediction."""
    return get_classifier().predict(domain, foldx_ddg, conservation_jsd,
                                    pore_f9, grantham_dist)


if __name__ == "__main__":
    print("Training GOF/LOF classifier on built-in labeled variants...\n")
    clf = GOFLOFClassifier()
    clf.fit()

    print("\nTest predictions for canonical variants:")
    print(f"{'Variant':<12} {'P(GOF)':<10} {'Prediction':<12} {'Confidence'}")
    print("-" * 50)

    test_cases = [
        ("R192Q (GOF)",  "voltage_sensor",    1.60, 0.95, 0.82, 43),
        ("S218L (GOF)",  "voltage_sensor",   -1.21, 0.88, 0.81, 145),
        ("T665M (LOF)",  "pore_lining",      -1.52, 0.88, 0.88, 81),
        ("R1351Q (LOF)", "pore_lining",       0.33, 0.88, 0.88, 43),
        ("A453T (B/LB)", "interdomain_linker", 2.54, 0.62, 0.55, 58),
    ]

    for name, domain, ddg, cons, pore, gran in test_cases:
        result = clf.predict(domain, ddg, cons, pore, gran)
        print(f"{name:<12} {result.p_gof:<10.3f} {result.prediction:<12} {result.confidence}")
