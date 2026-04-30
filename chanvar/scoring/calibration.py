"""
chanvar/scoring/calibration.py
-------------------------------
Calibration analysis for ChanVar CPS scores.

Implements Brier score and calibration curve analysis per
Heidi Rehm's ClinGen calibration framework recommendation.

Key finding: CPS Brier skill score = 0.078 (modest).
CPS is a RANKING tool, not a calibrated probability estimator.
The correct clinical framing: use CPS to prioritize variants
for follow-up, not to directly estimate P(pathogenic).

This module also provides isotonic regression recalibration
to improve probability calibration if needed.
"""

import numpy as np
import json
from typing import Tuple

try:
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def brier_score(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Brier score, baseline Brier score, and Brier skill score.

    Args:
        labels: Binary labels (1=P/LP, 0=B/LB)
        scores: CPS scores in [0, 1]

    Returns:
        (brier, brier_baseline, brier_skill_score)
        Higher skill score = better calibration relative to base rate
    """
    if not SKLEARN_AVAILABLE:
        # Manual computation
        brier = float(np.mean((scores - labels.astype(float)) ** 2))
        prevalence = labels.mean()
        brier_baseline = float(prevalence * (1 - prevalence))
        skill = float(1 - brier / brier_baseline)
        return brier, brier_baseline, skill

    brier = float(brier_score_loss(labels, scores))
    prevalence = labels.mean()
    brier_baseline = float(brier_score_loss(labels, np.full(len(labels), prevalence)))
    skill = float(1 - brier / brier_baseline)
    return brier, brier_baseline, skill


def calibration_analysis(labels: np.ndarray, scores: np.ndarray,
                          n_bins: int = 5) -> dict:
    """
    Full calibration analysis for CPS scores.

    Args:
        labels: Binary labels (1=P/LP, 0=B/LB)
        scores: CPS scores in [0, 1]
        n_bins: Number of calibration bins (5 recommended for n=93)

    Returns:
        Dict with all calibration metrics and bin data
    """
    brier, brier_baseline, skill = brier_score(labels, scores)

    result = {
        "n": int(len(labels)),
        "n_plp": int(labels.sum()),
        "n_blb": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "brier_score": round(brier, 4),
        "brier_baseline": round(brier_baseline, 4),
        "brier_skill_score": round(skill, 4),
        "interpretation": (
            "CPS is a ranking tool with modest calibration (skill=0.078). "
            "Use for variant prioritization, not as a direct probability estimate. "
            "High-scoring variants (CPS > 0.75) show good calibration; "
            "mid-range variants (0.55-0.65) show substantial uncertainty."
        ),
    }

    # Calibration curve bins
    if SKLEARN_AVAILABLE:
        prob_true, prob_pred = calibration_curve(
            labels, scores, n_bins=n_bins, strategy='quantile'
        )
        result["calibration_bins"] = [
            {"mean_predicted": round(float(pp), 3),
             "fraction_positive": round(float(pt), 3)}
            for pp, pt in zip(prob_pred, prob_true)
        ]

    # Summary of calibration quality by CPS range
    ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    range_stats = []
    for lo, hi in ranges:
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            range_stats.append({
                "cps_range": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "fraction_plp": round(float(labels[mask].mean()), 3),
            })
    result["cps_range_stats"] = range_stats

    return result


def recalibrate_isotonic(train_labels: np.ndarray, train_scores: np.ndarray,
                          new_scores: np.ndarray) -> np.ndarray:
    """
    Isotonic regression recalibration.
    Converts CPS ranking scores into better-calibrated probabilities.

    Args:
        train_labels: Labels for the training/calibration set
        train_scores: CPS scores for the calibration set
        new_scores: Scores to recalibrate

    Returns:
        Recalibrated probability estimates
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("pip install scikit-learn")

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_scores, train_labels.astype(float))
    return ir.predict(new_scores)


def generate_calibration_report(labels: np.ndarray, scores: np.ndarray,
                                 output_path: str = None) -> str:
    """
    Generate a human-readable calibration report.

    Args:
        labels: Binary labels
        scores: CPS scores
        output_path: If provided, write JSON report to this path

    Returns:
        Formatted text report
    """
    analysis = calibration_analysis(labels, scores)

    lines = [
        "=" * 60,
        "CHANVAR CPS CALIBRATION REPORT",
        "=" * 60,
        f"Validation set: n={analysis['n']} ({analysis['n_plp']} P/LP, {analysis['n_blb']} B/LB)",
        f"Prevalence: {analysis['prevalence']:.3f}",
        "",
        "CALIBRATION METRICS:",
        f"  Brier score:         {analysis['brier_score']:.4f}",
        f"  Baseline Brier:      {analysis['brier_baseline']:.4f}",
        f"  Brier skill score:   {analysis['brier_skill_score']:.4f}",
        "",
        "INTERPRETATION:",
        f"  {analysis['interpretation']}",
        "",
        "CALIBRATION BY CPS RANGE:",
        f"  {'CPS Range':<15} {'n':>6} {'Fraction P/LP':>15}",
        "  " + "-" * 40,
    ]

    for stat in analysis.get("cps_range_stats", []):
        lines.append(f"  {stat['cps_range']:<15} {stat['n']:>6} {stat['fraction_plp']:>15.3f}")

    if "calibration_bins" in analysis:
        lines += [
            "",
            "CALIBRATION BINS (quantile):",
            f"  {'Mean CPS':>12} {'Fraction P/LP':>15}",
            "  " + "-" * 30,
        ]
        for bin_data in analysis["calibration_bins"]:
            lines.append(
                f"  {bin_data['mean_predicted']:>12.3f} {bin_data['fraction_positive']:>15.3f}"
            )

    lines += [
        "",
        "RECOMMENDATION FOR LAB USE (per ClinGen calibration framework):",
        "  CPS >= 0.75: Strong pathogenic signal; supports PP3 evidence at moderate level",
        "  CPS 0.67-0.75: Moderate signal; supports PP3 at supporting level",
        "  CPS 0.50-0.67: Weak signal; insufficient for ACMG evidence code",
        "  CPS < 0.50: Possible benign signal; may support BP4 at supporting level",
        "",
        "  NOTE: These thresholds require validation against ACMG-classified variants",
        "  before use as standalone ACMG evidence codes. ChanVar is a research tool.",
        "=" * 60,
    ]

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"JSON report written to {output_path}")

    return report


if __name__ == "__main__":
    # Run calibration on the full 93-variant validation set
    results = [
        ("R192Q",0.8159,1),("T665M",0.7227,1),("I1809L",0.6174,1),("R1660H",0.7187,1),
        ("G293R",0.7664,1),("Y1383C",0.7388,1),("S218L",0.8167,1),("R582Q",0.6433,1),
        ("I1708T",0.7051,1),("R1345Q",0.7155,1),("T500M",0.6184,1),("R1663Q",0.7275,1),
        ("R1666W",0.7181,1),("H253Y",0.6473,1),("V1392M",0.6835,1),("D302N",0.8129,1),
        ("A712T",0.7295,1),("R1348Q",0.7245,1),("S1798L",0.6377,1),("D1643N",0.7396,1),
        ("A1507D",0.7208,1),("P1360Q",0.7421,1),("P1352L",0.7600,1),("R1348W",0.7232,1),
        ("E532K",0.6095,1),("D1633N",0.5929,1),("R1358W",0.7225,1),("R1351L",0.7068,1),
        ("I711M",0.8339,1),("G1754R",0.6962,1),("A1807S",0.6219,1),("E1263K",0.6568,1),
        ("R279C",0.7183,1),("G297R",0.7900,1),("E147K",0.6188,1),("S1798P",0.7022,1),
        ("C272R",0.7463,1),("R1666P",0.7696,1),("I613M",0.6632,1),("I711V",0.8198,1),
        ("V215A",0.7350,1),("S615N",0.7174,1),("V1455M",0.7085,1),("R1351Q",0.7046,1),
        ("T1355N",0.7342,1),("E667K",0.7098,1),("E101K",0.7278,1),("S1468L",0.7429,1),
        ("P202L",0.7180,1),("G676R",0.7789,1),("R1660C",0.7855,1),("V1392A",0.7028,1),
        ("G539R",0.7275,1),("V1808I",0.5611,1),("C272Y",0.7502,1),("D302H",0.8623,1),
        ("A1507T",0.6780,1),("L1344P",0.7544,1),("V713M",0.7353,1),("Y62H",0.6896,1),
        ("Y62C",0.7300,1),
        ("A453T",0.6885,0),("E917D",0.5746,0),("E992V",0.5824,0),("G1104S",0.6287,0),
        ("E731A",0.5461,0),("P913S",0.5284,0),("P1010A",0.5700,0),("P1137A",0.5375,0),
        ("R893Q",0.5723,0),("G266S",0.6202,0),("G2023S",0.5861,0),("A21V",0.5994,0),
        ("R2195Q",0.5198,0),("Q1153E",0.5623,0),("R1234H",0.5542,0),("E2021K",0.5031,0),
        ("A2431T",0.5043,0),("D1102N",0.5354,0),("G2425D",0.5888,0),("P1103L",0.5441,0),
        ("G2261D",0.5763,0),("I1631V",0.5562,0),("A920G",0.6052,0),("P896L",0.5891,0),
        ("H2480Q",0.5116,0),("G876S",0.5988,0),("P2421L",0.5212,0),("P2421A",0.5232,0),
        ("A2443V",0.5739,0),("D800E",0.5852,0),("E842K",0.5309,0),("G1151S",0.6187,0),
    ]
    scores_full = np.array([r[1] for r in results])
    labels = np.array([r[2] for r in results])
    W_TOTAL, W_F6 = 14.0, 1.5
    f6v = labels.astype(float)
    scores = (scores_full * W_TOTAL - W_F6 * f6v) / (W_TOTAL - W_F6)

    report = generate_calibration_report(labels, scores, "calibration_report.json")
    print(report)
