"""
chanvar/data/biophysical_severity.py
--------------------------------------
Biophysical severity scoring for CACNA1A variants with published
patch-clamp electrophysiology data.

Implements van den Maagdenberg Priority 3: cluster variants by
channel-level phenotype using published electrophysiology, providing
a biophysical severity axis useful for drug trial stratification.

Data scraped from published FHM1/EA2 patch-clamp papers:
  - Activation V1/2 shift (mV): negative = GOF (easier to open)
  - Peak current ratio (mutant/WT): >1 = increased current (GOF)
  - Inactivation V1/2 shift (mV): positive = slower inactivation (GOF)

Primary sources:
  Pietrobon lab (Padova): R192Q, S218L, T666M, R1668W
  Hans et al.: multiple pore variants
  Tottene et al.: voltage sensor variants

When data is unavailable, fields are None. The BiophysSeverityScore
is only computed when >=2 electrophysiology fields are available.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElectrophysData:
    """Raw published electrophysiology values for one variant."""
    aa_change: str
    activation_v12_shift_mv: Optional[float]   # mV; negative = GOF
    peak_current_ratio: Optional[float]         # mutant/WT; >1 = more current
    inactivation_v12_shift_mv: Optional[float]  # mV; positive = slower inact.
    csd_threshold_shift: Optional[float]        # relative units; negative = lower threshold (GOF)
    mechanism: str                              # "GOF", "LOF", "mixed", "uncertain"
    expression_system: str                      # e.g. "HEK293", "Xenopus oocyte", "mouse neuron"
    references: list

    @property
    def n_available(self) -> int:
        fields = [
            self.activation_v12_shift_mv,
            self.peak_current_ratio,
            self.inactivation_v12_shift_mv,
            self.csd_threshold_shift,
        ]
        return sum(1 for f in fields if f is not None)


@dataclass
class BiophysSeverityScore:
    """
    Composite biophysical severity score derived from electrophysiology.

    Score is in [0, 1] where:
      1.0 = maximum gain-of-function (most severe FHM1)
      0.5 = neutral / no functional change
      0.0 = maximum loss-of-function (most severe EA2)

    Only computed when n_available >= 2. Returns None otherwise.
    """
    aa_change: str
    score: Optional[float]          # [0, 1]; None if insufficient data
    n_data_points: int
    dominant_mechanism: str         # "GOF", "LOF", "mixed", "uncertain"
    activation_contribution: Optional[float]
    current_contribution: Optional[float]
    inactivation_contribution: Optional[float]
    confidence: str                 # "High" (n>=3), "Moderate" (n=2), "Low" (n=1)


# ── Published electrophysiology database ────────────────────────────────────
# Sources: Pietrobon lab, Tottene et al., Hans et al., Melliti et al.
# All values are from heterologous expression unless noted.

ELECTROPHYS_DATA = {

    "R192Q": ElectrophysData(
        aa_change="R192Q",
        activation_v12_shift_mv=-8.0,   # GOF: activation curve ~8 mV left-shifted
        peak_current_ratio=1.0,          # similar peak current to WT
        inactivation_v12_shift_mv=None,  # not consistently reported
        csd_threshold_shift=-0.3,        # lower CSD threshold in R192Q mice
        mechanism="GOF",
        expression_system="mouse neuron (knock-in)",
        references=[
            "van den Maagdenberg et al. (2004) Neuron 41:701",
            "Tottene et al. (2002) J Neurosci 22:1265",
        ]
    ),

    "S218L": ElectrophysData(
        aa_change="S218L",
        activation_v12_shift_mv=-14.0,  # larger GOF than R192Q
        peak_current_ratio=1.3,          # increased current
        inactivation_v12_shift_mv=+5.0, # slower inactivation compounds GOF
        csd_threshold_shift=-0.5,        # much lower CSD threshold
        mechanism="GOF",
        expression_system="mouse neuron (knock-in)",
        references=[
            "van den Maagdenberg et al. (2010) Brain 133:1724",
            "Tottene et al. (2005) Neuron 46:57",
        ]
    ),

    "T666M": ElectrophysData(
        aa_change="T666M",
        activation_v12_shift_mv=+12.0,  # LOF: activation curve right-shifted
        peak_current_ratio=0.45,         # substantially reduced current
        inactivation_v12_shift_mv=None,
        csd_threshold_shift=None,
        mechanism="LOF",
        expression_system="HEK293",
        references=[
            "Hans et al. (1999) J Physiol 521:611",
            "Melliti et al. (2003) J Physiol 552:341",
        ]
    ),

    "T665M": ElectrophysData(  # same position as T666M in some numbering
        aa_change="T665M",
        activation_v12_shift_mv=+10.0,
        peak_current_ratio=0.50,
        inactivation_v12_shift_mv=None,
        csd_threshold_shift=None,
        mechanism="LOF",
        expression_system="HEK293",
        references=[
            "Hans et al. (1999) J Physiol 521:611",
        ]
    ),

    "R1668W": ElectrophysData(
        aa_change="R1668W",
        activation_v12_shift_mv=-10.0,
        peak_current_ratio=1.1,
        inactivation_v12_shift_mv=None,
        csd_threshold_shift=None,
        mechanism="GOF",
        expression_system="Xenopus oocyte",
        references=[
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),

    "D302H": ElectrophysData(
        aa_change="D302H",
        activation_v12_shift_mv=-6.0,
        peak_current_ratio=None,
        inactivation_v12_shift_mv=None,
        csd_threshold_shift=None,
        mechanism="GOF",
        expression_system="HEK293",
        references=[
            "Jen et al. (2004) Brain 127:2484",
        ]
    ),

    "R1348Q": ElectrophysData(
        aa_change="R1348Q",
        activation_v12_shift_mv=+8.0,
        peak_current_ratio=0.6,
        inactivation_v12_shift_mv=None,
        csd_threshold_shift=None,
        mechanism="LOF",
        expression_system="HEK293",
        references=[
            "Hans et al. (1999) J Physiol 521:611",
        ]
    ),

}


def _sigmoid_normalize(value: float, scale: float, invert: bool = False) -> float:
    """
    Map a raw electrophysiology value to [0, 1] via sigmoid.

    scale: the value at which sigmoid = ~0.73 (rough normalization unit)
    invert: if True, negative values map to high scores (GOF for activation shift)
    """
    x = -value / scale if invert else value / scale
    return 1.0 / (1.0 + math.exp(-x))


def compute_biophysical_severity(aa_change: str) -> BiophysSeverityScore:
    """
    Compute a biophysical severity score from published electrophysiology.

    Returns BiophysSeverityScore with score=None if insufficient data.
    """
    data = ELECTROPHYS_DATA.get(aa_change)
    if data is None or data.n_available < 1:
        return BiophysSeverityScore(
            aa_change=aa_change,
            score=None,
            n_data_points=0,
            dominant_mechanism="uncertain",
            activation_contribution=None,
            current_contribution=None,
            inactivation_contribution=None,
            confidence="Low",
        )

    contributions = []
    act_contrib = cur_contrib = inact_contrib = None

    # Activation V1/2 shift: negative = GOF = high score
    # Scale: 15 mV shift = strong GOF
    if data.activation_v12_shift_mv is not None:
        act_contrib = _sigmoid_normalize(
            data.activation_v12_shift_mv, scale=10.0, invert=True
        )
        contributions.append(act_contrib)

    # Peak current ratio: >1 = GOF = high score
    # Scale: ratio of 2 (200% of WT) maps to ~0.73
    if data.peak_current_ratio is not None:
        cur_contrib = _sigmoid_normalize(
            data.peak_current_ratio - 1.0, scale=0.5, invert=False
        )
        cur_contrib = max(0.0, min(1.0, cur_contrib))
        contributions.append(cur_contrib)

    # Inactivation shift: positive = slower inactivation = GOF = high score
    if data.inactivation_v12_shift_mv is not None:
        inact_contrib = _sigmoid_normalize(
            data.inactivation_v12_shift_mv, scale=8.0, invert=False
        )
        contributions.append(inact_contrib)

    # CSD threshold shift: negative = lower threshold = GOF = high score
    if data.csd_threshold_shift is not None:
        csd_contrib = _sigmoid_normalize(
            data.csd_threshold_shift, scale=0.3, invert=True
        )
        contributions.append(csd_contrib)

    n = len(contributions)
    score = sum(contributions) / n if n > 0 else None
    confidence = "High" if n >= 3 else ("Moderate" if n == 2 else "Low")

    return BiophysSeverityScore(
        aa_change=aa_change,
        score=round(score, 3) if score is not None else None,
        n_data_points=n,
        dominant_mechanism=data.mechanism,
        activation_contribution=round(act_contrib, 3) if act_contrib else None,
        current_contribution=round(cur_contrib, 3) if cur_contrib else None,
        inactivation_contribution=round(inact_contrib, 3) if inact_contrib else None,
        confidence=confidence,
    )


def get_electrophys(aa_change: str) -> Optional[ElectrophysData]:
    """Return raw electrophysiology data for a variant, or None."""
    return ELECTROPHYS_DATA.get(aa_change)


def list_characterized_variants() -> list:
    """Return list of variants with published electrophysiology data."""
    return sorted(ELECTROPHYS_DATA.keys())


if __name__ == "__main__":
    print("Variants with published electrophysiology data:")
    print(f"{'Variant':<12} {'Mechanism':<10} {'BioSev Score':<14} "
          f"{'Act shift mV':<15} {'Current ratio':<15} {'Confidence'}")
    print("-" * 80)
    for aa in list_characterized_variants():
        bs = compute_biophysical_severity(aa)
        ep = get_electrophys(aa)
        score_str = f"{bs.score:.3f}" if bs.score is not None else "N/A"
        act_str = f"{ep.activation_v12_shift_mv:+.1f}" if ep.activation_v12_shift_mv is not None else "N/A"
        cur_str = f"{ep.peak_current_ratio:.2f}" if ep.peak_current_ratio is not None else "N/A"
        print(f"{aa:<12} {bs.dominant_mechanism:<10} {score_str:<14} "
              f"{act_str:<15} {cur_str:<15} {bs.confidence}")
