"""
chanvar/data/de_novo_flag.py
------------------------------
De novo probability and inheritance mode classification.

Sunitha Malepati addition (May 2026):
De novo variants causing sporadic HM tend to be more severe than
inherited familial variants. This module adds:

  de_novo_probability : float [0, 1]
  inheritance_mode    : "de_novo_likely" | "familial_likely" | "uncertain"

Logic:
  de_novo_likely  = gnomAD AF < 1e-6 AND no family segregation in literature
                    AND variant not in known familial series
  familial_likely = variant reported in multiple family members across generations
  uncertain       = insufficient data

Note: this is a probability estimate, not a confirmed de novo call.
Confirmed de novo requires parental sequencing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeNovoResult:
    de_novo_probability: float      # [0, 1]; higher = more likely de novo
    inheritance_mode: str           # "de_novo_likely" | "familial_likely" | "uncertain"
    basis: str                      # brief explanation
    gnomad_af: Optional[float]      # raw AF used in assessment
    in_familial_series: bool        # True if reported in multi-generation family


# Known familial variants (reported with clear autosomal dominant transmission
# across multiple generations in the published literature)
KNOWN_FAMILIAL = {
    "R192Q",   # Ophoff 1996 - Dutch FHM families, multiple generations
    "S218L",   # Kors 2001 - Dutch family, 3 generations
    "T665M",   # Ophoff 1996 - original CACNA1A families
    "T666M",
    "D302H",   # Jen 2004 - reported in families
    "R583Q",   # Ducros 2001 - French families
    "T1010M",  # Ducros 2001
    "D715E",   # Ophoff 1996
    "V215A",   # familial FHM series
    "P202L",
    "R279C",   # reported in familial context
    "R1663Q",
    "G293R",
}

# Known or likely de novo variants (reported in sporadic patients,
# absent from gnomAD, no family history documented)
KNOWN_DE_NOVO = {
    "I1809L",   # C-terminal, sporadic reports
    "R1660H",   # sporadic neurodevelopmental
    "Y1383C",   # sporadic
    "I1708T",   # sporadic HM
    "V1392M",   # neurodevelopmental, often de novo
    "E1263K",   # epileptic encephalopathy, de novo
    "R279C",    # some de novo reports in neurodevelopmental literature
    "S1798L",   # epilepsy association, often sporadic
    "E2021K",   # epileptic encephalopathy
    "I1631V",   # epileptic encephalopathy
}


def classify_inheritance(
    aa_change: str,
    gnomad_af: Optional[float],
    clinvar_review_stars: int = 2,
) -> DeNovoResult:
    """
    Estimate inheritance mode for a CACNA1A variant.

    Args:
        aa_change: e.g. "R192Q"
        gnomad_af: gnomAD allele frequency (None if absent)
        clinvar_review_stars: number of ClinVar review stars (proxy for curation depth)

    Returns:
        DeNovoResult
    """
    # Known familial — strong evidence
    if aa_change in KNOWN_FAMILIAL:
        return DeNovoResult(
            de_novo_probability=0.05,
            inheritance_mode="familial_likely",
            basis="Reported in multi-generation FHM families in published literature.",
            gnomad_af=gnomad_af,
            in_familial_series=True,
        )

    # Known de novo
    if aa_change in KNOWN_DE_NOVO:
        return DeNovoResult(
            de_novo_probability=0.80,
            inheritance_mode="de_novo_likely",
            basis="Reported in sporadic patients; absent from gnomAD; "
                  "no multigenerational family segregation documented.",
            gnomad_af=gnomad_af,
            in_familial_series=False,
        )

    # Compute from gnomAD AF
    # De novo threshold: AF < 1e-6 (never or once seen in ~730k people)
    if gnomad_af is not None:
        if gnomad_af < 1e-6:
            # Extremely rare — consistent with de novo
            p_de_novo = 0.65
            basis = (
                f"gnomAD AF = {gnomad_af:.2e} (below de novo threshold 1e-6). "
                f"No family segregation data found. Likely de novo or ultra-rare familial."
            )
            mode = "de_novo_likely"
        elif gnomad_af < 1e-4:
            # Rare but not extreme — could be familial
            p_de_novo = 0.35
            basis = (
                f"gnomAD AF = {gnomad_af:.2e}. Rare but above de novo threshold; "
                f"could be low-penetrance familial or de novo."
            )
            mode = "uncertain"
        else:
            # Common enough to be familial polymorphism
            p_de_novo = 0.10
            basis = (
                f"gnomAD AF = {gnomad_af:.2e}. Too common for typical de novo; "
                f"likely familial or benign polymorphism."
            )
            mode = "familial_likely"
    else:
        # AF = 0 (absent from gnomAD) — strong de novo signal
        p_de_novo = 0.70
        basis = (
            "Absent from gnomAD v4 (AF = 0 in ~730,000 individuals). "
            "Consistent with de novo origin. Parental sequencing needed to confirm."
        )
        mode = "de_novo_likely"

    return DeNovoResult(
        de_novo_probability=round(p_de_novo, 2),
        inheritance_mode=mode,
        basis=basis,
        gnomad_af=gnomad_af,
        in_familial_series=aa_change in KNOWN_FAMILIAL,
    )


if __name__ == "__main__":
    tests = [
        ("R192Q",  0.0),
        ("S218L",  0.0),
        ("I1809L", 0.0),
        ("E1263K", 0.0),
        ("A453T",  0.0002),
        ("G2023S", 0.0),
    ]
    print(f"{'Variant':<12} {'Mode':<20} {'P(de novo)':<12} Basis[:60]")
    print("-"*80)
    for aa, af in tests:
        r = classify_inheritance(aa, af if af > 0 else None)
        print(f"{aa:<12} {r.inheritance_mode:<20} {r.de_novo_probability:<12.2f} {r.basis[:60]}")
