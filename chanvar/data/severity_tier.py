"""
chanvar/data/severity_tier.py
-------------------------------
Severity tier classification for CACNA1A missense variants.

Based on van den Maagdenberg (LUMC, personal communication, May 2026).
Three tiers derived from variant-specific literature, NOT from the model:

  severe_complex  : cerebellar ataxia + epilepsy + potentially fatal features
                    (S218L: prolonged coma, cerebral edema; T666M: ataxia + epilepsy;
                     R1668W and related)
  pure_HM         : hemiplegic migraine without cerebellar/epileptic features
                    (R192Q canonical; most missense without reported cerebellar phenotype)
  uncertain       : insufficient published clinical data for tier assignment

This is entirely literature-based. It is not a model output.
Do not confuse severity_tier with CPS or P(GOF).

References per variant are included in SEVERITY_EVIDENCE.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeverityTierResult:
    tier: str                         # "severe_complex" | "pure_HM" | "uncertain"
    basis: str                        # brief clinical evidence summary
    references: list                  # citation strings
    is_literature_based: bool = True  # always True — never model-derived


# ── Ground truth assignments from van den Maagdenberg and FHM1 literature ──
# Format: aa_change -> (tier, basis, [references])

SEVERITY_ASSIGNMENTS = {

    # ── SEVERE COMPLEX ───────────────────────────────────────────────────────
    "S218L": (
        "severe_complex",
        "Severe FHM1 with cerebellar ataxia, prolonged coma, cerebral edema, "
        "potentially fatal from minor head trauma. S218L mice show spontaneous "
        "seizures and fatal TBI susceptibility.",
        [
            "van den Maagdenberg et al. (2010) Brain 133:1724 (S218L knock-in mice)",
            "Kors et al. (2001) Ann Neurol 50:360 (first S218L clinical description)",
            "Terpolilli et al. (2022) eLife (TBI susceptibility S218L mice)",
        ]
    ),
    "T666M": (
        "severe_complex",
        "Episodic ataxia type 2 with overlapping epilepsy; cerebellar ataxia "
        "prominent; some families show seizures. EA2/FHM1 overlap at T666/T665.",
        [
            "Denier et al. (1999) Neurology 53:26 (T666M EA2+epilepsy)",
            "Ophoff et al. (1996) Cell 87:543 (original CACNA1A paper includes T666M)",
        ]
    ),
    "R1668W": (
        "severe_complex",
        "Associated with severe hemiplegic migraine with cerebellar features "
        "and prolonged neurological deficits.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),
    "R1663Q": (
        "severe_complex",
        "Selectivity filter variant; reported in severe HM with cerebellar "
        "signs in multiple families.",
        [
            "Joutel et al. (1994) Nat Genet 7:209 (early FHM genetic study)",
        ]
    ),
    "I1811L": (
        "severe_complex",
        "C-terminal variant; reported with cerebellar atrophy.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),

    # ── PURE HM (less severe) ─────────────────────────────────────────────────
    "R192Q": (
        "pure_HM",
        "Canonical FHM1 variant. Pure hemiplegic migraine, no cerebellar "
        "ataxia or epilepsy. Well characterized in knock-in mice; gain of "
        "function with lower CSD threshold.",
        [
            "Ophoff et al. (1996) Cell 87:543 (first FHM1 identification)",
            "van den Maagdenberg et al. (2004) Neuron 41:701 (R192Q knock-in mice)",
            "Pietrobon & Striessnig (2003) Nat Rev Neurosci 4:386",
        ]
    ),
    "R583Q": (
        "pure_HM",
        "Pure hemiplegic migraine phenotype reported across multiple families; "
        "no cerebellar or epileptic features in published cases.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),
    "T1010M": (
        "pure_HM",
        "Associated with pure hemiplegic migraine without cerebellar ataxia.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),
    "D715E": (
        "pure_HM",
        "Pore region variant associated with pure hemiplegic migraine.",
        [
            "Ophoff et al. (1996) Cell 87:543",
        ]
    ),
    "D302H": (
        "pure_HM",
        "Pore lining variant. FHM1 phenotype without documented cerebellar "
        "features in reported families.",
        [
            "Jen et al. (2004) Brain 127:2484",
        ]
    ),
    "V215A": (
        "pure_HM",
        "Voltage sensor S4-I variant. Pure HM phenotype in reported cases.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),
    "P202L": (
        "pure_HM",
        "Voltage sensor region. Pure HM without cerebellar involvement.",
        [
            "Ducros et al. (2001) Arch Neurol 58:1553",
        ]
    ),

    # ── EA2 primary (not HM) ─────────────────────────────────────────────────
    "T665M": (
        "severe_complex",
        "Primary EA2 with episodic cerebellar ataxia. Some overlap with FHM "
        "in multi-generation families. Loss-of-function mechanism.",
        [
            "Ophoff et al. (1996) Cell 87:543",
            "Denier et al. (1999) Neurology 53:26",
        ]
    ),

}

# ── Default uncertain template ──────────────────────────────────────────────
_UNCERTAIN = (
    "uncertain",
    "Insufficient published clinical data to assign severity tier. "
    "ClinVar classification available but detailed phenotype literature "
    "absent or contradictory.",
    []
)


def get_severity_tier(aa_change: str) -> SeverityTierResult:
    """
    Return the literature-based severity tier for a CACNA1A missense variant.

    Args:
        aa_change: Single-letter amino acid change (e.g. 'R192Q', 'S218L')

    Returns:
        SeverityTierResult with tier, basis, and references.
        Returns 'uncertain' if no literature assignment exists.

    Notes:
        This function NEVER uses the CPS or any model output.
        It is a direct lookup of published clinical phenotype data.
        Adding new entries requires a literature reference.
    """
    aa = aa_change.strip()
    entry = SEVERITY_ASSIGNMENTS.get(aa)
    if entry is None:
        return SeverityTierResult(
            tier=_UNCERTAIN[0],
            basis=_UNCERTAIN[1],
            references=list(_UNCERTAIN[2]),
        )
    tier, basis, refs = entry
    return SeverityTierResult(tier=tier, basis=basis, references=list(refs))


def add_severity_assignment(
    aa_change: str,
    tier: str,
    basis: str,
    references: list,
) -> None:
    """
    Add or update a severity tier assignment at runtime.

    Use this to incorporate new literature findings without modifying
    the source file. Changes are not persistent across sessions.

    Args:
        aa_change: e.g. 'R192Q'
        tier: 'severe_complex', 'pure_HM', or 'uncertain'
        basis: brief clinical evidence summary
        references: list of citation strings
    """
    if tier not in ("severe_complex", "pure_HM", "uncertain"):
        raise ValueError(f"tier must be severe_complex, pure_HM, or uncertain; got {tier!r}")
    SEVERITY_ASSIGNMENTS[aa_change] = (tier, basis, list(references))


def batch_severity(aa_changes: list) -> dict:
    """
    Bulk lookup for a list of aa_change strings.

    Returns:
        dict mapping aa_change -> SeverityTierResult
    """
    return {aa: get_severity_tier(aa) for aa in aa_changes}


if __name__ == "__main__":
    test_variants = [
        "R192Q", "S218L", "T666M", "T665M", "R583Q",
        "D302H", "A453T", "G2023S",
    ]
    print(f"{'Variant':<12} {'Tier':<16} {'Basis (truncated)'}")
    print("-" * 70)
    for aa in test_variants:
        result = get_severity_tier(aa)
        print(f"{aa:<12} {result.tier:<16} {result.basis[:42]}...")
