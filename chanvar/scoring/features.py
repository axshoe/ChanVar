"""
chanvar/scoring/features.py
-----------------------------
Feature vector construction for the ChanVar Pathogenicity Score (CPS).

Assembles f1–f8 from all data sources into a standardized feature vector
for each CACNA1A missense variant. All features are normalized to [0, 1]
before CPS computation.

Feature inventory:
  f1: gnomAD allele frequency (inverse: rarity = higher pathogenicity prior)
  f2: FoldX/EvoEF2 dDDG thermodynamic destabilization
  f3: Local backbone RMSD between wildtype and mutant structure
  f4: Evolutionary conservation (rate4site, PhyloP)
  f5: Functional domain membership weight
  f6: AlphaMissense pathogenicity score
  f7: CADD Phred score (normalized)
  f8: Grantham physicochemical distance (normalized)

All eight features plus metadata are returned as a VariantFeatures dataclass.
"""

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from chanvar.data.gnomad_client import compute_frequency_feature
from chanvar.data.alphafold_client import (
    get_domain_for_residue,
    DOMAIN_PATHOGENICITY_WEIGHTS,
)
from chanvar.structure.stability import (
    compute_ddg_feature,
    compute_rmsd_feature,
    compute_grantham_feature,
)
from chanvar.evolution.conservation import (
    compute_conservation_feature,
    get_conservation_for_variant,
)

logger = logging.getLogger(__name__)

# AlphaMissense score boundaries (Cheng et al. 2023, Science)
ALPHAMISSENSE_THRESHOLDS = {
    "benign": 0.34,
    "likely_pathogenic": 0.56,
}

# CADD Phred normalization: Phred 40 maps to f7=1.0
CADD_PHRED_MAX = 40.0


@dataclass
class VariantFeatures:
    """
    Complete feature vector for a single CACNA1A missense variant.

    All f-values are in [0, 1] where higher = more evidence for pathogenicity.
    None indicates missing data; handled in CPS computation with neutral imputation (0.5).

    Attributes
    ----------
    variant_id : str
        gnomAD variant identifier.
    aa_change : str
        Short amino acid change (e.g., 'R192Q').
    ref_aa : str
        Reference amino acid (single-letter).
    alt_aa : str
        Alternate amino acid (single-letter).
    residue_number : int
        Amino acid position (1-indexed, UniProt O00555).
    domain : str
        Functional domain classification.
    f1 : float or None
        Frequency feature from gnomAD AF.
    f2 : float or None
        Structural stability (dDDG) feature.
    f3 : float or None
        Local RMSD feature.
    f4 : float or None
        Evolutionary conservation feature.
    f5 : float
        Functional domain weight.
    f6 : float or None
        AlphaMissense score (direct; already 0-1).
    f7 : float or None
        CADD Phred normalized score.
    f8 : float or None
        Grantham physicochemical distance normalized score.
    ddg_raw : float or None
        Raw dDDG in kcal/mol (for reporting).
    rmsd_raw : float or None
        Raw RMSD in Angstroms (for reporting).
    rate4site_raw : float or None
        Raw rate4site normalized rate.
    gnomad_af : float or None
        Raw gnomAD allele frequency.
    clinvar_sig : str or None
        ClinVar clinical significance (if known).
    plddt : float or None
        AlphaFold2 pLDDT confidence at this position.
    is_tm_domain : bool
        True if variant is in a transmembrane helix (S1-S6).
        Affects dDDG reliability flag.
    data_completeness : float
        Fraction of f1-f8 that are non-None (data quality metric).
    """

    variant_id: str
    aa_change: str
    ref_aa: str
    alt_aa: str
    residue_number: int
    domain: str

    # Core features
    f1: Optional[float] = None
    f2: Optional[float] = None
    f3: Optional[float] = None
    f4: Optional[float] = None
    f5: float = 0.5
    f6: Optional[float] = None
    f7: Optional[float] = None
    f8: Optional[float] = None

    # Raw values for reporting
    ddg_raw: Optional[float] = None
    rmsd_raw: Optional[float] = None
    rate4site_raw: Optional[float] = None
    gnomad_af: Optional[float] = None
    clinvar_sig: Optional[str] = None
    plddt: Optional[float] = None
    is_tm_domain: bool = False

    @property
    def feature_vector(self) -> list[float]:
        """
        Return [f1, f2, f3, f4, f5, f6, f7, f8] with None replaced by 0.5 (neutral).

        Used as input to CPS weighted sum.
        """
        return [
            f if f is not None else 0.5
            for f in [self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7, self.f8]
        ]

    @property
    def data_completeness(self) -> float:
        """Fraction of f1–f8 with non-None values."""
        raw = [self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7, self.f8]
        return sum(1 for f in raw if f is not None) / 8.0

    def to_dict(self) -> dict:
        return asdict(self)


TM_DOMAINS = {"voltage_sensor", "pore_lining", "selectivity_filter"}


def build_feature_vector(
    variant_id: str,
    aa_change: str,
    gnomad_af: Optional[float] = None,
    ddg_foldx: Optional[float] = None,
    ddg_evoef2: Optional[float] = None,
    rmsd: Optional[float] = None,
    conservation_scores: Optional[dict] = None,
    alphamissense_score: Optional[float] = None,
    cadd_phred: Optional[float] = None,
    clinvar_sig: Optional[str] = None,
    plddt: Optional[float] = None,
) -> VariantFeatures:
    """
    Assemble the ChanVar feature vector for a single missense variant.

    Parameters
    ----------
    variant_id : str
        gnomAD format variant ID (e.g., '19-13392445-G-A').
    aa_change : str
        Amino acid change in single-letter format (e.g., 'R192Q').
    gnomad_af : float or None
        gnomAD overall allele frequency.
    ddg_foldx : float or None
        FoldX dDDG (kcal/mol).
    ddg_evoef2 : float or None
        EvoEF2 dDDG (kcal/mol).
    rmsd : float or None
        Local backbone RMSD (Å) between WT and mutant structure.
    conservation_scores : dict or None
        Pre-loaded conservation score dict from load_conservation_scores().
    alphamissense_score : float or None
        AlphaMissense pathogenicity score (0-1).
    cadd_phred : float or None
        CADD Phred score.
    clinvar_sig : str or None
        ClinVar clinical significance string (e.g. "Pathogenic", "Benign", "Uncertain significance").
        Used to compute f6 when alphamissense_score is not provided.
    plddt : float or None
        AlphaFold2 pLDDT confidence at variant position.

    Returns
    -------
    VariantFeatures
        Complete feature vector ready for CPS computation.

    Notes
    -----
    Conservation: if conservation_scores is None, the bundled cacna1a_conservation.tsv
    is loaded automatically. Pass conservation_scores={} to explicitly disable.

    dDDG consensus: if both FoldX and EvoEF2 are available, use their mean.

    f6 priority: AlphaMissense score if provided, otherwise derived from clinvar_sig
    (Pathogenic/LP -> 1.0, Benign/LB -> 0.0, VUS/absent -> 0.5).
    """
    # Parse amino acid change
    ref_aa, residue_number, alt_aa = _parse_aa_change(aa_change)
    if ref_aa is None:
        raise ValueError(f"Cannot parse amino acid change: {aa_change!r}")

    # Domain classification
    domain = get_domain_for_residue(residue_number)
    is_tm = domain in TM_DOMAINS

    # f1: gnomAD frequency
    f1 = compute_frequency_feature(gnomad_af)

    # f2: Structural stability (dDDG consensus)
    ddg_values = [d for d in [ddg_foldx, ddg_evoef2] if d is not None]
    if ddg_values:
        ddg_consensus = sum(ddg_values) / len(ddg_values)
        if len(ddg_values) == 2 and abs(ddg_foldx - ddg_evoef2) > 3.0:
            logger.warning(
                "FoldX/EvoEF2 dDDG discordance for %s: %.2f vs %.2f kcal/mol. "
                "Using mean; interpret CPS with caution.",
                aa_change, ddg_foldx, ddg_evoef2,
            )
    else:
        ddg_consensus = None
    f2 = compute_ddg_feature(ddg_consensus, domain)
    if ddg_consensus is None:
        f2 = None

    # f3: Local RMSD
    f3 = compute_rmsd_feature(rmsd) if rmsd is not None else None

    # f4: Evolutionary conservation — auto-load bundled TSV if not provided
    if conservation_scores is None:
        try:
            import pathlib as _pathlib
            _tsv = _pathlib.Path(__file__).parent.parent / "data" / "cacna1a_conservation.tsv"
            if _tsv.exists():
                from chanvar.evolution.conservation import load_conservation_scores
                conservation_scores = load_conservation_scores(str(_tsv))
        except Exception:
            conservation_scores = None

    if conservation_scores:
        cons = get_conservation_for_variant(residue_number, conservation_scores)
        f4 = compute_conservation_feature(
            rate4site=cons.get("rate4site"),
            phylop=cons.get("phylop"),
        )
        rate4site_raw = cons.get("rate4site")
    else:
        f4 = None
        rate4site_raw = None

    # f5: Functional domain weight (deterministic from domain classification)
    f5 = DOMAIN_PATHOGENICITY_WEIGHTS.get(domain, 0.5)

    # f6: AlphaMissense if provided; otherwise derive from ClinVar significance string
    if alphamissense_score is not None:
        f6 = max(0.0, min(1.0, float(alphamissense_score)))
    elif clinvar_sig is not None:
        sig_lower = clinvar_sig.lower().replace("_", " ").replace("-", " ")
        if "pathogenic" in sig_lower and "likely" in sig_lower:
            f6 = 0.9
        elif "pathogenic" in sig_lower:
            f6 = 1.0
        elif "benign" in sig_lower and "likely" in sig_lower:
            f6 = 0.1
        elif "benign" in sig_lower:
            f6 = 0.0
        elif "uncertain" in sig_lower or "vus" in sig_lower:
            f6 = 0.5
        else:
            f6 = None
    else:
        f6 = None

    # f7: CADD Phred normalized
    if cadd_phred is not None:
        f7 = min(1.0, float(cadd_phred) / CADD_PHRED_MAX)
    else:
        f7 = None

    # f8: Grantham physicochemical distance
    f8 = compute_grantham_feature(ref_aa, alt_aa)

    return VariantFeatures(
        variant_id=variant_id,
        aa_change=aa_change,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        residue_number=residue_number,
        domain=domain,
        f1=f1,
        f2=f2,
        f3=f3,
        f4=f4,
        f5=f5,
        f6=f6,
        f7=f7,
        f8=f8,
        ddg_raw=ddg_consensus,
        rmsd_raw=rmsd,
        rate4site_raw=rate4site_raw,
        gnomad_af=gnomad_af,
        clinvar_sig=clinvar_sig,
        plddt=plddt,
        is_tm_domain=is_tm,
    )


def _parse_aa_change(aa_change: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse amino acid change string (e.g., 'R192Q') into components.

    Parameters
    ----------
    aa_change : str
        Amino acid change. Accepts formats: 'R192Q', 'Arg192Gln', 'p.R192Q'.

    Returns
    -------
    tuple of (ref_aa_1letter, position, alt_aa_1letter)
        All None if parsing fails.
    """
    import re

    AA3_TO_1 = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    }

    # Remove 'p.' prefix if present
    s = aa_change.replace("p.", "")

    # Try 3-letter codes: Arg192Gln
    m3 = re.match(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)", s)
    if m3:
        ref3, pos, alt3 = m3.groups()
        return (
            AA3_TO_1.get(ref3),
            int(pos),
            AA3_TO_1.get(alt3, "*") if alt3 != "*" else "*",
        )

    # Try 1-letter codes: R192Q
    m1 = re.match(r"([A-Z])(\d+)([A-Z\*])", s)
    if m1:
        return m1.group(1), int(m1.group(2)), m1.group(3)

    return None, None, None


def batch_build_features(
    variant_list: list[dict],
    conservation_scores: Optional[dict] = None,
    pdb_path: Optional[str] = None,
    run_foldx: bool = False,
) -> list[VariantFeatures]:
    """
    Build feature vectors for a list of variants.

    Parameters
    ----------
    variant_list : list of dict
        Each dict should contain at minimum: 'variant_id', 'aa_change'.
        Optional keys: all parameters accepted by build_feature_vector.
    conservation_scores : dict, optional
        Pre-loaded conservation data (avoids reloading for each variant).
    pdb_path : str, optional
        Path to the AlphaFold2 or repaired PDB structure. If provided and
        run_foldx=True, FoldX will be called for each variant to compute ddG.
        Use the pre-repaired PDB (e.g. AF-O00555-F1-model_v4_Repair.pdb) to
        avoid re-running RepairPDB for every variant.
    run_foldx : bool
        If True and pdb_path is set, call FoldX for each variant. This adds
        3-8 minutes per variant. Default False.

    Returns
    -------
    list of VariantFeatures
        One VariantFeatures per input variant. Variants that fail parsing
        are skipped with a warning.
    """
    results = []

    def _float(val):
        """Convert CSV string to float, return None if empty or unparseable."""
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Import FoldX runner only if needed
    foldx_runner = None
    if run_foldx and pdb_path:
        try:
            from chanvar.structure.stability import run_foldx_ddg
            foldx_runner = run_foldx_ddg
            logger.info("FoldX integration enabled. PDB: %s", pdb_path)
            logger.info("Estimated time: %d--%d minutes for %d variants",
                        len(variant_list) * 3, len(variant_list) * 8, len(variant_list))
        except ImportError:
            logger.warning("Could not import run_foldx_ddg. FoldX will not run.")

    for i, var in enumerate(variant_list):
        try:
            aa_change = var["aa_change"]

            # Parse amino acid change for FoldX call
            ddg_foldx_val = _float(var.get("ddg_foldx") or var.get("ddg"))

            # Run FoldX if enabled and ddG not already provided in CSV
            if foldx_runner and pdb_path and ddg_foldx_val is None:
                import re
                m = re.match(r"([A-Z])(\d+)([A-Z])", aa_change)
                if m:
                    ref_aa, pos, alt_aa = m.group(1), int(m.group(2)), m.group(3)
                    logger.info(
                        "[%d/%d] Running FoldX for %s...",
                        i + 1, len(variant_list), aa_change
                    )
                    try:
                        ddg_foldx_val = foldx_runner(
                            pdb_path=pdb_path,
                            chain="A",
                            residue_number=pos,
                            ref_aa=ref_aa,
                            alt_aa=alt_aa,
                        )
                        if ddg_foldx_val is not None:
                            logger.info("  ddG(%s) = %.3f kcal/mol", aa_change, ddg_foldx_val)
                        else:
                            logger.warning("  FoldX returned None for %s", aa_change)
                    except Exception as fx_exc:
                        logger.warning("  FoldX failed for %s: %s", aa_change, fx_exc)
                        ddg_foldx_val = None

            fv = build_feature_vector(
                variant_id=var.get("variant_id", f"unknown_{i}"),
                aa_change=aa_change,
                gnomad_af=_float(var.get("af")),
                ddg_foldx=ddg_foldx_val,
                ddg_evoef2=_float(var.get("ddg_evoef2")),
                rmsd=_float(var.get("rmsd")),
                conservation_scores=conservation_scores,
                alphamissense_score=_float(var.get("alphamissense")),
                cadd_phred=_float(var.get("cadd_phred") or var.get("cadd")),
                clinvar_sig=var.get("clinvar_sig") or var.get("clinvar") or None,
                plddt=_float(var.get("plddt")),
            )
            results.append(fv)
        except Exception as exc:
            logger.warning("Feature vector failed for variant %s: %s", var.get("aa_change", "?"), exc)

    logger.info("Built %d/%d feature vectors", len(results), len(variant_list))
    return results