"""
chanvar/scoring/features.py
-----------------------------
Feature vector construction for the ChanVar Pathogenicity Score (CPS).

Assembles f1-f9 from all data sources into a standardized feature vector
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
  f9: Pore-axis distance (Brunger et al. Brain 2023)
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

ALPHAMISSENSE_THRESHOLDS = {
    "benign": 0.34,
    "likely_pathogenic": 0.56,
}

CADD_PHRED_MAX = 40.0

TM_DOMAINS = {"voltage_sensor", "pore_lining", "selectivity_filter"}


@dataclass
class VariantFeatures:
    """
    Complete feature vector for a single CACNA1A missense variant.

    All f-values are in [0, 1] where higher = more evidence for pathogenicity.
    None indicates missing data; handled in CPS computation with neutral
    imputation (0.5).
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
    f9: Optional[float] = None   # pore-axis distance

    # Raw values for reporting
    ddg_raw: Optional[float] = None
    rmsd_raw: Optional[float] = None
    rate4site_raw: Optional[float] = None
    gnomad_af: Optional[float] = None
    clinvar_sig: Optional[str] = None
    plddt: Optional[float] = None
    is_tm_domain: bool = False

    @property
    def feature_vector(self) -> list:
        """
        Return [f1..f9] with None replaced by 0.5 (neutral imputation).
        Used as input to CPS weighted sum.
        """
        return [
            f if f is not None else 0.5
            for f in [
                self.f1, self.f2, self.f3, self.f4,
                self.f5, self.f6, self.f7, self.f8, self.f9
            ]
        ]

    @property
    def data_completeness(self) -> float:
        """Fraction of f1-f9 with non-None values (f3 excluded as inactive)."""
        # f3 is reserved/inactive so we use 8 active features: f1,f2,f4,f5,f6,f7,f8,f9
        active = [self.f1, self.f2, self.f4, self.f5, self.f6, self.f7, self.f8, self.f9]
        return sum(1 for f in active if f is not None) / 8.0

    def to_dict(self) -> dict:
        return asdict(self)


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
    pdb_path: Optional[str] = None,
) -> VariantFeatures:
    """
    Assemble the ChanVar feature vector for a single missense variant.

    Parameters
    ----------
    variant_id : str
    aa_change : str
        e.g. 'R192Q'
    gnomad_af : float or None
    ddg_foldx : float or None
    ddg_evoef2 : float or None
    rmsd : float or None
    conservation_scores : dict or None
    alphamissense_score : float or None
    cadd_phred : float or None
    clinvar_sig : str or None
    plddt : float or None
    pdb_path : str or None
        If provided, f9 pore distance is computed from the PDB structure.

    Returns
    -------
    VariantFeatures
    """
    ref_aa, residue_number, alt_aa = _parse_aa_change(aa_change)
    if ref_aa is None:
        raise ValueError(f"Cannot parse amino acid change: {aa_change!r}")

    domain = get_domain_for_residue(residue_number)
    is_tm = domain in TM_DOMAINS

    # f1: gnomAD frequency
    f1 = compute_frequency_feature(gnomad_af)

    # f2: Structural stability (dDDG consensus)
    ddg_values = [d for d in [ddg_foldx, ddg_evoef2] if d is not None]
    if ddg_values:
        ddg_consensus = sum(ddg_values) / len(ddg_values)
        if (len(ddg_values) == 2
                and ddg_foldx is not None
                and ddg_evoef2 is not None
                and abs(ddg_foldx - ddg_evoef2) > 3.0):
            logger.warning(
                "FoldX/EvoEF2 dDDG discordance for %s: %.2f vs %.2f kcal/mol.",
                aa_change, ddg_foldx, ddg_evoef2,
            )
    else:
        ddg_consensus = None

    f2 = compute_ddg_feature(ddg_consensus, domain) if ddg_consensus is not None else None

    # f3: Local RMSD (currently inactive placeholder)
    f3 = compute_rmsd_feature(rmsd) if rmsd is not None else None

    # f4: Evolutionary conservation
    if conservation_scores is None:
        try:
            import pathlib as _pathlib
            _tsv = (
                _pathlib.Path(__file__).parent.parent
                / "data"
                / "cacna1a_conservation.tsv"
            )
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

    # f5: Functional domain weight
    f5 = DOMAIN_PATHOGENICITY_WEIGHTS.get(domain, 0.5)

    # f6: AlphaMissense or ClinVar-derived prior
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
    f7 = min(1.0, float(cadd_phred) / CADD_PHRED_MAX) if cadd_phred is not None else None

    # f8: Grantham physicochemical distance
    f8 = compute_grantham_feature(ref_aa, alt_aa)

    # f9: Pore-axis distance
    f9 = None
    if pdb_path is not None:
        try:
            from chanvar.structure.pore_distance import get_f9
            raw_f9 = get_f9(residue_number, pdb_path=pdb_path)
            # Apply pLDDT correction: unreliable structure -> neutral
            if plddt is not None and plddt < 70:
                f9 = 0.5
            else:
                f9 = raw_f9
        except Exception as e:
            logger.warning("f9 pore distance failed for %s: %s", aa_change, e)
            f9 = None
    else:
        # Use precomputed fallback table (no PDB required)
        try:
            from chanvar.structure.pore_distance import get_f9
            f9 = get_f9(residue_number)
        except Exception:
            f9 = None

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
        f9=f9,
        ddg_raw=ddg_consensus,
        rmsd_raw=rmsd,
        rate4site_raw=rate4site_raw,
        gnomad_af=gnomad_af,
        clinvar_sig=clinvar_sig,
        plddt=plddt,
        is_tm_domain=is_tm,
    )


def _parse_aa_change(aa_change: str):
    """
    Parse amino acid change string (e.g., 'R192Q') into components.

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
    variant_list: list,
    conservation_scores: Optional[dict] = None,
    pdb_path: Optional[str] = None,
    run_foldx: bool = False,
) -> list:
    """
    Build feature vectors for a list of variants.

    Parameters
    ----------
    variant_list : list of dict
        Each dict must have 'variant_id' and 'aa_change'.
        Optional keys: af, ddg_foldx, ddg, ddg_evoef2, rmsd,
        alphamissense, cadd_phred, cadd, clinvar_sig, clinvar, plddt.
    conservation_scores : dict, optional
    pdb_path : str, optional
        Path to AlphaFold2 PDB. Used for f9 computation and optionally FoldX.
    run_foldx : bool
        If True and pdb_path set, calls FoldX for each variant.

    Returns
    -------
    list of VariantFeatures
    """
    results = []

    def _float(val):
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Load conservation scores once for the whole batch
    if conservation_scores is None:
        try:
            import pathlib as _pathlib
            _tsv = (
                _pathlib.Path(__file__).parent.parent
                / "data"
                / "cacna1a_conservation.tsv"
            )
            if _tsv.exists():
                from chanvar.evolution.conservation import load_conservation_scores
                conservation_scores = load_conservation_scores(str(_tsv))
        except Exception:
            conservation_scores = None

    # Set up FoldX runner if requested
    foldx_runner = None
    if run_foldx and pdb_path:
        try:
            from chanvar.structure.stability import run_foldx_ddg
            foldx_runner = run_foldx_ddg
            logger.info("FoldX enabled. PDB: %s", pdb_path)
        except ImportError:
            logger.warning("Could not import run_foldx_ddg.")

    # Load pore distances once for all variants (single PDB parse)
    pore_f9_cache = {}
    if pdb_path:
        try:
            from chanvar.structure.pore_distance import compute_pore_distances
            pore_f9_cache = compute_pore_distances(pdb_path)
            logger.info("Pore distances loaded for %d residues.", len(pore_f9_cache))
        except Exception as e:
            logger.warning("Pore distance batch load failed: %s", e)

    for i, var in enumerate(variant_list):
        try:
            aa_change = var["aa_change"]
            ddg_foldx_val = _float(var.get("ddg_foldx") or var.get("ddg"))

            # Optionally run FoldX
            if foldx_runner and pdb_path and ddg_foldx_val is None:
                import re
                m = re.match(r"([A-Z])(\d+)([A-Z])", aa_change)
                if m:
                    ref_aa, pos, alt_aa = m.group(1), int(m.group(2)), m.group(3)
                    logger.info("[%d/%d] FoldX: %s", i + 1, len(variant_list), aa_change)
                    try:
                        ddg_foldx_val = foldx_runner(
                            pdb_path=pdb_path,
                            chain="A",
                            residue_number=pos,
                            ref_aa=ref_aa,
                            alt_aa=alt_aa,
                        )
                    except Exception as fx_exc:
                        logger.warning("FoldX failed for %s: %s", aa_change, fx_exc)

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
                pdb_path=None,  # f9 handled below using the pre-loaded cache
            )

            # Apply f9 from pre-loaded cache
            plddt_val = _float(var.get("plddt"))
            if pore_f9_cache:
                raw_f9 = pore_f9_cache.get(fv.residue_number, 0.5)
                fv.f9 = 0.5 if (plddt_val is not None and plddt_val < 70) else raw_f9
            else:
                # Fall back to precomputed table
                try:
                    from chanvar.structure.pore_distance import PRECOMPUTED_F9
                    fv.f9 = PRECOMPUTED_F9.get(fv.residue_number, 0.5)
                except Exception:
                    fv.f9 = 0.5

            results.append(fv)

        except Exception as exc:
            logger.warning(
                "Feature vector failed for %s: %s",
                var.get("aa_change", "?"), exc
            )

    logger.info("Built %d/%d feature vectors.", len(results), len(variant_list))
    return results