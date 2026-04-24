"""
chanvar/data/clinvar_parser.py
-------------------------------
ClinVar parser for CACNA1A variant classification data.

Extracts:
  - Pathogenic/Likely Pathogenic variants (gold standard positive set)
  - Benign/Likely Benign variants (gold standard negative set)
  - VUS (inference targets)
  - Associated phenotypes (FHM1, EA2, SCA6, unspecified migraine)
  - Review star ratings for quality stratification

Data source: ClinVar FTP
  ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
  ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarFullRelease_00-latest.xml.gz

Reference:
  Landrum et al. (2014). ClinVar: public archive of relationships among
  sequence variation and human phenotype. Nucleic Acids Research, 42(D1), D980–D985.
"""

import gzip
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ClinVar classification strings -> canonical categories
PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}
VUS_TERMS = {"Uncertain significance", "Uncertain_significance", "VUS"}

# Review star ratings (higher = more reliable)
REVIEW_STARS = {
    "no assertion provided": 0,
    "no assertion criteria provided": 0,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "reviewed by expert panel": 3,
    "practice guideline": 4,
}

# MedGen/OMIM phenotypes associated with CACNA1A
FHM1_CONDITIONS = {
    "Familial hemiplegic migraine",
    "Familial hemiplegic migraine type 1",
    "FHM1",
    "Migraine, familial hemiplegic, 1",
}
EA2_CONDITIONS = {
    "Episodic ataxia type 2",
    "Episodic ataxia, type 2",
    "EA2",
}
SCA6_CONDITIONS = {
    "Spinocerebellar ataxia type 6",
    "Spinocerebellar ataxia 6",
    "SCA6",
}


class ClinVarRecord:
    """
    Represents a single ClinVar classification record for a CACNA1A variant.

    Attributes
    ----------
    variant_id : str
        Variant in gnomAD format: 'chrom-pos-ref-alt'
    hgvsp : str
        Protein-level HGVS notation, e.g. 'NP_023035.1:p.Arg192Gln'
    aa_change : str
        Short amino acid change, e.g. 'R192Q'
    classification : str
        Canonical classification: 'pathogenic', 'benign', 'vus', 'conflicting', 'other'
    raw_classification : str
        Original ClinVar clinical significance string
    review_stars : int
        ClinVar review status star rating (0-4)
    review_status : str
        ClinVar review status string
    conditions : list of str
        Associated phenotype/condition names
    phenotype_class : str
        Assigned phenotype class: 'FHM1', 'EA2', 'SCA6', 'migraine_other', 'unknown'
    submitters : int
        Number of submitters (from ClinVar record)
    """

    def __init__(
        self,
        variant_id: str,
        hgvsp: str,
        aa_change: str,
        raw_classification: str,
        review_status: str,
        conditions: list,
        submitters: int = 1,
    ):
        self.variant_id = variant_id
        self.hgvsp = hgvsp
        self.aa_change = aa_change
        self.raw_classification = raw_classification
        self.review_status = review_status
        self.conditions = conditions
        self.submitters = submitters

        self.classification = self._canonicalize(raw_classification)
        self.review_stars = REVIEW_STARS.get(review_status.lower(), 0)
        self.phenotype_class = self._assign_phenotype(conditions)

    def _canonicalize(self, raw: str) -> str:
        """Map raw ClinVar classification to canonical category."""
        if any(term in raw for term in PATHOGENIC_TERMS):
            return "pathogenic"
        elif any(term in raw for term in BENIGN_TERMS):
            return "benign"
        elif any(term in raw for term in VUS_TERMS):
            return "vus"
        elif "Conflicting" in raw or "conflicting" in raw:
            return "conflicting"
        return "other"

    def _assign_phenotype(self, conditions: list) -> str:
        """
        Assign phenotype class from condition names.

        Priority: FHM1 > EA2 > SCA6 > migraine_other > unknown.
        """
        cond_set = set(conditions)
        if cond_set & FHM1_CONDITIONS:
            return "FHM1"
        if cond_set & EA2_CONDITIONS:
            return "EA2"
        if cond_set & SCA6_CONDITIONS:
            return "SCA6"
        if any("migraine" in c.lower() for c in conditions):
            return "migraine_other"
        return "unknown"

    def is_high_confidence(self, min_stars: int = 2) -> bool:
        """
        Check whether this record meets minimum confidence threshold.

        ACMG/AMP guidelines recommend 2+ star ClinVar records for
        computational predictor benchmarking.
        """
        return self.review_stars >= min_stars

    def __repr__(self) -> str:
        return (
            f"ClinVarRecord(aa_change={self.aa_change!r}, "
            f"classification={self.classification!r}, "
            f"stars={self.review_stars}, "
            f"phenotype={self.phenotype_class!r})"
        )


def parse_clinvar_vcf(vcf_path: str) -> list[ClinVarRecord]:
    """
    Parse a ClinVar VCF file for CACNA1A variants.

    Extracts clinical significance, review status, and condition information
    from the CLNSIG, CLNREVSTAT, and CLNDN INFO fields in the ClinVar VCF.

    Parameters
    ----------
    vcf_path : str
        Path to ClinVar VCF (can be .vcf.gz). Must be the genome build GRCh38 version.

    Returns
    -------
    list of ClinVarRecord
        All CACNA1A variants found, unfiltered by classification or confidence.

    Notes
    -----
    The ClinVar VCF uses pipe-delimited values in INFO fields (not comma-delimited).
    CLNDN uses '\\x2C' for literal commas within condition names.
    """
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    records = []

    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, _id, ref, alt, _qual, _flt, info_str = fields[:8]

            # Filter to chromosome 19 CACNA1A region
            chrom_clean = chrom.replace("chr", "")
            if chrom_clean != "19":
                continue
            try:
                pos_int = int(pos)
            except ValueError:
                continue
            if not (13_200_000 <= pos_int <= 13_520_000):
                continue

            # Parse INFO field
            info = _parse_info_field(info_str)

            # Must have clinical significance
            clnsig = info.get("CLNSIG", "")
            if not clnsig:
                continue

            # Only missense for chanvar (MC field or CLNVC)
            mc = info.get("MC", "")
            clnvc = info.get("CLNVC", "")
            is_missense = "missense_variant" in mc or "single_nucleotide_variant" in clnvc
            # Fallback: include if no MC specified (let downstream filter)

            # Extract fields
            review_status = info.get("CLNREVSTAT", "no assertion provided").replace("_", " ")
            conditions_raw = info.get("CLNDN", "not_specified")
            conditions = [c.replace("\\x2C", ",") for c in conditions_raw.split("|")]
            hgvsp = info.get("CLNHGVS", "")

            # Parse amino acid change from HGVS protein notation
            aa_change = _extract_aa_change(hgvsp)

            variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"

            record = ClinVarRecord(
                variant_id=variant_id,
                hgvsp=hgvsp,
                aa_change=aa_change,
                raw_classification=clnsig.replace("_", " "),
                review_status=review_status,
                conditions=conditions,
            )
            records.append(record)

    logger.info("Parsed %d CACNA1A ClinVar records", len(records))
    return records


def _parse_info_field(info_str: str) -> dict:
    """Parse VCF INFO string into key-value dict. Handles flag fields (no '=')."""
    result = {}
    for entry in info_str.split(";"):
        if "=" in entry:
            key, val = entry.split("=", 1)
            result[key] = val
        else:
            result[entry] = True
    return result


def _extract_aa_change(hgvs_str: str) -> str:
    """
    Extract short amino acid change code from HGVS protein notation.

    e.g. 'NP_023035.1:p.Arg192Gln' -> 'R192Q'
    Uses 3-letter to 1-letter amino acid code conversion.
    Returns empty string if parsing fails.
    """
    AA3_TO_1 = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        "Ter": "*", "Sec": "U",
    }

    import re
    match = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)", hgvs_str)
    if not match:
        return ""
    ref_aa3, pos, alt_aa3 = match.groups()
    ref_aa1 = AA3_TO_1.get(ref_aa3, ref_aa3[0])
    alt_aa1 = AA3_TO_1.get(alt_aa3, alt_aa3[0]) if alt_aa3 != "*" else "*"
    return f"{ref_aa1}{pos}{alt_aa1}"


def get_training_sets(
    records: list[ClinVarRecord],
    min_stars: int = 2,
    phenotype_filter: Optional[list] = None,
) -> tuple[list, list]:
    """
    Split ClinVar records into positive (P/LP) and negative (B/LB) training sets.

    Parameters
    ----------
    records : list of ClinVarRecord
        All parsed ClinVar records.
    min_stars : int
        Minimum review star rating. Default 2 (criteria provided, multiple submitters).
    phenotype_filter : list of str, optional
        If provided, only include records with phenotype_class in this list.
        e.g. ['FHM1', 'EA2'] to restrict to channelopathy phenotypes.

    Returns
    -------
    positives : list of ClinVarRecord
        High-confidence P/LP variants.
    negatives : list of ClinVarRecord
        High-confidence B/LB variants.

    Notes
    -----
    Conflicting interpretations records are excluded from both sets.
    This is the standard approach for ACMG/AMP calibration studies (Pejaver et al. 2022).
    """
    positives, negatives = [], []
    for rec in records:
        if not rec.is_high_confidence(min_stars):
            continue
        if phenotype_filter and rec.phenotype_class not in phenotype_filter:
            continue
        if rec.classification == "pathogenic":
            positives.append(rec)
        elif rec.classification == "benign":
            negatives.append(rec)

    logger.info(
        "Training set: %d P/LP, %d B/LB (min_stars=%d)",
        len(positives), len(negatives), min_stars,
    )
    return positives, negatives


def summarize_clinvar(records: list[ClinVarRecord]) -> dict:
    """Generate summary statistics for a set of ClinVar records."""
    from collections import Counter
    return {
        "total": len(records),
        "by_classification": dict(Counter(r.classification for r in records)),
        "by_phenotype": dict(Counter(r.phenotype_class for r in records)),
        "by_stars": dict(Counter(r.review_stars for r in records)),
        "high_confidence": sum(1 for r in records if r.is_high_confidence()),
    }
