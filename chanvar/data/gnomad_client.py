"""
chanvar/data/gnomad_client.py
------------------------------
gnomAD v4 interface for CACNA1A variant queries.

Supports:
  - REST API queries for individual variants
  - Batch VCF parsing for genome-wide CACNA1A variant sets
  - Per-ancestry allele frequency extraction
  - Constraint metric extraction (pLI, LOEUF, MPC)

gnomAD v4 REST API: https://gnomad.broadinstitute.org/api
gnomAD VCF download: https://gnomad.broadinstitute.org/downloads
CACNA1A region (hg38): chr19:13,206,000-13,512,000

Reference: Karczewski et al. (2020). The mutational constraint spectrum from
variation in 141,456 humans. Nature, 581, 434–443.
"""

import time
import logging
import math
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# CACNA1A genomic coordinates (hg38)
CACNA1A_CHROM = "19"
CACNA1A_START = 13_206_000
CACNA1A_END = 13_512_000
CACNA1A_GENE_ID = "ENSG00000141837"
CACNA1A_UNIPROT = "O00555"

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

# Ancestry group keys in gnomAD v4
ANCESTRY_GROUPS = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas", "mid", "remaining"]


def _graphql_query(query: str, variables: dict, retries: int = 3) -> dict:
    """Execute a GraphQL query against the gnomAD API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                GNOMAD_API_URL,
                json={"query": query, "variables": variables},
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logger.warning("gnomAD API returned errors: %s", data["errors"])
            return data.get("data", {})
        except requests.RequestException as exc:
            logger.warning("gnomAD API attempt %d/%d failed: %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError("gnomAD API unavailable after %d attempts" % retries)


_VARIANT_QUERY = """
query VariantQuery($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
    variantId
    chrom
    pos
    ref
    alt
    exome {
      ac
      an
      af
      ac_hom
      populations {
        id
        ac
        an
        af
      }
    }
    genome {
      ac
      an
      af
      ac_hom
      populations {
        id
        ac
        an
        af
      }
    }
    rsid
    in_silico_predictors {
      id
      value
      flags
    }
    consequence
    hgvsc
    hgvsp
    transcript_consequence {
      amino_acids
      codons
      consequence_terms
      lof
    }
    clinvar_variation_id
    clinvar {
      clinical_significance
      review_status
      conditions {
        name
      }
    }
  }
}
"""

_GENE_VARIANTS_QUERY = """
query GeneVariants($geneId: String!, $datasetId: DatasetId!) {
  gene(gene_id: $geneId, reference_genome: GRCh38) {
    variants(dataset: $datasetId) {
      variantId
      pos
      ref
      alt
      consequence
      hgvsp
      exome {
        ac
        an
        af
      }
    }
  }
}
"""


def query_variant(variant_id: str, dataset: str = "gnomad_r4") -> Optional[dict]:
    """
    Query a single variant from gnomAD.

    Parameters
    ----------
    variant_id : str
        gnomAD variant ID in format 'chrom-pos-ref-alt' e.g. '19-13392445-G-A'
    dataset : str
        gnomAD dataset ID. Use 'gnomad_r4' for gnomAD v4 exomes+genomes.

    Returns
    -------
    dict or None
        Parsed variant record with frequency and annotation fields.
        Returns None if variant not found in gnomAD.

    Notes
    -----
    gnomAD REST API rate limit: ~10 requests/second. The client includes
    exponential backoff on 429/503 responses.
    """
    data = _graphql_query(
        _VARIANT_QUERY,
        {"variantId": variant_id, "datasetId": dataset},
    )
    variant = data.get("variant")
    if variant is None:
        logger.info("Variant %s not found in gnomAD %s", variant_id, dataset)
        return None
    return _parse_variant_record(variant)


def _parse_variant_record(raw: dict) -> dict:
    """
    Normalize a gnomAD variant record into the ChanVar internal format.

    Merges exome + genome counts (gnomAD v4 separates these).
    Computes overall AF as (AC_exome + AC_genome) / (AN_exome + AN_genome).
    """
    exome = raw.get("exome") or {}
    genome = raw.get("genome") or {}

    ac_total = (exome.get("ac") or 0) + (genome.get("ac") or 0)
    an_total = (exome.get("an") or 0) + (genome.get("an") or 0)
    af_total = ac_total / an_total if an_total > 0 else 0.0
    ac_hom = (exome.get("ac_hom") or 0) + (genome.get("ac_hom") or 0)

    # Per-ancestry frequencies (from exome; genome if exome absent)
    ancestry_afs = {}
    pop_source = exome.get("populations") or genome.get("populations") or []
    for pop in pop_source:
        pop_id = pop["id"].lower()
        if pop["an"] and pop["an"] > 0:
            ancestry_afs[pop_id] = pop["ac"] / pop["an"]
        else:
            ancestry_afs[pop_id] = None

    # ClinVar integration
    clinvar = raw.get("clinvar") or {}
    clinvar_sig = clinvar.get("clinical_significance")
    clinvar_review = clinvar.get("review_status")
    clinvar_conditions = [c["name"] for c in clinvar.get("conditions") or []]

    # In silico predictor passthrough
    predictors = {}
    for pred in raw.get("in_silico_predictors") or []:
        predictors[pred["id"]] = pred["value"]

    return {
        "variant_id": raw["variantId"],
        "chrom": raw["chrom"],
        "pos": raw["pos"],
        "ref": raw["ref"],
        "alt": raw["alt"],
        "hgvsc": raw.get("hgvsc"),
        "hgvsp": raw.get("hgvsp"),
        "consequence": raw.get("consequence"),
        "ac": ac_total,
        "an": an_total,
        "af": af_total,
        "ac_hom": ac_hom,
        "ancestry_afs": ancestry_afs,
        "clinvar_sig": clinvar_sig,
        "clinvar_review": clinvar_review,
        "clinvar_conditions": clinvar_conditions,
        "in_silico_predictors": predictors,
    }


def compute_frequency_feature(af: Optional[float]) -> float:
    """
    Map gnomAD allele frequency to frequency-based pathogenicity feature f1.

    Score interpretation:
      AF = 0 (absent)     -> 1.00 (maximum prior for pathogenicity)
      AF = 0.0001         -> 0.80
      AF = 0.001          -> 0.30
      AF >= 0.01          -> 0.00 (common variant, effectively benign)

    Uses log-linear interpolation between anchor points.

    Parameters
    ----------
    af : float or None
        gnomAD allele frequency. None is treated as AF=0 (absent from database).

    Returns
    -------
    float
        f1 score in [0, 1].
    """
    if af is None or af == 0:
        return 1.0
    if af >= 0.01:
        return 0.0

    # Log-linear interpolation: anchor points in log10(AF) space
    # (log10(0.0001), 0.80) and (log10(0.001), 0.30)
    log_af = math.log10(af)
    log_very_rare = math.log10(0.0001)   # -4
    log_rare = math.log10(0.001)          # -3
    log_common = math.log10(0.01)         # -2

    if log_af <= log_very_rare:
        # Linear from 0 AF (1.0) to AF=0.0001 (0.80)
        # Treat log_af = -inf as score = 1.0
        t = (log_af - log_very_rare) / (log_very_rare - (-10))  # normalize
        return max(0.80, min(1.0, 0.80 + 0.20 * (-t)))
    elif log_af <= log_rare:
        # Between 0.0001 and 0.001: linear from 0.80 to 0.30
        t = (log_af - log_very_rare) / (log_rare - log_very_rare)
        return 0.80 - t * 0.50
    else:
        # Between 0.001 and 0.01: linear from 0.30 to 0.00
        t = (log_af - log_rare) / (log_common - log_rare)
        return max(0.0, 0.30 - t * 0.30)


def parse_gnomad_vcf(vcf_path: str, af_threshold: float = 0.01) -> list[dict]:
    """
    Parse a gnomAD VCF file for CACNA1A missense variants.

    Filters to:
      - CACNA1A region (chr19:13,206,000-13,512,000)
      - Missense consequence only (CSQ field from VEP)
      - AF < af_threshold (rare variants)

    Parameters
    ----------
    vcf_path : str
        Path to gnomAD VCF (can be bgzipped .vcf.gz).
    af_threshold : float
        Maximum allele frequency to retain. Default 0.01 (1%).

    Returns
    -------
    list of dict
        List of variant records in ChanVar internal format.

    Notes
    -----
    Requires pysam. Install with: pip install pysam
    Large gnomAD VCFs are indexed with tabix; pysam uses the index for
    region-based queries without loading the full file.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for VCF parsing. Install with: pip install pysam"
        )

    variants = []
    region = f"chr19:{CACNA1A_START}-{CACNA1A_END}"

    with pysam.VariantFile(vcf_path) as vcf:
        for rec in vcf.fetch(region=region):
            # Extract AF from INFO field
            af = rec.info.get("AF")
            if isinstance(af, tuple):
                af = af[0]
            if af is None or af >= af_threshold:
                continue

            # Check for missense via VEP CSQ annotation
            csq_list = rec.info.get("vep") or rec.info.get("CSQ") or []
            is_missense = any("missense_variant" in str(csq) for csq in csq_list)
            if not is_missense:
                continue

            variants.append({
                "variant_id": f"{rec.chrom}-{rec.pos}-{rec.ref}-{rec.alts[0]}",
                "chrom": rec.chrom.replace("chr", ""),
                "pos": rec.pos,
                "ref": rec.ref,
                "alt": rec.alts[0],
                "af": af,
                "ac": rec.info.get("AC", (0,))[0] if isinstance(rec.info.get("AC"), tuple) else rec.info.get("AC", 0),
                "an": rec.info.get("AN", 0),
                "ac_hom": rec.info.get("nhomalt", 0),
            })

    logger.info("Parsed %d rare CACNA1A missense variants from VCF", len(variants))
    return variants
