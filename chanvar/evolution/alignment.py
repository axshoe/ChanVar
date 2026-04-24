"""
chanvar/evolution/alignment.py
-------------------------------
Multiple sequence alignment for CACNA1A ortholog conservation analysis.

Builds vertebrate alignment of CACNA1A protein sequences to estimate
per-position evolutionary rates. The alignment is used as input to
the rate4site implementation in conservation.py.

Alignment species set (20–30 vertebrate species spanning ~500 Myr):
  Mammals: human, chimp, macaque, mouse, rat, guinea pig, dog, cat, horse, cow
  Birds: chicken, zebrafinch
  Reptiles: anole lizard
  Amphibians: Xenopus (Western clawed frog)
  Fish: zebrafish, medaka, stickleback, coelacanth (lobe-finned)
  Cartilaginous: elephant shark (Callorhinchus milii)

Software dependencies:
  MUSCLE v5 (fast multiple sequence alignment): https://github.com/rcedgar/muscle
  BioPython: pip install biopython (for FASTA IO and alignment parsing)

References
----------
Edgar, R.C. (2022). MUSCLE v5 enables improved estimates of phylogenetic tree
    confidence by ensemble bootstrapping. bioRxiv.
Katoh, K. & Standley, D.M. (2013). MAFFT multiple sequence alignment software
    version 7: improvements in performance and usability.
    Molecular Biology and Evolution, 30(4), 772–780.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# NCBI HomoloGene Group ID for CACNA1A (for automated ortholog fetching)
CACNA1A_HOMOLOGENE_ID = 37396

# Key vertebrate species for alignment (NCBI taxonomy IDs)
ALIGNMENT_SPECIES = {
    "homo_sapiens": 9606,
    "pan_troglodytes": 9598,
    "macaca_mulatta": 9544,
    "mus_musculus": 10090,
    "rattus_norvegicus": 10116,
    "canis_lupus_familiaris": 9615,
    "bos_taurus": 9913,
    "equus_caballus": 9796,
    "gallus_gallus": 9031,
    "danio_rerio": 7955,
    "xenopus_tropicalis": 8364,
    "latimeria_chalumnae": 7897,
    "callorhinchus_milii": 7868,
}


def fetch_orthologs_from_ensembl(gene_id: str = "ENSG00000141837") -> dict[str, str]:
    """
    Fetch CACNA1A ortholog protein sequences from Ensembl REST API.

    Uses the Ensembl ortholog endpoint to get protein sequences for all
    configured alignment species.

    Parameters
    ----------
    gene_id : str
        Ensembl gene ID for CACNA1A. Default 'ENSG00000141837'.

    Returns
    -------
    dict mapping species_name -> protein_sequence (str)

    Notes
    -----
    Requires internet access. Rate limit: 15 requests/second.
    Sequences are the canonical isoform translation from Ensembl.
    """
    import requests

    ENSEMBL_REST = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json"}

    # Get ortholog IDs
    url = f"{ENSEMBL_REST}/homology/id/{gene_id}"
    params = {"type": "orthologues", "content-type": "application/json", "format": "full"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Ensembl ortholog query failed: %s", exc)
        return {}

    sequences = {}
    homologies = data.get("data", [{}])[0].get("homologies", [])

    for hom in homologies:
        target = hom.get("target", {})
        species = target.get("species", "")
        protein_id = target.get("protein_id", "")

        if not protein_id:
            continue

        # Fetch protein sequence
        seq_url = f"{ENSEMBL_REST}/sequence/id/{protein_id}"
        seq_params = {"type": "protein", "content-type": "text/plain"}
        try:
            seq_resp = requests.get(seq_url, params=seq_params, timeout=30)
            if seq_resp.ok:
                sequences[species] = seq_resp.text.strip()
        except Exception:
            continue

    logger.info("Fetched %d ortholog sequences from Ensembl", len(sequences))
    return sequences


def write_fasta(sequences: dict[str, str], output_path: str) -> str:
    """
    Write sequences dict to FASTA format.

    Parameters
    ----------
    sequences : dict
        {sequence_id: sequence_string} pairs.
    output_path : str
        Output file path.

    Returns
    -------
    str
        Path to written FASTA file.
    """
    with open(output_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n")
            # Wrap at 60 characters per line
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")
    return output_path


def run_muscle_alignment(
    input_fasta: str,
    output_fasta: str,
    muscle_executable: str = "muscle",
    threads: int = 4,
) -> Optional[str]:
    """
    Run MUSCLE v5 multiple sequence alignment.

    MUSCLE v5 uses an iterative refinement algorithm (Super5 mode for large sets).
    Expected runtime for 20–30 CACNA1A sequences: 5–30 minutes depending on
    length and alignment quality parameters.

    Parameters
    ----------
    input_fasta : str
        Path to unaligned sequences in FASTA format.
    output_fasta : str
        Path for aligned output FASTA.
    muscle_executable : str
        Path to MUSCLE binary.
    threads : int
        Number of CPU threads for parallel alignment.

    Returns
    -------
    str or None
        Path to aligned FASTA, or None if MUSCLE failed.

    Notes
    -----
    MUSCLE v5 command syntax changed substantially from v3. This function
    uses the v5 syntax (-align input -output output).
    """
    if subprocess.run(["which", muscle_executable], capture_output=True).returncode != 0:
        logger.warning(
            "MUSCLE not found at '%s'. "
            "Install from https://github.com/rcedgar/muscle/releases. "
            "Pre-computed alignment available at data/cacna1a_alignment.fasta.",
            muscle_executable,
        )
        return None

    cmd = [
        muscle_executable,
        "-align", input_fasta,
        "-output", output_fasta,
        "-threads", str(threads),
    ]

    logger.info("Running MUSCLE alignment: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("MUSCLE failed: %s", result.stderr[:500])
        return None

    logger.info("MUSCLE alignment complete -> %s", output_fasta)
    return output_fasta


def run_mafft_alignment(
    input_fasta: str,
    output_fasta: str,
    mafft_executable: str = "mafft",
    threads: int = 4,
) -> Optional[str]:
    """
    Run MAFFT alignment as an alternative to MUSCLE.

    MAFFT L-INS-i mode is more accurate than MUSCLE for structurally
    complex proteins but slower. For CACNA1A (2505 aa, 20-30 sequences),
    FFT-NS-2 mode is used for speed.

    Parameters
    ----------
    input_fasta : str
        Unaligned sequences FASTA.
    output_fasta : str
        Output aligned FASTA path.
    mafft_executable : str
        Path to MAFFT binary.
    threads : int
        CPU threads.

    Returns
    -------
    str or None
        Path to aligned output, or None on failure.
    """
    cmd = [
        mafft_executable,
        "--thread", str(threads),
        "--auto",
        "--out", output_fasta,
        input_fasta,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("MAFFT failed: %s", result.stderr[:300])
        return None

    return output_fasta


def parse_aligned_fasta(fasta_path: str) -> dict[str, str]:
    """
    Parse aligned FASTA file into dict of {sequence_id: aligned_sequence}.

    Aligned sequences may contain gap characters ('-') and must be equal length.

    Parameters
    ----------
    fasta_path : str
        Path to aligned FASTA file.

    Returns
    -------
    dict
        {sequence_id: aligned_sequence_string}

    Raises
    ------
    ValueError
        If sequences have unequal lengths (invalid alignment).
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)

    if not sequences:
        return {}

    lengths = set(len(s) for s in sequences.values())
    if len(lengths) > 1:
        raise ValueError(
            f"Alignment has sequences of unequal length: {lengths}. "
            "File may be unaligned or truncated."
        )

    return sequences


def extract_alignment_column(
    alignment: dict[str, str], position: int, reference_species: str = "homo_sapiens"
) -> list[str]:
    """
    Extract the alignment column corresponding to a human residue position.

    Because the alignment has gap characters, the human amino acid at position N
    does not necessarily correspond to alignment column N. This function maps
    ungapped human positions to alignment columns.

    Parameters
    ----------
    alignment : dict
        {species: aligned_sequence} dict from parse_aligned_fasta.
    position : int
        Human (ungapped) residue position (1-indexed).
    reference_species : str
        Key for the reference sequence in alignment dict.

    Returns
    -------
    list of str
        One amino acid character per species at the alignment column
        corresponding to the human position. Gaps are included.
    """
    if reference_species not in alignment:
        # Try partial match
        matches = [k for k in alignment if reference_species in k]
        if not matches:
            logger.warning("Reference species '%s' not in alignment", reference_species)
            return []
        reference_species = matches[0]

    human_seq = alignment[reference_species]

    # Map ungapped position to alignment column
    ungapped_pos = 0
    target_col = None
    for col_idx, char in enumerate(human_seq):
        if char != "-":
            ungapped_pos += 1
            if ungapped_pos == position:
                target_col = col_idx
                break

    if target_col is None:
        logger.debug("Position %d beyond human sequence length", position)
        return []

    return [seq[target_col] for seq in alignment.values() if target_col < len(seq)]


def compute_all_position_conservation(
    alignment_path: str,
    reference_species: str = "homo_sapiens",
    output_tsv: Optional[str] = None,
) -> dict[int, dict]:
    """
    Compute conservation metrics for all CACNA1A positions.

    Runs Shannon entropy, Jensen-Shannon divergence, and simplified rate4site
    across all alignment columns. Optionally saves to TSV for caching.

    Parameters
    ----------
    alignment_path : str
        Path to aligned FASTA file.
    reference_species : str
        Key identifying the reference (human) sequence.
    output_tsv : str, optional
        If provided, save results to this TSV file.

    Returns
    -------
    dict mapping position (int) -> conservation metrics dict
    """
    from chanvar.evolution.conservation import (
        compute_shannon_entropy,
        estimate_rate4site,
        compute_jensenshannon_divergence,
    )

    alignment = parse_aligned_fasta(alignment_path)
    if not alignment:
        return {}

    # Determine human sequence length (ungapped)
    human_key = next((k for k in alignment if reference_species in k), None)
    if not human_key:
        logger.error("Reference species '%s' not found in alignment", reference_species)
        return {}

    human_seq = alignment[human_key]
    human_length = len(human_seq.replace("-", ""))

    results = {}
    for pos in range(1, human_length + 1):
        column = extract_alignment_column(alignment, pos, human_key)
        if not column:
            continue

        entropy = compute_shannon_entropy(column)
        rate = estimate_rate4site(column)
        js_div = compute_jensenshannon_divergence(column)

        # Dominant amino acid in alignment column (excluding gaps)
        filtered = [aa for aa in column if aa not in ("-", ".", "X")]
        if filtered:
            from collections import Counter
            dominant_aa = Counter(filtered).most_common(1)[0][0]
        else:
            dominant_aa = "?"

        results[pos] = {
            "rate4site": rate,
            "shannon_entropy": entropy,
            "js_divergence": js_div,
            "dominant_aa": dominant_aa,
            "n_species": len(filtered),
        }

    if output_tsv:
        _write_conservation_tsv(results, output_tsv)

    logger.info("Computed conservation for %d positions", len(results))
    return results


def _write_conservation_tsv(scores: dict, output_path: str) -> None:
    """Write per-position conservation scores to TSV file."""
    cols = ["rate4site", "shannon_entropy", "js_divergence", "dominant_aa", "n_species"]
    with open(output_path, "w") as f:
        f.write("position\t" + "\t".join(cols) + "\n")
        for pos in sorted(scores.keys()):
            row = scores[pos]
            f.write(
                f"{pos}\t"
                + "\t".join(str(row.get(c, "NA")) for c in cols)
                + "\n"
            )
    logger.info("Conservation scores written to %s", output_path)
