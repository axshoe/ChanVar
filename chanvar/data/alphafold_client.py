"""
chanvar/data/alphafold_client.py
---------------------------------
AlphaFold2 structure download and PDB parsing for CACNA1A.

AlphaFold2 (Jumper et al. 2021, Nature) has predicted structures for essentially
all human proteins. CACNA1A (UniProt O00555) is available as a full-length
predicted structure with per-residue pLDDT confidence scores.

Download URL: https://alphafold.ebi.ac.uk/files/AF-O00555-F1-model_v4.pdb
Supplementary: For cryo-EM Cav2.1 structure see PDB 7MIY (Zhao et al. 2019).

pLDDT interpretation (from AlphaFold2 paper):
  pLDDT > 90: very high confidence (backbone accurate to ~0.5 Å RMSD)
  pLDDT 70–90: high confidence
  pLDDT 50–70: low confidence (may be disordered)
  pLDDT < 50: very low confidence (intrinsically disordered regions)

Transmembrane helix note: AlphaFold2 predicts transmembrane helices as helices
but does not model membrane embedding. For S1-S6 domain geometry, the cryo-EM
structure (PDB 7MIY) is preferred. Both are supported here.

References
----------
Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold.
    Nature, 596, 583–589.
Zhao, Y. et al. (2019). Cryo-EM structures of apo and antagonist-bound human
    Cav2.2. Nature, 576, 492–497.  [Note: 7MIY is Cav2.1 complex]
"""

import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F{fragment}-model_v4.pdb"
CACNA1A_UNIPROT = "O00555"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Residue ranges for CACNA1A functional regions (approximate, UniProt O00555)
DOMAIN_BOUNDARIES = {
    # Four repeat domains (I-IV), each with S1-S6 TM helices
    "domain_I": (116, 440),
    "domain_II": (541, 830),
    "domain_III": (1003, 1310),
    "domain_IV": (1339, 1660),
    # Voltage sensor S4 segments (arginine-rich)
    "S4_I": (181, 226),
    "S4_II": (752, 771),
    "S4_III": (1197, 1216),
    "S4_IV": (1554, 1573),
    # Pore-lining S5-S6 segments
    "pore_I": (290, 330),
    "pore_II": (636, 720),
    "pore_III": (1330, 1380),
    "pore_IV": (1640, 1680),
    # Selectivity filter EEEE locus (approximate positions)
    "selectivity_filter": [(293, 295), (636, 638), (1349, 1351), (1663, 1665)],
    # III-IV linker (inactivation gate)
    "IIIIV_linker": (1310, 1339),
    # C-terminal tail (SNARE interaction domain)
    "C_terminal": (1800, 2505),
    # N-terminal domain
    "N_terminal": (1, 116),
}


def download_alphafold_structure(
    uniprot_id: str = CACNA1A_UNIPROT,
    output_dir: str = "data/structures",
    fragment: int = 1,
) -> Optional[str]:
    """
    Download AlphaFold2 predicted structure for a UniProt entry.

    For very large proteins (>1400 aa), AlphaFold2 may split the structure
    into multiple fragments. CACNA1A (2505 aa) is provided as a single fragment
    in the v4 database.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession. Default 'O00555' (CACNA1A_HUMAN).
    output_dir : str
        Directory to save PDB file.
    fragment : int
        Fragment number (1-indexed). Usually 1 for most proteins.

    Returns
    -------
    str or None
        Path to downloaded PDB file, or None if download failed.
    """
    url = ALPHAFOLD_URL.format(uniprot_id=uniprot_id, fragment=fragment)
    output_path = Path(output_dir) / f"AF-{uniprot_id}-F{fragment}-model_v4.pdb"

    if output_path.exists():
        logger.info("AlphaFold2 structure already cached at %s", output_path)
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Downloading AlphaFold2 structure from %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info("Saved structure to %s (%d bytes)", output_path, len(resp.content))
        return str(output_path)
    except requests.RequestException as exc:
        logger.error("Failed to download AlphaFold2 structure: %s", exc)
        return None


def download_cryo_em_structure(pdb_id: str = "7MIY", output_dir: str = "data/structures") -> Optional[str]:
    """
    Download experimental cryo-EM structure from RCSB PDB.

    PDB 7MIY: Cav2.1/alpha2delta-1/beta2 complex at 3.7Å resolution.
    Zhao et al. 2019. The transmembrane domain geometry is more reliable
    than AlphaFold2 for S1-S6 segments.

    Parameters
    ----------
    pdb_id : str
        RCSB PDB identifier. Default '7MIY' (Cav2.1 complex).
    output_dir : str
        Directory to save PDB file.

    Returns
    -------
    str or None
        Path to downloaded PDB file.
    """
    url = RCSB_URL.format(pdb_id=pdb_id)
    output_path = Path(output_dir) / f"{pdb_id}.pdb"

    if output_path.exists():
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info("Downloaded cryo-EM structure %s to %s", pdb_id, output_path)
        return str(output_path)
    except requests.RequestException as exc:
        logger.error("Failed to download PDB %s: %s", pdb_id, exc)
        return None


def parse_pdb_structure(pdb_path: str):
    """
    Parse a PDB file using BioPython.

    Returns
    -------
    Bio.PDB.Structure.Structure
        Parsed structure object for downstream RMSD and coordinate operations.

    Raises
    ------
    ImportError
        If BioPython is not installed.
    FileNotFoundError
        If pdb_path does not exist.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError(
            "BioPython is required for PDB parsing. Install with: pip install biopython"
        )

    if not Path(pdb_path).exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    logger.info("Parsed PDB structure from %s", pdb_path)
    return structure


def get_residue_plddt(structure, residue_number: int, chain_id: str = "A") -> Optional[float]:
    """
    Extract pLDDT confidence score for a specific residue.

    In AlphaFold2 PDB output, pLDDT is stored in the B-factor column.
    Values range 0-100; >70 is considered reliable.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Parsed AlphaFold2 PDB structure.
    residue_number : int
        Residue number (1-indexed, matching UniProt sequence numbering).
    chain_id : str
        Chain identifier. AlphaFold2 single-chain structures use 'A'.

    Returns
    -------
    float or None
        pLDDT score for the residue, or None if residue not found.
    """
    try:
        chain = structure[0][chain_id]
        residue = chain[(" ", residue_number, " ")]
        # Average B-factor (pLDDT) across all atoms in residue
        bfactors = [atom.get_bfactor() for atom in residue.get_atoms()]
        return sum(bfactors) / len(bfactors) if bfactors else None
    except KeyError:
        logger.debug("Residue %d not found in structure", residue_number)
        return None


def get_local_coordinates(
    structure, residue_number: int, window: int = 10, chain_id: str = "A"
) -> Optional[list]:
    """
    Extract Cα coordinates for a local window around a residue.

    Used for computing local RMSD between wildtype and mutant structures.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Parsed PDB structure.
    residue_number : int
        Central residue number.
    window : int
        Half-window size (residues on each side). Default 10.
    chain_id : str
        Chain identifier.

    Returns
    -------
    list of Bio.PDB.Atom.Atom or None
        List of Cα atoms in the window, or None if chain not found.
    """
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return None

    atoms = []
    for res_num in range(residue_number - window, residue_number + window + 1):
        try:
            residue = chain[(" ", res_num, " ")]
            if "CA" in residue:
                atoms.append(residue["CA"])
        except KeyError:
            continue
    return atoms if atoms else None


def get_domain_for_residue(residue_number: int) -> str:
    """
    Map a CACNA1A residue position to its functional domain.

    Parameters
    ----------
    residue_number : int
        Amino acid position (1-indexed, UniProt O00555 numbering).

    Returns
    -------
    str
        Functional domain label. One of:
        'voltage_sensor', 'pore_lining', 'selectivity_filter',
        'IIIIV_linker', 'C_terminal', 'N_terminal', 'interdomain_linker'
    """
    # Check selectivity filter (highest priority, overlaps pore region)
    for sf_range in DOMAIN_BOUNDARIES["selectivity_filter"]:
        if sf_range[0] <= residue_number <= sf_range[1]:
            return "selectivity_filter"

    # Voltage sensor S4 segments
    for s4_key in ["S4_I", "S4_II", "S4_III", "S4_IV"]:
        start, end = DOMAIN_BOUNDARIES[s4_key]
        if start <= residue_number <= end:
            return "voltage_sensor"

    # Pore lining S5-S6
    for pore_key in ["pore_I", "pore_II", "pore_III", "pore_IV"]:
        start, end = DOMAIN_BOUNDARIES[pore_key]
        if start <= residue_number <= end:
            return "pore_lining"

    # III-IV linker
    start, end = DOMAIN_BOUNDARIES["IIIIV_linker"]
    if start <= residue_number <= end:
        return "IIIIV_linker"

    # Within a domain but not voltage sensor or pore (inter-S-segment linkers)
    for dom_key in ["domain_I", "domain_II", "domain_III", "domain_IV"]:
        start, end = DOMAIN_BOUNDARIES[dom_key]
        if start <= residue_number <= end:
            return "interdomain_linker"

    # Termini
    start, end = DOMAIN_BOUNDARIES["N_terminal"]
    if start <= residue_number <= end:
        return "N_terminal"

    start, end = DOMAIN_BOUNDARIES["C_terminal"]
    if start <= residue_number <= end:
        return "C_terminal"

    return "interdomain_linker"


# Domain-specific weight for pathogenicity scoring (f5 in CPS)
DOMAIN_PATHOGENICITY_WEIGHTS = {
    "voltage_sensor": 0.92,
    "selectivity_filter": 0.92,
    "pore_lining": 1.00,
    "interdomain_linker": 0.57,
    "N_terminal": 0.73,
    "C_terminal": 0.30,
    "IIIIV_linker": 0.57,    # same as interdomain_linker
}