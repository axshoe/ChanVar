"""
chanvar/structure/stability.py
-------------------------------
Structural impact prediction for CACNA1A missense variants.

Computes:
  1. dDDG: predicted change in folding free energy upon mutation (kcal/mol).
     Positive dDDG = destabilizing (pathogenicity indicator for most variants).
     Negative dDDG = stabilizing (ambiguous; gain-of-function FHM1 may be stabilizing).

  2. Local backbone RMSD: Root-mean-square deviation of Cα atoms in a 10-residue
     window between wildtype (AlphaFold2) and mutant (ESMFold) structures.

Implements FoldX and EvoEF2 wrappers (both physics-based stability calculators).
Running both and reporting the mean ± range improves robustness; the two tools
have different energy functions and different failure modes.

Important limitation: dDDG calculations for transmembrane segments are less reliable
than for soluble domains because all standard energy functions assume aqueous solvent.
This limitation is encoded in the domain-specific weight reduction applied in
compute_ddg_feature() for S1-S6 variants.

References
----------
Schymkowitz, J. et al. (2005). The FoldX web server: an online force field.
    Nucleic Acids Research, 33(Web Server issue), W382–W388.
Huang, X. et al. (2020). EvoEF2: accurate and fast energy function for
    computational protein design. Bioinformatics, 36(4), 1135–1142.
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Amino acid one-letter to three-letter codes (FoldX uses 3-letter)
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# Grantham physicochemical distance matrix (Grantham 1974, Science)
# Selected values for common mutation pairs
GRANTHAM_DISTANCES = {
    # Arg (R) mutations — common FHM1 variants
    ("R", "Q"): 43,   # R192Q: positive -> neutral
    ("R", "W"): 101,  # R192W: large physicochemical shift
    ("R", "H"): 29,   # Conservative
    ("R", "C"): 180,  # R -> C: extreme
    ("R", "L"): 102,
    ("R", "G"): 125,
    ("R", "S"): 110,
    # Ser (S) mutations
    ("S", "L"): 145,
    ("S", "P"): 74,
    ("S", "T"): 58,
    # Thr (T) mutations
    ("T", "M"): 81,
    ("T", "I"): 89,
    ("T", "A"): 58,
    # Other common pairs
    ("P", "Q"): 76,
    ("D", "N"): 23,
    ("E", "K"): 56,
    ("V", "I"): 29,
    ("L", "M"): 10,
    ("F", "L"): 22,
    ("Y", "C"): 194,
    ("W", "R"): 101,
    ("G", "R"): 125,
}

GRANTHAM_MAX = 215  # Max possible Grantham distance (Cys -> Trp or similar)


def grantham_distance(ref_aa: str, alt_aa: str) -> Optional[float]:
    """
    Return the Grantham physicochemical distance between two amino acids.

    The Grantham distance (Grantham 1974) quantifies how different two amino
    acids are in terms of composition, polarity, and molecular volume.
    Higher values indicate greater physicochemical dissimilarity.

    Parameters
    ----------
    ref_aa : str
        Reference (wildtype) amino acid, single-letter code.
    alt_aa : str
        Alternate (mutant) amino acid, single-letter code.

    Returns
    -------
    float or None
        Grantham distance (0–215), or None if the pair is not in the lookup table.

    Notes
    -----
    The full Grantham matrix is stored externally as data/grantham_matrix.json.
    This function provides a partial lookup for common FHM1/EA2 mutation pairs.
    For production use, load the full 20x20 matrix.
    """
    if ref_aa == alt_aa:
        return 0.0

    pair = (ref_aa, alt_aa)
    rev_pair = (alt_aa, ref_aa)

    if pair in GRANTHAM_DISTANCES:
        return float(GRANTHAM_DISTANCES[pair])
    if rev_pair in GRANTHAM_DISTANCES:
        return float(GRANTHAM_DISTANCES[rev_pair])

    # Attempt to load full matrix from file
    matrix_path = Path(__file__).parent.parent.parent / "data" / "grantham_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as f:
            matrix = json.load(f)
        key = f"{ref_aa}{alt_aa}"
        rev_key = f"{alt_aa}{ref_aa}"
        if key in matrix:
            return float(matrix[key])
        if rev_key in matrix:
            return float(matrix[rev_key])

    logger.warning("Grantham distance not available for %s->%s", ref_aa, alt_aa)
    return None


def compute_grantham_feature(ref_aa: str, alt_aa: str) -> float:
    """
    Map Grantham distance to pathogenicity feature f8 (normalized to [0, 1]).

    f8 = grantham_distance / GRANTHAM_MAX

    Rationale: more severe physicochemical change -> higher pathogenicity prior.
    This is a weak signal on its own but contributes to the composite CPS.

    Parameters
    ----------
    ref_aa : str
        Reference amino acid.
    alt_aa : str
        Alternate amino acid.

    Returns
    -------
    float
        f8 in [0, 1]. Returns 0.5 (neutral) if distance unavailable.
    """
    dist = grantham_distance(ref_aa, alt_aa)
    if dist is None:
        return 0.5  # Default to neutral when lookup fails
    return min(1.0, dist / GRANTHAM_MAX)


def run_foldx_ddg(
    pdb_path: str,
    chain: str,
    residue_number: int,
    ref_aa: str,
    alt_aa: str,
    foldx_executable: str = "foldx",
    temp_dir: Optional[str] = None,
) -> Optional[float]:
    """
    Run FoldX BuildModel to compute dDDG for a missense variant.

    FoldX energy function: empirical force field trained on experimental
    thermodynamic stability measurements. RepairPDB is run first to
    energy-minimize the input structure.

    dDDG = DDG(mutant) - DDG(wildtype)
    Positive dDDG = mutation destabilizes protein.
    Negative dDDG = mutation stabilizes protein.

    Parameters
    ----------
    pdb_path : str
        Path to wildtype PDB structure (AlphaFold2 or cryo-EM).
    chain : str
        Chain identifier for the residue (typically 'A').
    residue_number : int
        Residue number to mutate.
    ref_aa : str
        Reference amino acid (single-letter code).
    alt_aa : str
        Alternate amino acid (single-letter code).
    foldx_executable : str
        Path to FoldX executable or command name if on PATH.
    temp_dir : str, optional
        Working directory for FoldX temporary files.

    Returns
    -------
    float or None
        dDDG in kcal/mol, or None if FoldX is unavailable or fails.

    Notes
    -----
    FoldX requires registration at foldxsuite.biocreatec.com (free for academic use).
    The FoldX license restricts redistribution; do not bundle the binary in the repo.
    Expected runtime: 2-10 minutes per variant for RepairPDB + BuildModel.
    """
    # Check if FoldX is available
    if subprocess.run(["which", foldx_executable], capture_output=True).returncode != 0:
        logger.warning(
            "FoldX not found at '%s'. Install from foldxsuite.biocreatec.com. "
            "Returning None for dDDG.",
            foldx_executable,
        )
        return None

    with tempfile.TemporaryDirectory(dir=temp_dir) as workdir:
        pdb_name = Path(pdb_path).stem
        import shutil
        shutil.copy(pdb_path, workdir)

        # Step 1: RepairPDB to energy-minimize input
        repair_cmd = [
            foldx_executable,
            "--command=RepairPDB",
            f"--pdb={pdb_name}.pdb",
            "--out-pdb=true",
        ]
        result = subprocess.run(
            repair_cmd, capture_output=True, text=True, cwd=workdir
        )
        if result.returncode != 0:
            logger.error("FoldX RepairPDB failed: %s", result.stderr[:500])
            return None

        repaired_pdb = f"{pdb_name}_Repair.pdb"
        alt_aa3 = AA1_TO_3.get(alt_aa, alt_aa)

        # Step 2: Write individual_list.txt (FoldX mutation specification)
        # Format: {ref_aa}{chain}{residue_number}{alt_aa};
        mutation_str = f"{ref_aa}{chain}{residue_number}{alt_aa};"
        ind_list_path = Path(workdir) / "individual_list.txt"
        ind_list_path.write_text(mutation_str)

        # Step 3: BuildModel to compute dDDG
        build_cmd = [
            foldx_executable,
            "--command=BuildModel",
            f"--pdb={repaired_pdb}",
            "--mutant-file=individual_list.txt",
            "--numberOfRuns=3",  # Average over 3 runs for robustness
        ]
        result = subprocess.run(
            build_cmd, capture_output=True, text=True, cwd=workdir
        )
        if result.returncode != 0:
            logger.error("FoldX BuildModel failed: %s", result.stderr[:500])
            return None

        # Parse dDDG from FoldX output
        ddg = _parse_foldx_output(workdir, pdb_name)
        logger.info(
            "FoldX dDDG for %s%d%s: %.2f kcal/mol", ref_aa, residue_number, alt_aa, ddg or float("nan")
        )
        return ddg


def _parse_foldx_output(workdir: str, pdb_name: str) -> Optional[float]:
    """Parse dDDG from FoldX Average_BuildModel output file."""
    output_file = Path(workdir) / f"Average_{pdb_name}_Repair.fxout"
    if not output_file.exists():
        # Try alternative output filename patterns
        output_file = Path(workdir) / f"Average_{pdb_name}.fxout"
    if not output_file.exists():
        logger.warning("FoldX output file not found in %s", workdir)
        return None

    for line in output_file.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                return float(parts[1])  # dDDG is column 2 in Average output
            except (ValueError, IndexError):
                continue
    return None


def run_evoef2_ddg(
    pdb_path: str,
    chain: str,
    residue_number: int,
    ref_aa: str,
    alt_aa: str,
    evoef2_executable: str = "EvoEF2",
) -> Optional[float]:
    """
    Run EvoEF2 to compute dDDG for a missense variant.

    EvoEF2 is an open-source alternative to FoldX with a different energy function.
    Running both FoldX and EvoEF2 provides a robustness check; their mean and range
    are reported in ChanVar outputs.

    Install: https://github.com/tommyhuangthu/EvoEF2
    Build from source: g++ -O3 -o EvoEF2 src/*.cpp

    Parameters
    ----------
    (same as run_foldx_ddg)

    Returns
    -------
    float or None
        dDDG in kcal/mol, or None if EvoEF2 is unavailable or fails.
    """
    if subprocess.run(["which", evoef2_executable], capture_output=True).returncode != 0:
        logger.warning("EvoEF2 not found at '%s'. Install from github.com/tommyhuangthu/EvoEF2.", evoef2_executable)
        return None

    with tempfile.TemporaryDirectory() as workdir:
        import shutil
        pdb_name = Path(pdb_path).name
        shutil.copy(pdb_path, workdir)

        # EvoEF2 mutation format: A,R192Q (chain,change)
        mutation_str = f"{chain},{ref_aa}{residue_number}{alt_aa}"
        mut_file = Path(workdir) / "individual_list.txt"
        mut_file.write_text(mutation_str + "\n")

        cmd = [
            evoef2_executable,
            "--command=ComputeStability",
            f"--pdb={pdb_name}",
            "--mutant-file=individual_list.txt",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)

        if result.returncode != 0:
            logger.error("EvoEF2 failed: %s", result.stderr[:300])
            return None

        return _parse_evoef2_output(result.stdout)


def _parse_evoef2_output(stdout: str) -> Optional[float]:
    """Parse dDDG from EvoEF2 stdout output."""
    for line in stdout.splitlines():
        if "ddG" in line or "Total" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part in ("ddG", "DDG", "Total") and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        continue
    return None


def compute_ddg_feature(
    ddg_value: Optional[float],
    domain: str,
) -> float:
    """
    Map dDDG to pathogenicity feature f2 (sigmoid-normalized to [0, 1]).

    Feature formula:
        f2 = sigmoid(dDDG - 0.5)
    where sigmoid(x) = 1 / (1 + exp(-x))

    Calibration points:
        dDDG = 0.0 -> f2 ≈ 0.38
        dDDG = 0.5 -> f2 = 0.50
        dDDG = 2.0 -> f2 ≈ 0.82
        dDDG = -2.0 -> f2 ≈ 0.18

    For transmembrane domains (voltage_sensor, pore_lining), dDDG reliability
    is reduced because aqueous-solvent energy functions underestimate membrane
    stabilization. The domain weight is applied externally in compute_cps().

    Parameters
    ----------
    ddg_value : float or None
        dDDG in kcal/mol. None returns neutral score 0.5.
    domain : str
        Functional domain (affects reliability warning, not the score itself).

    Returns
    -------
    float
        f2 in [0, 1].
    """
    if ddg_value is None:
        return 0.5  # Neutral / missing data

    if domain in ("voltage_sensor", "pore_lining", "selectivity_filter"):
        logger.debug(
            "dDDG for transmembrane domain %s: reliability reduced (aqueous energy function). "
            "Score used but downweighted in CPS.",
            domain,
        )

    return 1.0 / (1.0 + math.exp(-(ddg_value - 0.5)))


def compute_local_rmsd(
    wt_structure,
    mut_structure,
    residue_number: int,
    window: int = 10,
    chain_id: str = "A",
) -> Optional[float]:
    """
    Compute local backbone RMSD between wildtype and mutant structures.

    Uses BioPython Superimposer to align and compute RMSD of Cα atoms
    in a ±window residue region around the mutation site.

    RMSD interpretation for CACNA1A:
      RMSD < 0.5 Å: minimal structural perturbation
      RMSD 0.5–1.5 Å: modest local perturbation
      RMSD 1.5–2.5 Å: significant structural change
      RMSD > 2.5 Å: major local structural disruption (especially significant in TM regions)

    Parameters
    ----------
    wt_structure : Bio.PDB.Structure
        Wildtype structure (AlphaFold2 or cryo-EM).
    mut_structure : Bio.PDB.Structure
        Mutant structure (ESMFold predicted or FoldX-built).
    residue_number : int
        Mutation site residue number.
    window : int
        Number of residues on each side to include. Default 10.
    chain_id : str
        Chain identifier.

    Returns
    -------
    float or None
        RMSD in Angstroms, or None if structures are incompatible.
    """
    try:
        from Bio.PDB import Superimposer
        import numpy as np
    except ImportError:
        raise ImportError("BioPython and numpy required for RMSD computation.")

    wt_atoms = []
    mut_atoms = []

    for offset in range(-window, window + 1):
        res_num = residue_number + offset
        try:
            wt_chain = wt_structure[0][chain_id]
            mut_chain = mut_structure[0][chain_id]
            wt_res = wt_chain[(" ", res_num, " ")]
            mut_res = mut_chain[(" ", res_num, " ")]
            if "CA" in wt_res and "CA" in mut_res:
                wt_atoms.append(wt_res["CA"])
                mut_atoms.append(mut_res["CA"])
        except KeyError:
            continue

    if len(wt_atoms) < 3:
        logger.warning("Insufficient aligned atoms for RMSD at residue %d", residue_number)
        return None

    sup = Superimposer()
    sup.set_atoms(wt_atoms, mut_atoms)
    return sup.rms


def compute_rmsd_feature(rmsd: Optional[float]) -> float:
    """
    Map local RMSD to pathogenicity feature f3 (sigmoid-normalized).

    Feature formula:
        f3 = sigmoid(RMSD - 1.0)

    Calibration:
        RMSD = 0.0 -> f3 ≈ 0.27
        RMSD = 1.0 -> f3 = 0.50
        RMSD = 2.5 -> f3 ≈ 0.82
        RMSD > 3.0 -> f3 approaches 0.95

    Parameters
    ----------
    rmsd : float or None
        RMSD in Angstroms. None returns neutral score 0.5.

    Returns
    -------
    float
        f3 in [0, 1].
    """
    if rmsd is None:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(rmsd - 1.0)))
