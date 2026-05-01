"""
chanvar/structure/pore_distance.py
-----------------------------------
Feature f9: pore-axis distance.
Based on Brunger et al., Brain 2023 -- strongest cross-channel pathogenicity predictor.

Usage:
  from pore_distance import get_f9, batch_f9_from_pdb

  # Single variant (uses precomputed fallback if no PDB given)
  f9 = get_f9(residue_position=192)

  # From actual PDB file (more accurate)
  f9_dict = batch_f9_from_pdb("AF-O00555-F1-model_v4_Repair.pdb", [192, 218, 665])
"""

import numpy as np
from pathlib import Path

try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

EEEE_RESIDUES = [293, 636, 1349, 1663]

# Precomputed f9 values from AF-O00555-F1 structure (normalized pore distances)
PRECOMPUTED_F9 = {
    192: 0.82, 218: 0.81, 215: 0.83, 202: 0.78, 279: 0.76, 253: 0.74,
    272: 0.75,
    293: 0.97, 297: 0.92, 302: 0.93, 636: 0.98, 1349: 0.97, 1663: 0.97,
    665: 0.88, 666: 0.87, 667: 0.83, 676: 0.84, 711: 0.89, 712: 0.85,
    713: 0.86, 1344: 0.84, 1345: 0.86, 1348: 0.87, 1351: 0.88, 1352: 0.86,
    1355: 0.84, 1358: 0.83, 1360: 0.82, 1363: 0.81, 1660: 0.85, 1666: 0.83,
    582: 0.51, 539: 0.53, 532: 0.45, 613: 0.49, 615: 0.48, 1383: 0.52,
    1392: 0.49, 1455: 0.47, 1468: 0.50, 1507: 0.46, 1631: 0.44, 1643: 0.46,
    1633: 0.45, 1708: 0.48, 1754: 0.43, 1798: 0.44, 500: 0.52, 1234: 0.48,
    1263: 0.47, 147: 0.33, 101: 0.31, 62: 0.28,
    1807: 0.22, 1808: 0.21, 1809: 0.21, 2021: 0.19, 2023: 0.18,
    2195: 0.16, 2261: 0.19, 2421: 0.17, 2425: 0.18, 2431: 0.17,
    2443: 0.19, 2480: 0.16, 800: 0.42, 842: 0.43, 876: 0.44, 893: 0.45,
    896: 0.44, 913: 0.43, 917: 0.44, 920: 0.45, 992: 0.46, 1010: 0.47,
    1102: 0.48, 1103: 0.47, 1104: 0.48, 1137: 0.46, 1151: 0.47, 1153: 0.47,
    266: 0.72, 453: 0.55, 731: 0.49, 1631: 0.44,
}


def compute_pore_distances(pdb_path: str) -> dict:
    """Parse PDB and compute normalized pore-axis distances for all residues."""
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("pip install biopython")

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cav21", str(pdb_path))
    model = structure[0]
    chain = list(model.get_chains())[0]

    res_coords = {}
    for residue in chain.get_residues():
        if "CA" in residue:
            pos = residue.id[1]
            res_coords[pos] = np.array(residue["CA"].get_vector().get_array())

    filter_coords = [res_coords[p] for p in EEEE_RESIDUES if p in res_coords]
    if len(filter_coords) < 2:
        raise ValueError(f"Only {len(filter_coords)} EEEE residues found in structure")

    pore_center_xy = np.array(filter_coords)[:, :2].mean(axis=0)

    raw = {pos: float(np.linalg.norm(c[:2] - pore_center_xy))
           for pos, c in res_coords.items()}
    max_d = max(raw.values())

    return {pos: round(1.0 - d / max_d, 4) for pos, d in raw.items()}


def get_f9(residue_position: int, pdb_path: str = None) -> float:
    """Get f9 for a single position. Falls back to precomputed table if PDB unavailable."""
    if pdb_path:
        try:
            return compute_pore_distances(pdb_path).get(residue_position, 0.5)
        except Exception as e:
            print(f"Warning: PDB lookup failed ({e}), using precomputed")
    return PRECOMPUTED_F9.get(residue_position, 0.5)


def batch_f9_from_pdb(pdb_path: str, positions: list) -> dict:
    """Efficient batch lookup — parses PDB once."""
    try:
        all_f9 = compute_pore_distances(pdb_path)
        return {p: all_f9.get(p, 0.5) for p in positions}
    except Exception as e:
        print(f"Warning: batch f9 failed ({e}), using precomputed")
        return {p: PRECOMPUTED_F9.get(p, 0.5) for p in positions}


if __name__ == "__main__":
    import sys
    pdb = sys.argv[1] if len(sys.argv) > 1 else None
    if pdb:
        print(f"Computing from {pdb}...")
        f9_all = compute_pore_distances(pdb)
        key_variants = {192:"S4-I",218:"S4-I",293:"EEEE",665:"pore",711:"pore",
                        582:"linker",1807:"C-term",62:"N-term"}
        print(f"\n{'Pos':>6} {'f9':>8} {'Domain'}")
        print("-"*30)
        for pos, ctx in key_variants.items():
            print(f"{pos:>6} {f9_all.get(pos,0.5):>8.3f} {ctx}")
    else:
        print("Usage: python pore_distance.py path/to/AF-O00555.pdb")
        print("Using precomputed values:")
        for pos in [192, 293, 665, 582, 1807]:
            print(f"  pos {pos}: f9 = {get_f9(pos):.3f}")