from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from rdkit.Geometry import Point3D
import numpy as np

BOHR_TO_ANG = 0.52917721092


def replace_resnames_by_chain(
    pdb_file: str,
    chain_resname_map: Dict[str, str],
    output_file: Optional[str] = None,
) -> Path:
    """
    Replace residue names in a PDB based on the *chain* (branch) letter.

    Assumes input like:
        HETATM 1  C1  UNK A   1 ...

    where:
        - resname field (cols 18-21, 0-based 17-20) is 'UNK' (or some old name)
        - chain ID (col 22, 0-based 21) encodes the molecule type: 'A', 'B', 'C', ...

    For each ATOM/HETATM line:
        - Look up the chain ID in chain_resname_map
        - If found, set the resname field to the mapped value
        - Blank out the chain column (no more branches)

    Example:
        chain_resname_map = {'A': 'TRIO', 'B': 'TRIP'}
    """
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    out_path = Path(output_file) if output_file is not None else pdb_path

    lines = pdb_path.read_text().splitlines(keepends=True)
    modified_lines = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            # Need at least up to chain ID column
            if len(line) < 22:
                raise ValueError(f"ATOM/HETATM line too short or malformed:\n{line!r}")

            # Chain ID lives at column 22 (0-based index 21)
            chain = line[21]

            if chain in chain_resname_map:
                new_resname = chain_resname_map[chain]

                # Resname field: columns 18–21 (0-based 17–20), 4 characters
                # Right-align, truncate to 4 chars if longer
                new_resname_field = f"{new_resname:>4}"[:4]

                # Rebuild line:
                line = (
                    line[:17]  # up to (but not including) resname
                    + new_resname_field  # new resname (4 chars)
                    + " "  # blank chain ID
                    + line[22:]  # rest of the line
                )

        modified_lines.append(line)

    out_path.write_text("".join(modified_lines))
    return out_path


def _read_fchk_coords(fchk_path, labels=None, assume_bohr=True):
    """
    Parse a Gaussian .fchk file and return a list of (x,y,z) coordinates.

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file.
    labels : list of str, optional
        Possible labels for the coordinate block in the fchk.
        Common ones include "Current cartesian coordinates".
    assume_bohr : bool
        If True, converts from Bohr to Ångström.

    Returns
    -------
    coords : list[tuple[float,float,float]]
        List of (x,y,z) in Ångström or raw units.
    """
    if labels is None:
        labels = [
            "Current cartesian coordinates",
            "Cartesian Coordinates",
        ]

    with open(fchk_path, "r") as f:
        lines = f.read().splitlines()

    n_atoms = None
    start_idx = None

    # Find the coordinate block
    for i, line in enumerate(lines):
        for label in labels:
            if line.startswith(label):
                # fchk line looks like:
                # "Current cartesian coordinates
                # We extract N after "N="
                if "N=" not in line:
                    continue
                try:
                    n_atoms = int(line.split("N=")[1].strip())
                except ValueError:
                    continue
                start_idx = i + 1
                break
        if start_idx is not None:
            break

    if start_idx is None or n_atoms is None:
        raise ValueError(
            "Could not find coordinate block in fchk. "
            "Check the label names or file contents."
        )

    # Read 3*N floating-point numbers after that line
    n_vals = 3 * n_atoms
    vals = []
    idx = start_idx

    while len(vals) < n_vals and idx < len(lines):
        parts = lines[idx].split()
        vals.extend(float(x) for x in parts)
        idx += 1

    if len(vals) < n_vals:
        raise ValueError(f"Expected {n_vals} coordinate values, found {len(vals)}.")

    # Build list of (x, y, z); convert Bohr -> angstroms if requested
    factor = BOHR_TO_ANG if assume_bohr else 1.0
    coords = []
    for i in range(n_atoms):
        x, y, z = vals[3 * i : 3 * i + 3]
        coords.append((x * factor, y * factor, z * factor))

    return coords


def update_mol_coords_from_fchk(mol, fchk_path, conf_id=-1, assume_bohr=True):
    """
    Update the coordinates of an RDKit Mol using geometry from a Gaussian .fchk.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule whose coordinates will be updated (in-place).
    fchk_path : str
        Path to the .fchk file with the new geometry.
    conf_id : int, optional
        Conformer ID to update (default: -1, i.e. the default conformer).
    assume_bohr : bool, optional
        If True, assumes coordinates in fchk are in Bohr and converts to Å.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The same mol object, with updated coordinates.
    """
    coords = _read_fchk_coords(fchk_path, assume_bohr=assume_bohr)

    if mol.GetNumAtoms() != len(coords):
        raise ValueError(
            f"Atom count mismatch: Mol has {mol.GetNumAtoms()} atoms, "
            f"fchk has {len(coords)} coordinates."
        )

    conf = mol.GetConformer(conf_id)

    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol



def add_rxn(
    react_stoic,
    prod_stoic,
    rxn_names,
    ks,
    species_idx,
    reactants,
    products,
    k,
    name,
):
    """Build one reaction column using full-length vectors"""
    ns = len(species_idx)
    # Each reaction is a column vector (species are rows)
    r_col = np.zeros((ns, 1), dtype=float)
    p_col = np.zeros((ns, 1), dtype=float)

    for nm in reactants:
        r_col[species_idx[nm], 0] += 1.0
    for nm in products:
        p_col[species_idx[nm], 0] += 1.0

    # Append as new columns
    if len(react_stoic) == 0:
        react_stoic.append(r_col)
        prod_stoic.append(p_col)
    else:
        react_stoic.append(r_col)
        prod_stoic.append(p_col)

    ks.append(float(k))
    rxn_names.append(name)