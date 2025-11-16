from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from pathlib import Path
import json
import shutil
import subprocess
import re
import sys
import os


# Mark which real OPLS types are alkene C / H
ALKENE_C_TYPES = {"opls_142"}  # should be extended if more alkene C types
ALKENE_H_TYPES = {"opls_144"}  # alkene hydrogens


def _filter_problem_dihedrals(
    diheds: list[tuple[int, int, int, int, int]],
    atom_records: list[AtomRecord],
) -> list[tuple[int, int, int, int, int]]:
    """
    Remove dihedrals that involve an alkene H at one end and an alkene C
    among the two central atoms. These have no RB defaults in OPLS-AA.
    """
    filtered: list[tuple[int, int, int, int, int]] = []

    for i, j, k, l, funct in diheds:
        ti = atom_records[i].type
        tj = atom_records[j].type
        tk = atom_records[k].type
        tl = atom_records[l].type

        # Condition: one end is alkene H, and one central atom is alkene C
        end_has_alkene_H = (ti in ALKENE_H_TYPES) or (tl in ALKENE_H_TYPES)
        center_has_alkene_C = (tj in ALKENE_C_TYPES) or (tk in ALKENE_C_TYPES)

        if end_has_alkene_H and center_has_alkene_C:
            # This is exactly the pattern that blows up:
            # e.g.  136-136-142-144  or  144-142-136-136
            continue

        filtered.append((i, j, k, l, funct))

    return filtered


@dataclass
class AtomRecord:
    idx: int
    name: str
    type: str
    resname: str
    resid: int
    charge: float
    mass: float


PERIODIC_MASS = {
    1: 1.008,  # H
    6: 12.011,  # C
    7: 14.007,  # N
    8: 15.999,  # O
    9: 18.998,  # F
    15: 30.974,  # P
    16: 32.06,  # S
    17: 35.45,  # Cl
    35: 79.904,  # Br
}


# SMARTS-based atom typing scaffold for glycerides (OPLS-AA)
class OPLSAtomTyper:
    """
    Rule-based SMARTS typing tailored for glycerides (tri-, di-, monoacylglycerols).
    This is a *starter scaffold*: replace the 'type' and 'charge' with authoritative values.

    Rules are evaluated in order. First match wins.
    Each rule tuple: (SMARTS, type_name, default_charge, name_hint)

    YOU SHOULD:
      - Replace 'type_name' (e.g., 'opls_EST_C') with real OPLS-AA types (e.g., 'opls_235') if applicable.
      - Replace charges with literature/forcefield values for your target subclass (TAG, DAG, MAG).
    """

    def __init__(self) -> None:
        self.rules: List[Tuple[Chem.Mol, str, float, str]] = []
        self._build_rules()

    def _add(self, smarts: str, type_name: str, charge: float, hint: str) -> None:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS: {smarts}")
        self.rules.append((patt, type_name, charge, hint))

    def _build_rules(self) -> None:

        # Ester Core
        self._add(
            "[C;X3;$([C](=O)O)]",
            "opls_EST_C",
            0.70,
            "Ester carbonyl C",
        )

        # Carbonyl O (double-bond O) in ester
        self._add(
            "[O;X1;$([O]=C)]",
            "opls_EST_O",
            -0.55,
            "Ester C=O oxygen",
        )

        # Alkoxy O (single-bond O) directly attached to ester carbonyl C
        self._add(
            "[O;X2;$([O]C(=O))]",
            "opls_ALKOXY_O",
            -0.45,
            "Ester alkoxy oxygen",
        )

        # Glycerol backbone rules (three sp3 carbons with hydroxyls/esters)
        # Central glycerol carbon (sn-2): secondary alcohol carbon
        self._add("[C;X4](O)(CO)C", "opls_GLY_C2", 0.10, "Glycerol C2 sp3")

        # Terminal glycerol carbons (sn-1, sn-3) directly attached to O
        self._add("[CH2;X4]CO", "opls_GLY_CH2", -0.02, "Glycerol CH2")

        # Glycerol oxygens (non-carbonyl) in ester linkages or alcohols
        self._add("[O;X2]C[CH2]", "opls_GLY_O", -0.45, "Glycerol O (linkage)")

        # saturated alkyl chain
        # terminal methyl
        self._add("[CH3;X4][CH2;X4]", "opls_ALK_CH3", -0.06, "terminal methyl")

        # methylene
        self._add("[CH2;X4][CH2;X4]", "opls_ALK_CH2", -0.02, "methylene")

        # R2C= : sp2 carbon with no H (fully substituted)
        self._add(
            "[C;X3H0]=[C;X3]",
            "opls_ALKENE_C_R2",
            -0.10,
            "alkene C (R2-C=)",
        )

        # RH-C= : sp2 carbon with one H and one substituent
        self._add(
            "[CH;X3]=[C;X3]",
            "opls_ALKENE_C_RH",
            -0.10,  # placeholder
            "alkene C (RH-C=)",
        )

        # H2-C= : sp2 carbon with two hydrogens (CH2=)
        self._add(
            "[CH2;X3]=[C;X3]",
            "opls_ALKENE_C_H2",
            -0.10,  # placeholder
            "alkene C (H2-C=)",
        )

        # Alkene H: hydrogen directly bound to an sp2 carbon in a C=C
        # (this will pick up H on either side of the double bond)
        self._add(
            "[H][C;X3]=[C;X3]",
            "opls_ALKENE_H",
            0.10,  # placeholder
            "alkene H (H-C=)",
        )
        self._add("[H]", "opls_HGEN", 0.0, "generic H")

    def assign(self, mol: Chem.Mol) -> List[Tuple[str, float, str]]:
        """
        Returns for each atom: (type_name, charge, name_hint)
        """
        out = [("opls_UNK", 0.0, "untyped") for _ in range(mol.GetNumAtoms())]
        for i, atom in enumerate(mol.GetAtoms()):
            # a_mol = Chem.MolFromSmarts(f"[#{atom.GetAtomicNum()}]")
            # Apply first-match rules
            for patt, tname, q, hint in self.rules:
                matches = mol.GetSubstructMatches(patt, uniquify=True)
                for match in matches:
                    if i in match:
                        out[i] = (tname, q, hint)
                        break
                if out[i][0] != "opls_UNK":
                    break
            # Fallback to element-based generic types if still unk
            if out[i][0] == "opls_UNK":
                z = atom.GetAtomicNum()
                if z == 6:
                    out[i] = ("opls_CGEN", 0.0, "generic C")
                elif z == 8:
                    out[i] = ("opls_OGEN", 0.0, "generic O")
                elif z == 1:
                    out[i] = ("opls_HGEN", 0.0, "generic H")
        return out


def _is_H(atom):
    return atom.GetAtomicNum() == 1


def _is_sp2_carbon(atom):
    return (
        atom.GetAtomicNum() == 6
        and atom.GetHybridization() == Chem.HybridizationType.SP2
    )


def _skip_dihedral(mol, i, j, k, l):
    ai = mol.GetAtomWithIdx(i)
    aj = mol.GetAtomWithIdx(j)
    ak = mol.GetAtomWithIdx(k)
    al = mol.GetAtomWithIdx(l)

    # Skip dihedrals with an H at one end and an sp2 carbon in the middle.
    if (_is_H(ai) or _is_H(al)) and (_is_sp2_carbon(aj) or _is_sp2_carbon(ak)):
        return True
    return False


def build_bonded_lists(mol: Chem.Mol):
    bonds = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bonds.append((i, j, 1))  # harmonic bond

    # Angles
    angles = []
    for j in range(mol.GetNumAtoms()):
        nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(j).GetNeighbors()]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i = nbrs[a]
                k = nbrs[b]
                angles.append((i, j, k, 1))

    # Dihedrals
    diheds = set()
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        jnbrs = [
            a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k
        ]
        knbrs = [
            a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j
        ]
        for i in jnbrs:
            for l in knbrs:
                if i == l:
                    continue
                if _skip_dihedral(mol, i, j, k, l):
                    continue
                diheds.add((i, j, k, l))

    diheds = [(i, j, k, l, 3) for (i, j, k, l) in sorted(diheds)]
    return bonds, angles, diheds


def _nm(x):  # ang to nm
    return x * 0.1


def write_gro(mol: Chem.Mol, coords: np.ndarray, fn: str, title: str = "glyceride"):
    n = mol.GetNumAtoms()
    lines = [title, str(n)]
    resid = 1
    resname = "GLYCR"
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        name = atom.GetSymbol()[:5]
        x, y, z = map(_nm, coords[i - 1])
        lines.append(f"{resid:5d}{resname:<5}{name:>5}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
    # simple box
    lines.append("  5.00000  5.00000  5.00000")
    Path(fn).write_text("\n".join(lines))


def write_itp(
    mol: Chem.Mol, types: List[Tuple[str, float, str]], fn: str, resname: str = "GLYCR"
):
    # atom names: element plus index
    atom_records: List[AtomRecord] = []
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        name = f"{sym}{i+1}"
        tname, default_q, _ = types[i]
        # Prefer per-atom _Charge property (e.g., CM5 * 1.20) if present
        if atom.HasProp("_Charge"):
            try:
                q = atom.GetDoubleProp("_Charge")
            except Exception:
                q = default_q
        else:
            q = default_q
        mass = PERIODIC_MASS.get(atom.GetAtomicNum(), 12.0)
        atom_records.append(
            AtomRecord(
                idx=i,
                name=name,
                type=tname,
                resname=resname,
                resid=1,
                charge=q,
                mass=mass,
            )
        )
    bonds, angles, diheds = build_bonded_lists(mol)

    lines: List[str] = []
    lines.append(f"[ moleculetype ]")
    lines.append(f"; name    nrexcl")
    lines.append(f"{resname}  3")
    lines.append("")
    lines.append("[ atoms ]")
    lines.append(";  nr  type   resnr  resid  atom  cgnr   charge   mass")
    for i, rec in enumerate(atom_records, start=1):
        lines.append(
            f"{i:5d} {rec.type:<10} {rec.resid:5d} {rec.resname:<6} {rec.name:<6} {i:5d} {rec.charge:8.4f} {rec.mass:8.3f}"
        )

    lines.append("")
    lines.append("[ bonds ]")
    lines.append(";  ai  aj  funct")
    for i, j, funct in bonds:
        lines.append(f"{i+1:5d}{j+1:5d}{funct:6d}")

    lines.append("")
    lines.append("[ angles ]")
    lines.append(";  ai  aj  ak  funct")
    for i, j, k, funct in angles:
        lines.append(f"{i+1:5d}{j+1:5d}{k+1:5d}{funct:6d}")

    lines.append("")
    lines.append("[ dihedrals ]")
    lines.append(";  ai  aj  ak  al  funct")
    for i, j, k, l, funct in diheds:
        lines.append(f"{i+1:5d}{j+1:5d}{k+1:5d}{l+1:5d}{funct:6d}")

    Path(fn).write_text("\n".join(lines))


def write_top(system_name: str, include_itp: str, fn: str, n_mol: int = 1):
    lines = []
    lines.append("; GROMACS topology generated by gromacs_topgen.py")
    lines.append("; Make sure oplsaa.ff is available in your GMX topology path")
    lines.append('#include "oplsaa.ff/forcefield.itp"')
    lines.append("")
    lines.append(f'#include "{include_itp}"')
    lines.append("")
    lines.append("[ system ]")
    lines.append(system_name)
    lines.append("")
    lines.append("[ molecules ]")
    lines.append(f"GLYCR  {n_mol}")
    Path(fn).write_text("\n".join(lines))


def rdkit_embed_minimize(mol: Chem.Mol, max_iters: int = 500) -> np.ndarray:
    """Return coordinates (ang). If mol already has a conformer, use it; otherwise embed+minimize."""
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=float,
        )
        return coords
    # else: create a fresh conformer on a copy with Hs
    m2 = Chem.AddHs(Chem.Mol(mol))
    AllChem.EmbedMolecule(m2, AllChem.ETKDGv3())
    try:
        AllChem.UFFOptimizeMolecule(m2, maxIters=max_iters)
    except Exception:
        pass
    conf = m2.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(m2.GetNumAtoms())], dtype=float
    )
    return coords


def build_top_for_glyceride_with_cm5(
    glx_mol: Chem.Mol, cm5_xyzq_path: str, prefix="glyceride"
):
    base = Chem.Mol(glx_mol)
    Chem.SanitizeMol(base)
    # ensure base has coordinates; embed if needed (used for NN matching)
    if base.GetNumConformers() == 0:
        _ = rdkit_embed_minimize(base)
    # attach CM5
    assign_cm5_to_mol_by_coords(base, cm5_xyzq_path, tol=0.3)
    # typing + files
    coords = rdkit_embed_minimize(base)  # or reuse existing conformer coords
    typer = OPLSAtomTyper()
    types = typer.assign(base)
    itp_fn = f"{prefix}.itp"
    gro_fn = f"{prefix}.gro"
    top_fn = f"{prefix}.top"
    write_itp(base, types, itp_fn)  # picks up _Charge
    write_gro(base, coords, gro_fn, title=prefix)
    write_top(prefix, itp_fn, top_fn)
    return itp_fn, top_fn, gro_fn, types


def dump_typing_report(
    types: List[Tuple[str, float, str]],
    mol: Chem.Mol,
    report_fn: str = "typing_report.json",
):
    rep = []
    for i, atom in enumerate(mol.GetAtoms()):
        t, q, hint = types[i]
        rep.append(
            {
                "index": i + 1,
                "element": atom.GetSymbol(),
                "type": t,
                "charge": q,
                "hint": hint,
            }
        )
    Path(report_fn).write_text(json.dumps(rep, indent=2))


def execute_multiwfn_cm5(
    fchk_path: str,
    input_file: str | None = None,
    exe: str | Path | None = None,
    stdout_path: str = "multiwfn.stdout",
):
    """
    Run Multiwfn in batch mode to print CM5 charges.

    Path handling:
      - If `exe` is a path (e.g. '~/bin/Multiwfn' or './Multiwfn'), it is
        expanded (including '~') and resolved to an absolute path.
      - If `exe` is a bare program name (e.g. 'Multiwfn'), we use it as-is and
        rely on the environment PATH.
      - If `exe` is None, we search PATH for 'Multiwfn' or 'multiwfn' with
        shutil.which().

    This version:
      - does NOT use check=True
      - always writes stdout to stdout_path
      - treats the infamous Fortran 'severe (59)' stdin EOF as a soft failure
        IF CM5 output is present in stdout.
    """
    fchk = Path(fchk_path).expanduser().resolve()
    if not fchk.exists():
        raise FileNotFoundError(f"fchk not found: {fchk}")

    # Resolve executable:
    #   - if exe is None, search PATH
    #   - if exe is a path and exists, use its absolute path
    #   - otherwise, treat exe as a bare command name and rely on PATH
    if exe is None:
        found = shutil.which("Multiwfn") or shutil.which("multiwfn")
        if found is None:
            raise RuntimeError(
                "Multiwfn not found on PATH. "
                "Either install it or pass `exe` as an explicit path/name."
            )
        exe_str = found  # already an absolute path from which()
    else:
        # Allow strings or Path objects, with '~' and relatives
        exe_path = Path(exe).expanduser()
        if exe_path.exists():
            exe_str = str(exe_path.resolve())
        else:
            # Not an existing path: assume it's a command name ('Multiwfn') and
            # let the shell / PATH resolve it.
            exe_str = str(exe)

    # If no script is given, auto-generate one; otherwise, use what the caller provided
    if input_file is None:
        script = Path("cm5.txt")
        # TODO: adjust these numbers to the menu sequence that works for your Multiwfn build
        script.write_text("7\n16\n1\ny\nq\n")
        input_file = str(script)

    # Make sure the input script is a proper (possibly absolute) path for the shell
    input_path = Path(input_file).expanduser().resolve()

    # Build shell command; quote paths in case of spaces
    cmd = f'"{exe_str}" "{fchk}" < "{input_path}"'

    # IMPORTANT: no check=True here; we'll inspect return code ourselves
    p = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr
    stdout_txt = stdout_bytes.decode(errors="ignore")
    stderr_txt = stderr_bytes.decode(errors="ignore")

    # Always write stdout so it can be parsed even if rc != 0
    Path(stdout_path).write_bytes(stdout_bytes)

    print("\n======= DEBUG: CM5 LINES FROM MULTIWFN =======")
    with open(stdout_path, "r") as f:
        for line in f:
            if "CM5" in line or "charge" in line or "Charge" in line:
                print(line.rstrip())
    print("===============================================\n")

    if p.returncode == 0:
        # clean exit
        return stdout_path

    # Special handling: Fortran "list-directed I/O syntax error" (code 59),
    # but CM5 output appears in stdout.
    if (
        p.returncode == 59
        and "forrtl: severe (59)" in stderr_txt
        and "CM5" in stdout_txt
    ):
        print(
            "Warning: Multiwfn returned code 59 (stdin EOF), "
            "but CM5 output appears present; proceeding anyway.",
            file=sys.stderr,
        )
        return stdout_path

    # Anything else is a real failure
    raise ValueError(
        f"Multiwfn failed with errorcode {p.returncode} and stderr:\n{stderr_txt}"
    )


def read_cm5_xyzq_table(path: str):
    """
    Parse a Multiwfn dump with lines: Element  x  y  z  charge
    Returns (elements: [str], coords: (N,3) float Å, charges: (N,) float).
    """
    import numpy as np

    elems, coords, qs = [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            toks = line.split()
            if len(toks) != 5:
                # skip non-data lines if any
                continue
            el, x, y, z, q = toks
            elems.append(el)
            coords.append([float(x), float(y), float(z)])
            qs.append(float(q))
    if not qs:
        raise RuntimeError(f"No data parsed from {path}")
    return np.array(elems), np.array(coords, float), np.array(qs, float)


def parse_cm5_from_multiwfn(
    stdout_path: str, natoms: Optional[int] = None, debug: bool = True
) -> List[float]:
    """
    Parse CM5 charges from a Multiwfn stdout file.

    Parameters
    ----------
    stdout_path : str
        Path to Multiwfn stdout.
    natoms : int, optional
        If provided, stop after reading this many charges and sanity-check.
    debug : bool
        If True, print a short debug summary of what was parsed.

    Returns
    -------
    charges : list[float]
        List of CM5 atomic charges.
    """
    charges: List[float] = []
    in_block = False

    # Example line:
    # " Atom:    1C   CM5 charge:   -0.048081  Hirshfeld charge:    0.029297"
    cm5_line_re = re.compile(
        r"^\s*Atom:\s*(\d+)([A-Za-z]+)\s+CM5 charge:\s*([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)"
    )

    with open(stdout_path, "r") as f:
        for line in f:
            # Detect the CM5 header
            if (not in_block) and ("CM5 charges" in line and "--------" in line):
                in_block = True
                if debug:
                    print(f"[CM5 DEBUG] Found CM5 header line:\n{line.rstrip()}")
                continue

            if not in_block:
                continue

            # Stop when we reach the summary line
            if "Summing up all CM5 charges" in line:
                if debug:
                    print(
                        "[CM5 DEBUG] Reached 'Summing up all CM5 charges' line. Stopping parse."
                    )
                break

            # Only parse lines that look like "Atom: ..."
            if not line.lstrip().startswith("Atom:"):
                # If we already collected some charges, and the line no longer
                # looks like an Atom line, we assume the Atom block ended.
                if charges:
                    if debug:
                        print("[CM5 DEBUG] Left Atom-block; stopping parse.")
                    break
                # If no charges yet, might still be some noise/header; skip.
                continue

            m = cm5_line_re.match(line)
            if not m:
                # Line starts with "Atom:" but doesn't match expected pattern
                if debug:
                    print(
                        f"[CM5 DEBUG] WARNING: 'Atom:' line did not match regex:\n{line.rstrip()}"
                    )
                continue

            idx_str, elem, q_str = m.groups()
            q = float(q_str)
            charges.append(q)

            if natoms is not None and len(charges) >= natoms:
                if debug:
                    print(
                        f"[CM5 DEBUG] Reached natoms={natoms}; " "stopping parse early."
                    )
                break

    if not charges:
        raise RuntimeError(f"Failed to find CM5 charges in {stdout_path}")

    if debug:
        print(f"[CM5 DEBUG] Parsed {len(charges)} CM5 charges from {stdout_path}")
        if len(charges) <= 10:
            print("[CM5 DEBUG] CM5 charges:", charges)
        else:
            print("[CM5 DEBUG] First 5 CM5 charges:", charges[:5])
            print("[CM5 DEBUG] Last 5 CM5 charges:", charges[-5:])

    if natoms is not None and len(charges) != natoms:
        print(
            f"[CM5 DEBUG] WARNING: number of CM5 charges ({len(charges)}) "
            f"!= natoms ({natoms})."
        )

    return charges


def assign_cm5_to_mol_by_coords(mol: Chem.Mol, xyzq_path: str, tol=0.2):
    """
    Read CM5 (EL x y z q) and attach q to RDKit atoms via nearest-neighbor
    coordinate matching (Å). If counts match and RMSD small, assumes same order.
    """
    import numpy as np

    elems, coords, charges = read_cm5_xyzq_table(xyzq_path)

    if mol.GetNumAtoms() != len(charges):
        raise ValueError("Atom count mismatch between mol and CM5 file")

    # Get RDKit coordinates
    if mol.GetNumConformers() == 0:
        raise RuntimeError("Molecule has no conformer; cannot match coordinates")
    conf = mol.GetConformer()
    rdk = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    # quick check: if order is likely identical (small RMSD), assign directly
    rmsd_guess = np.sqrt(np.mean((rdk - coords) ** 2))
    if rmsd_guess < tol:
        for atom, q in zip(mol.GetAtoms(), charges):
            atom.SetDoubleProp("_Charge", float(q))
        return

    # robust: NN match
    used = set()
    idx_map = [-1] * len(rdk)
    for i, a in enumerate(rdk):
        d2 = np.sum((coords - a) ** 2, axis=1)
        order = np.argsort(d2)
        match = None
        for j in order:
            if j not in used and np.sqrt(d2[j]) <= tol:
                match = j
                break
        if match is None:
            # if nothing within tol, still take nearest but warn
            match = int(order[0])
        used.add(match)
        idx_map[i] = match

    # assign in RDKit order
    for i, atom in enumerate(mol.GetAtoms()):
        scale = 1.2  # or 1.20
        atom.SetDoubleProp("_Charge", float(scale * charges[idx_map[i]]))


def _load_opls_cfg() -> dict:
    import importlib.resources

    with importlib.resources.open_text("glyze", "opls_glyceride.json") as file:
        data = json.load(file)
    return data


def resolve_opls_type(scaffold_name: str, cfg: dict | None = None) -> str:
    """
    Map your SMARTS scaffold (e.g., 'opls_EST_C') to a real 'opls_####'.
    Falls back to scaffold if not found (so you notice unmapped ones).
    """
    cfg = cfg or _load_opls_cfg()
    return cfg.get("map", {}).get(scaffold_name, scaffold_name)


def opls_type_metadata(opls_type: str, cfg: dict | None = None) -> dict:
    """
    Return metadata for a real OPLS type, e.g., {'mass':..., 'comment':..., 'label':...}
    """
    cfg = cfg or _load_opls_cfg()
    return cfg.get("types", {}).get(opls_type, {})


def validate_scaffold_types_present(
    scaffold_types: list[str], strict: bool = False
) -> list[str]:
    """
    Check which scaffold types are missing in the JSON mapping.
    Returns a list of missing scaffold names. If strict=True, raises on first missing.
    """
    cfg = _load_opls_cfg()
    mp = cfg.get("map", {})
    missing = [t for t in scaffold_types if t not in mp]
    if strict and missing:
        raise KeyError(f"Missing scaffold→opls mapping for: {missing}")
    return missing


def build_top_from_mol_with_charges(
    mol: Chem.Mol,
    prefix: str = "triolein",
    resname: str = "GLYCR",
):
    """
    Given an RDKit molecule with _Charge properties already set,
    run OPLS atom typing and generate .itp, .top, and .gro.
    """
    # Make sure there's a conformer; if not, embed
    if mol.GetNumConformers() == 0:
        _ = rdkit_embed_minimize(mol)

    # Coordinates for .gro (angstroms -> nm conversion handled inside write_gro)
    coords = rdkit_embed_minimize(mol)

    # OPLS scaffold typing
    typer = OPLSAtomTyper()
    types = typer.assign(mol)

    # File names
    itp_fn = f"{prefix}.itp"
    gro_fn = f"{prefix}.gro"
    top_fn = f"{prefix}.top"

    # Write files (write_itp uses _Charge if present)
    write_itp(mol, types, itp_fn, resname=resname)
    write_gro(mol, coords, gro_fn, title=prefix)
    write_top(prefix, itp_fn, top_fn, n_mol=1)

    return itp_fn, top_fn, gro_fn, types


def write_itp(
    mol: Chem.Mol, types: List[Tuple[str, float, str]], fn: str, resname: str = "GLYCR"
):
    # atom names: element plus index
    atom_records: List[AtomRecord] = []
    cfg = _load_opls_cfg() 

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        name = f"{sym}{i+1}"
        tname_scaffold, default_q, _ = types[i]

        # Prefer per-atom _Charge property (e.g., CM5 * 1.20) if present
        if atom.HasProp("_Charge"):
            try:
                q = atom.GetDoubleProp("_Charge")
            except Exception:
                q = default_q
        else:
            q = default_q

        # Mass: keep element mass
        mass = PERIODIC_MASS.get(atom.GetAtomicNum(), 12.0)

        # translate scaffold -> real opls_####
        tname_real = resolve_opls_type(tname_scaffold, cfg=cfg)

        atom_records.append(
            AtomRecord(
                idx=i,
                name=name,
                type=tname_real,
                resname=resname,
                resid=1,
                charge=q,
                mass=mass,
            )
        )

    # Build bonded lists
    bonds, angles, diheds = build_bonded_lists(mol)

    diheds = _filter_problem_dihedrals(diheds, atom_records)

    lines: List[str] = []
    lines.append(f"[ moleculetype ]")
    lines.append(f"; name    nrexcl")
    lines.append(f"{resname}  3")
    lines.append("")
    lines.append("[ atoms ]")
    lines.append(";  nr  type   resnr  resid  atom  cgnr   charge   mass")

    for i, rec in enumerate(atom_records, start=1):
        meta = opls_type_metadata(rec.type, cfg=cfg)
        label = meta.get("label", "")
        comment = f" ; {label}" if label else ""
        lines.append(
            f"{i:5d} {rec.type:<10} {rec.resid:5d} {rec.resname:<6} {rec.name:<6} "
            f"{i:5d} {rec.charge:8.4f} {rec.mass:8.3f}{comment}"
        )

    lines.append("")
    lines.append("[ bonds ]")
    lines.append(";  ai  aj  funct")
    for i_, j_, funct in bonds:
        lines.append(f"{i_+1:5d}{j_+1:5d}{funct:6d}")

    lines.append("")
    lines.append("[ angles ]")
    lines.append(";  ai  aj  ak  funct")
    for i_, j_, k_, funct in angles:
        lines.append(f"{i_+1:5d}{j_+1:5d}{k_+1:5d}{funct:6d}")

    lines.append("")
    lines.append("[ dihedrals ]")
    lines.append(";  ai  aj  ak  al  funct")
    for i_, j_, k_, l_, funct in diheds:
        lines.append(f"{i_+1:5d}{j_+1:5d}{k_+1:5d}{l_+1:5d}{funct:6d}")

    Path(fn).write_text("\n".join(lines))
