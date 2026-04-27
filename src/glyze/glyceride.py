from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from pathlib import Path
import numpy as np
from functools import cached_property

# Fragment values
DELTA_H_MAG = 4.173e7
DELTA_H_DAG = 3.486e7
DELTA_H_TAG = -3.476e7

DELTA_G_MAG = -1.986e7
DELTA_G_DAG = -4.687e7
DELTA_G_TAG = -7.388e7

_POLYMORPH_IDX = {"alpha": 0, "beta_prime": 1, "beta": 2}

R = 8.314
# In form -> Param: (alpha, beta', beta)
# Params are taken from the Seilert PII-DL model. 2021, Wesdorp revisited.
# Take from: 
#              Seilert J, Moorthy AS, Kearsley AJ, Floter E. Revisiting a model
#             to predict pure triglyceride thermodynamic properties: parameter
#             optimization and performance. J Am Oil Chem Soc. 2021;98: 837-850.
params = {
    "h": (2.65, 3.48, 3.78),
    "s": (5.92, 9.71, 11.55),
    "h0": (-25.26, -26.36, -23.9),
    "s0": (-2.47, -13.81, -7.62),
    "hxy": (-16.97, -22.13, -19.96),
    "sxy": (-45.51, -67.35, -66.92),
    "k": (3.78, 2.28, 3.92),
    "x0": (2.68, 3.28, 1.45),
    "hodd": (0, 0, 0),
    "sodd": (0, 0, 0),
    "A0": (-1.39, -2.08, -1.26),
    "B0": (-8.99, -6.55, -14.57),
    "AO": (-9.0581, -7.4543, -8.0481),
    "Aodd": (-0.1958, -0.3075, -0.0193),
    "Bodd": (-0.0025, 0.005, 0.0082),
    "Ax": (0.0029, -0.1036, 0.07413),
    "Ax2": (-0.0619116, -0.018881, -0.0348596),
    "Axy": (0.115128, 0.073941, 0.0077142),
    "Ay": (-0.453461, -0.49721, -0.404136),
    "Ay2": (0.005827, 0.0115995, 0.0111938),
    "Bx": (-0.00111, 0.54997, -0.31675),
    "Bx2": (0.148938, 0.074136, 0.086967),
    "Bxy": (-0.365917, -0.340928, 0.040642),
    "By": (1.41154, 2.34238, 0.5504),
    "By2": (-0.001766, -0.135735, 0.000945),
    "Tinf": (401.15, 401.15, 401.15),
    "AE": (-0.42, -1.13, -1.3),
    "Al": (-2.82, -3.97, -4.4),
    "Ale": (-5.06, -4.55, -12.37),
    "AOO": (-2.14, -0.65, -2.91),
    "AEE": (-1.59, 0.06, 0.21),
    "All": (-1.79, -1.26, -59.47),
    "Alele": (-0.54, 0.55, -25.18),
    "AOl": (-0.12, 0.06, 0.74),
    "Aole": (2, 1.4, 2.94),
    "Alle": (-1.28, -0.89, 57.4),
    "BO": (-8.99, -6.55, -14.57),
    "Bl": (4.47, 5.27, 4.3),
    "Ble": (2.83, 0.3, 30.99),
    "fxy": (0.276020353, 0.807078349, 0.248679327),
    "Hsat": (107.8559346, 136.7393561, 167.6963606),
    "Hunsat": (75.03593461, 115.2993561, 140.2063606),
    "h0hat": (-29.94406539, -44.22064387, -28.86363937),
    "s0hat": (-15.03168627, -68.16672684, -21.75885719),
    "As": (-8.760511124, -5.686821343, -5.751999692),
    "Bs": (-2.539136193, -7.020260231, -1.88388374),
    "Tsat": (320.174013, 333.4100601, 336.4433316),
    "Au": (-10.15051112, -7.766821343, -7.011999692),
    "Bu": (-11.52913619, -13.57026023, -16.45388374),
    "Tunsat": (296.1470406, 308.9663356, 312.5445738),
}


def _optimize_mol(mol: Chem.Mol, confId: int) -> Chem.Mol:
    """
    Optimize the 3D structure of an RDKit molecule with ETKDG v2 and force fields.

    Parameters:
        mol (Chem.mol) : unoptimized RDKit molecule
        confId (int) : configuration state

    Returns:
        Chem.mol : optimized RDKit Molecule
    """

    # Helper: one embed attempt with ETKDG v2 and a toggle for random coords
    def _try_embed(use_random: bool) -> int:
        return AllChem.EmbedMolecule(
            mol,
            maxAttempts=8000,
            randomSeed=0xBEEF,
            useRandomCoords=use_random,
            boxSizeMult=2.0,
            randNegEig=True,
            numZeroFail=1,
            forceTol=0.001,
            ignoreSmoothingFailures=False,
            enforceChirality=True,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            useSmallRingTorsions=False,
            useMacrocycleTorsions=True,
            ETversion=2,
        )

    # Single-conformer attempts: deterministic -> random
    confId = _try_embed(use_random=False)
    if confId == -1:
        confId = _try_embed(use_random=True)

    if confId == -1:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=24,
                maxAttempts=8000,
                randomSeed=0xBEEF,
                useRandomCoords=True,
                boxSizeMult=2.0,
                randNegEig=True,
                numZeroFail=1,
                forceTol=0.001,
                ignoreSmoothingFailures=False,
                enforceChirality=True,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                useSmallRingTorsions=False,
                useMacrocycleTorsions=True,
                ETversion=2,
            )
        )
        if not conf_ids:
            raise RuntimeError("Conformer embedding failed (no conformers generated).")

        # Optimize all with UFF and select best
        res = AllChem.UFFOptimizeMoleculeConfs(mol, confIds=conf_ids, maxIters=1000)
        energies = [r[1] for r in res]
        best_idx = min(range(len(conf_ids)), key=lambda i: energies[i])
        confId = conf_ids[best_idx]

    #  Force-field relaxation of the chosen conformer
    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=confId, maxIters=2000)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, confId=confId, maxIters=2000)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=confId, maxIters=2000)

    # Assign stereochemistry after coords exist
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _fa_key(fa: Optional[FattyAcid]) -> tuple:
    """
    Return a fully comparable, canonical sorting key for a FattyAcid (or None).

    This function is used to impose a total ordering on fatty acids so that
    sn-1 and sn-3 positions can be treated symmetrically in SymmetricGlyceride.

    The returned key is a tuple of immutable, orderable components that
    completely describe the canonical fatty acid structure.

    Rules:
    - None (EMPTY) sorts before any actual FattyAcid.
    - Otherwise, ordering is determined lexicographically by:
        (length, db_positions, db_stereo, branches)
        where each field comes from the canonicalized FattyAcid.
    """
    if fa is None:
        # Empty slot (e.g., diacylglyceride)
        return (0,)

    # Canonicalize first (sorts db positions, normalizes Z/E, etc.)
    fac = fa.canonical()

    # The key must contain only immutable, natively comparable types.
    # Each attribute is already sorted and normalized in canonical().
    return (
        1,  # Tag so all real FAs > EMPTY
        fac.length,  # Chain length
        tuple(fac.db_positions),  # Sorted double bond positions
        tuple(fac.db_stereo),  # Aligned stereochem (Z/E)
        tuple(fac.branches),  # Sorted branches (pos, label)
    )


# TODO: Fix the docstrings in the Glyceride and FattyAcid class
@dataclass(frozen=True)
class FattyAcid:
    """
    Fatty acid chain is built, charteristics of them are classified here and is used throughout
    the package.

    Attributes
    ----------
    (length=[int], db_position=(Tuple[int, ]), db_stero(Tuple[int, ]), branches(Tuple[Tuple[int, str]])):
    length, is for how many carbon chains the free fatty acid contains. db is the double bonds that
    is in the free fatty acid. They can be between any carbon (1 to 24). branches is where the fatty acid
    splits into two strands of carbon chains.

    Fatty acids must have a length [int]

    Class Methods
    ---------

    from_name: create a fatty acid using the naming scheme.

    Methods
    ---------

    canonical: the normalized fatty acid/object, to make sure it is the same fatty acid

    to_rdkit_mol: takes the molecule generated in rdkit and calculates it into moles

    vapor_pressure: calculates the vapor pressure of the fatty acid at a temperture, units in Pascals

    Properties
    ---------
    num_carbons: indicats the total amount of carbons in the free fatty acid

    name: generates an identification of the fatty acid, using the specified naming convention

    [Cashed]:
    _cached_rdkit_mol: the stored molecules moles

    _cached_optimized_rdkit_mol: optimize the stored molecules in moles

    molar_mass: calculate the molar mass of the molecules rdkit generates (fatty acid)

    """

    # Use of fields to ensure mutable objects are not shared between instances
    length: int
    db_positions: Tuple[int, ...] = field(default_factory=tuple)
    db_stereo: Tuple[str, ...] = field(default_factory=tuple)
    branches: Tuple[Tuple[int, str], ...] = field(default_factory=tuple)

    def __post_init__(self):

        # Ensure the input is of the right format
        if self.length < 0:
            raise ValueError("length must be >= 0")
        if len(self.db_positions) != len(self.db_stereo):
            raise ValueError("db_positions and db_stereo must be same length")
        for k in self.db_positions:
            if not (1 <= k <= max(self.length - 1, 0)):
                raise ValueError(
                    f"double-bond position delta{k} out of range for C{self.length}"
                )

        # Check that the branches input is of the right format
        for pos, _ in self.branches:
            if not (1 <= pos <= self.length):
                raise ValueError(
                    f"branch position C{pos} out of range for C{self.length}"
                )
            # TODO: Ensure the branch is valid

    @classmethod
    def from_name(cls, fa_str: str) -> Optional[FattyAcid]:
        """
        Create a Fatty Acid using the naming scheme:

        Fatty acid format:

        N{CC}D{DD}[P{pp}{S}{pp}{S}...][M{pp}...][OH{pp}...]

        N{CC} - Number of Carbons like 06 or 12
        D{DD} - Number of double bonds
        [P{pp}(S){pp{S}...] - double bond position and stereo where S is trans and Z is cis (e.g. 06Z )
        M{pp}... -  Methyl branches at position pp
        OH{pp}... - Hydroxyl branches at position pp.

        Example: N18D1P09Z means 18 carbons, 1 double bond at position 9 with Z (cis) stereo and is oleic acid.

        Args:
            name (str): Name of the fatty acid in the specified format.

        Returns:
            FattyAcid: The corresponding FattyAcid object.

        """

        # Empty case
        if fa_str == "EMPTY":
            return None

        # Ensure the string has the right beginning format
        if not fa_str.startswith("N"):
            raise ValueError(f"Invalid fatty acid format: {fa_str}")

        # Grab the length
        length = int(fa_str[1:3])

        # Grab the double bond count
        if "D" not in fa_str:
            raise ValueError(f"Invalid fatty acid format: {fa_str}")
        d_index = fa_str.index("D")
        num_db = int(fa_str[d_index + 1 : d_index + 3])
        db_positions = []
        db_stereo = []
        branches = []

        # Parse the double bond positions and steric conformation
        i = d_index + 3
        while i < len(fa_str):
            if fa_str[i] == "P":
                i += 1
                for _ in range(num_db):
                    pos = int(fa_str[i : i + 2])
                    db_positions.append(pos)
                    i += 2
                    if i < len(fa_str) and fa_str[i] in (
                        "Z",
                        "E",
                        "z",
                        "e",
                        "C",
                        "c",
                        "T",
                        "t",
                    ):
                        db_stereo.append(fa_str[i].upper())
                        i += 1
                    else:
                        db_stereo.append("Z")  # Default to Z if not specified

            # Parse branches
            elif fa_str[i] == "M":
                i += 1
                while i + 1 < len(fa_str) and fa_str[i : i + 2].isdigit():
                    pos = int(fa_str[i : i + 2])
                    branches.append((pos, "Me"))
                    i += 2
            elif fa_str[i] == "O":
                if fa_str[i : i + 2] == "OH":
                    i += 2
                    while i + 1 < len(fa_str) and fa_str[i : i + 2].isdigit():
                        pos = int(fa_str[i : i + 2])
                        branches.append((pos, "OH"))
                        i += 2
                else:
                    raise ValueError(f"Invalid fatty acid format: {fa_str}")
            else:
                raise ValueError(f"Invalid fatty acid format: {fa_str}")
        if len(db_positions) != num_db:
            raise ValueError(
                f"Number of double bonds does not match positions in: {fa_str}"
            )
        return FattyAcid(length, tuple(db_positions), tuple(db_stereo), tuple(branches))

    def canonical(self) -> "FattyAcid":
        """
        Returns a canonical (normalized) version of the fatty acid. Ensures that
        object signature is identical for equivalent fatty acids.

        Parameters: None

        Returns: FattyAcid
        """

        def norm_st(s: str) -> str:
            s = s.strip().lower()
            return {"cis": "z", "trans": "e"}.get(s, s.upper())  # supports "Z"/"E"

        # Sort the db position, to ensure that list go from least to greatest
        positions = tuple(sorted(self.db_positions))
        # Keep stereo aligned with sorted positions
        if positions:
            # Build mapping from original pos to stereo
            mp = {p: norm_st(st) for p, st in zip(self.db_positions, self.db_stereo)}
            stereo = tuple(mp[p] for p in positions)
        else:
            stereo = tuple()

        # Ensure branches are also sorted
        branches = tuple(sorted((int(p), str(lbl)) for p, lbl in self.branches))
        return FattyAcid(self.length, positions, stereo, branches)

    def vapor_pressure(self, T_K: float) -> float:
        """
        Estimate vapor pressure using the Ceriani & Meirelles (2004)
        group-contribution method (Eqs. A.16-A.20, Table A4).
        """
        n = self.num_carbons

        # Group parameters: (A1k, B1k, C1k, D1k, A2k, B2k, C2k, D2k)
        GP = {
            "CH3": (
                -117.5,
                7232.3,
                -22.7939,
                0.0361,
                0.00338,
                -63.3963,
                -0.00106,
                0.000015,
            ),
            "CH2": (
                8.4816,
                -10987.8,
                1.4067,
                -0.00167,
                -0.00091,
                6.7157,
                0.000041,
                -0.00000126,
            ),
            "COOH": (
                8.0734,
                -20478.3,
                0.0359,
                -0.00207,
                0.00399,
                -63.9929,
                -0.00132,
                0.00001,
            ),
            "CH=CH_cis": (2.4317, 1410.3, 0.7868, -0.004, 0, 0, 0, 0),
        }

        # Molecular weight: CnH(2n)O2 minus 2H per double bond
        n_db = len(self.db_positions)
        Mi = 14.027 * n + 32.0 - 2.016 * n_db

        # Group counts
        n_CH2 = n - 2 - 2 * n_db
        groups = [("CH3", 1), ("CH2", n_CH2), ("COOH", 1)]
        if n_db > 0:
            groups.append(("CH=CH_cis", n_db))

        # Compound-specific parameters for fatty acids
        f0, f1, s0, s1 = 0.001, 0.0, 0.0, 0.0

        # N_CT = total carbons, N_CS = substitute fraction carbons (0 for free FA)
        N_CT = n
        N_CS = 0

        # Build A', B', C', D' (Eqs. A.17–A.20)
        Ap = Bp = Cp = Dp = 0.0
        for grp, Nk in groups:
            A1k, B1k, C1k, D1k, A2k, B2k, C2k, D2k = GP[grp]
            Ap += Nk * (A1k + Mi * A2k)
            Bp += Nk * (B1k + Mi * B2k)
            Cp += Nk * (C1k + Mi * C2k)
            Dp += Nk * (D1k + Mi * D2k)

        # Q correction terms (alpha, beta, gamma, delta)
        corr = f0 + N_CT * f1
        Ap += 3.4443 * corr + (s0 + N_CS * s1)
        Bp += -499.3 * corr
        Cp += 0.6136 * corr
        Dp += -0.00517 * corr

        # Eq. A.16
        return np.exp(Ap + Bp / (T_K**1.5) - Cp * np.log(T_K) - Dp * T_K)

    def to_rdkit_mol(self, optimize: bool = False) -> Chem.Mol:
        return Chem.Mol(
            self._cached_optimized_rdkit_mol if optimize else self._cached_rdkit_mol
        )

    def _build_rdkit_mol(self, optimize: bool = False) -> Chem.Mol:
        """
        Convert the fatty acid to an RDKit molecule.

        Returns:
            Chem.Mol: The RDKit molecule representing the fatty acid.
        """
        rw = Chem.RWMol()

        # add the carboxyl group
        carboxyl = rw.AddAtom(Chem.Atom(6))  # the atoms
        o_double = rw.AddAtom(Chem.Atom(8))
        o_single = rw.AddAtom(Chem.Atom(8))

        rw.AddBond(carboxyl, o_double, Chem.BondType.DOUBLE)
        rw.AddBond(carboxyl, o_single, Chem.BondType.SINGLE)

        # Build the rest of the chain (C1...Cn)
        chain_idx = []
        last = carboxyl  # adds to the chain
        for i in range(2, self.length + 1):
            ci = rw.AddAtom(Chem.Atom(6))
            chain_idx.append(ci)
            if last is not None: 
                rw.AddBond(last, ci, Chem.BondType.SINGLE)
            last = ci
        # Branhces
        for pos, lbl in self.branches: 
            # Ensure pos maps to chain_idx[pos -1]
            if lbl.lower() in ("me", "methyl"):
                if 1 <= pos <= self.length:
                    c = rw.AddAtom(Chem.Atom(6))
                    rw.AddBond(chain_idx[pos - 2], c, Chem.BondType.SINGLE)
            else:
                raise NotImplementedError(
                    f"Branch label '{lbl}' not implemented yet (only 'Me')."
                )
        # Double bonds along chain
        # Map positions k to indices
        for k, st in zip(self.db_positions, self.db_stereo):
            a = chain_idx[k - 2]
            b = chain_idx[k - 1]
            bond = rw.GetBondBetweenAtoms(a, b)
            if bond is None:
                raise RuntimeError("Internal: expected a bond to set C=C.")
            bond.SetBondType(Chem.BondType.DOUBLE)
            # Assign E/Z stereo if possible
            # Pick on neighbor on each side that is not the other double-bond atom
            a_neighbors = [
                nbr.GetIdx()
                for nbr in rw.GetAtomWithIdx(a).GetNeighbors()
                if nbr.GetIdx() != b
            ]
            b_neighbors = [
                nbr.GetIdx()
                for nbr in rw.GetAtomWithIdx(b).GetNeighbors()
                if nbr.GetIdx() != a
            ]
            if a_neighbors and b_neighbors:
                bond.SetStereoAtoms(a_neighbors[0], b_neighbors[0])
                norm = st.strip().lower()
                if norm in ("z", "ciz"):
                    bond.SetStereo(Chem.BondStereo.STEREOZ)
                elif norm in ("e",):
                    bond.SetStereo(Chem.BondStereo.STEREOE)
                else:
                    pass

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        if optimize:
            mol = _optimize_mol(mol, confId=-1)
        return mol

    @cached_property
    def molar_mass(self) -> float:
        """Calculate the molar mass of the fatty acid in g/mol"""
        # Build RDkit molecule and sum atomic masses
        mol = self.to_rdkit_mol()
        mass = 0
        for atom in mol.GetAtoms():
            mass += atom.GetMass()

        return mass

    @cached_property
    def _cached_rdkit_mol(self) -> Chem.Mol:
        return self._build_rdkit_mol(optimize=False)

    @cached_property
    def _cached_optimized_rdkit_mol(self) -> Chem.Mol:
        return self._build_rdkit_mol(optimize=True)

    @property
    def num_carbons(self) -> int:
        """Return the number of carbons in the fatty acid"""
        return self.length

    @property
    def name(self) -> str:
        """
        Generate a standardized name for the fatty acid using the format:

        N{CC}D{DD}[P{pp}{S}{pp}{S}...][M{pp}...][OH{pp}...]

        N{CC} - Number of Carbons like 06 or 12
        D{DD} - Number of double bonds
        [P{pp}(S){pp{S}...] - double bond position and stereo where S is trans and Z is cis (e.g. 06Z )
        M{pp}... -  Methyl branches at position pp
        OH{pp}... - Hydroxyl branches at position pp.

        Example: N18D1P09Z means 18 carbons, 1 double bond at position 9 with Z (cis) stereo and is oleic acid.
        """
        parts = [f"N{self.length:02d}", f"D{len(self.db_positions):02d}"]
        if self.db_positions:
            pos_stereo = []
            for p, s in zip(self.db_positions, self.db_stereo):
                pos_stereo.append(f"P{p:02d}{s.upper()}")
            parts.extend(pos_stereo)
        for bpos, _ in self.branches:
            parts.append(f"M{bpos:02d}")  # Only 'Me' supported now
        return "".join(parts)

    @property
    def num_dbs(self) -> int:
        """Return the number of double bonds present in the fatty acid"""
        return len(self.db_positions)


class Glyceride:
    """
    Description of a glyceride

    Attributes:
    -----------
    sn (Tuple[Optional[FattyAcid], Optional[FattyAcid], Optional[FattyAcid]]): tuple of three Optional[FattyAcid] in sn-1, sn-2, sn-3 order.
        Use None for an emtpy chain (e.g. diacylglyceride embedding).

    Class Methods:
    --------------
    from_name: Create a Glyceride usign the naming scheme

    Methods:
    --------
    add_fatty_acid: Add a fatty acid to the glyceride and return a deepcopy of the new glyceride.
    signature_tuple: canoncial, detechs the glyceride entered
    remove_fatty_acid: Remove a fatty acid to the glyceride and return a deepcopy of the new glyceride.
    swap_fatty_acids: Swaps two of the fatty acids in the glycerides and returns a deepcopy of the new glyceride.
    to_rdkit_mol: Takes the molecule build in rdkit and returns the chemical moles for simulation purposes or prebuilt code purposes.
    rdkit_mol_to_gaussian_gjf: The molecule generated in rdkit and puts it gaussian for Molecular Dynamic simulations.
    vapor_pressure: The vapor pressure for the glyceride at any given temperature.


    [Static Method]:
    mol_to_pdb: Convert an RDKit Mol object into a PDB file at the specified path.
    The file will persist until manually deleted.


    Properties:
    -----------
    molar_mass (float): Calculate the molar mass of a glyceride in g/mol
    num_fatty_acids (int): Finds the number of fatty acids in a glyceride (1, 2, or 3 fatty acids)
    num_carbons (int): Finds the number of carbons in the glyceride molecule (3 to 75 carbons or more)
    enthalpy_of_vaporitzation (float): the enthalphy of vaporization of a glyceride (kj/mol)
    gibbs_of_vaporization (float): the gibbs free energy of a glyceride (kj/mol)
    chain_lengths (Tuple): Tuple of chain lengths (0 if empty) in sn-1, sn-2, sn-3 order.
    name (str): Generate a standardized name for the glycerides
    """

    def __init__(
        self, sn: Tuple[Optional[FattyAcid], Optional[FattyAcid], Optional[FattyAcid]]
    ):
        self.sn = sn
        if len(self.sn) != 3:
            raise ValueError("sn must have length 3 (sn-1, sn-2, sn-3)")

    @classmethod
    def from_name(cls, name: str) -> "Glyceride":
        """
        Create a Glyceride using the naming scheme:

        Fatty acid format:

        N{CC}D{DD}[P{pp}{S}{pp}{S}...][M{pp}...][OH{pp}...]

        N{CC} - Number of Carbons like 06 or 12
        D{DD} - Number of double bonds
        [P{pp}(S){pp{S}...] - double bond position and stereo where S is trans and Z is cis (e.g. 06Z )
        M{pp}... -  Methyl branches at position pp
        OH{pp}... - Hydroxyl branches at position pp.

        Example: N18D1P09Z means 18 carbons, 1 double bond at position 9 with Z (cis) stereo and is oleic acid.

        Glyceride format:

        G_{FA1}_{FA2}_{FA3}

        Example: G_N18D1P09Z_N18D1P09Z_N18D1P09Z is triolein.

        Args:
            name (str): Name of the glyceride in the specified format.

        Returns:
            Glyceride: The corresponding Glyceride object.
        """

        # Make sure the input string has the right format (roughly)
        if not name.startswith("G_") and not name.startswith("Glycerol"):
            raise ValueError(f"Invalid glyceride format: {name}")

        # Edge case: Glycerol (no fatty acids)
        if name == "Glycerol":
            return cls((None, None, None))

        # Split up the string into its fatty acids.
        parts = name[2:].split("_")
        if len(parts) != 3:
            raise ValueError(f"Glyceride must have three fatty acids: {name}")

        # Create instances of the fatty acids
        fa1 = FattyAcid.from_name(parts[0])
        fa2 = FattyAcid.from_name(parts[1])
        fa3 = FattyAcid.from_name(parts[2])

        # Create an instance of the Glyceride class
        return cls((fa1, fa2, fa3))

    def signature_tuple(self) -> Tuple:
        """Canonical, hashable structure signature (topology only)."""
        parts = []
        for fa in self.sn:
            if fa is None:
                parts.append(("EMPTY",))
            else:
                fac = fa.canonical()
                parts.append(
                    ("FA", fac.length, fac.db_positions, fac.db_stereo, fac.branches)
                )
        return tuple(parts)

    def add_fatty_acid(self, index: int, fatty_acid: FattyAcid, deep_copy: bool = True):
        """
        Add a fatty acid to the glyceride and return a deepcopy of the new glyceride.

        Paramters:
        ----------
            index (int): Index (0, 1, or 2) to add the fatty acid to.
            fatty_acid (FattyAcid): The fatty acid to add.
            deep_copy (bool): Whether to perform a deep copy of the glyceride.

        Returns:
        --------
            Glyceride: A new Glyceride instance with the added fatty acid.
        """
        if index not in (0, 1, 2):
            raise ValueError("Index must be 0, 1, or 2.")
        if self.sn[index] is not None:
            raise ValueError(f"Position sn-{index + 1} is already occupied.")

        if deep_copy:
            new_sn = list(self.sn)
            new_sn[index] = copy.deepcopy(fatty_acid)
            return self.__class__(tuple(new_sn))
        else:
            new_sn = list(self.sn)
            new_sn[index] = fatty_acid
            self.sn = tuple(new_sn)
            return self

    def remove_fatty_acid(self, index: int):
        """
        remove a fatty acid to the glyceride and return a removed fatty acid of the new glyceride.

        Paramters:
            index (int): Index (0, 1, or 2) to add the fatty acid to.

        Returns:
            Glyceride: A new Glyceride instance with the fatty acid removed.
        """
        if index not in (0, 1, 2):
            raise ValueError("Index must be 0, 1, or 2.")
        if self.sn[index] is None:
            raise ValueError(f"Position sn-{index} is already empty.")

        new_sn = list(self.sn)
        fa = new_sn[index]
        new_sn[index] = None
        return self.__class__(tuple(new_sn)), fa

    def swap_fatty_acids(self, index1: int, index2: int, deep_copy: bool = True):
        """
        Swap two fatty acids in the glyceride and return a deepcopy of the new glyceride.

        Paramters:
            index1 (int): Index (0, 1, or 2) of the first fatty acid to swap.
            index2 (int): Index (0, 1, or 2) of the second fatty acid to swap.
            deep_copy (bool): Whether to perform a deep copy of the glyceride.

        Returns:
            Glyceride: A new Glyceride instance with the swapped fatty acids.
        """
        if index1 not in (0, 1, 2) or index2 not in (0, 1, 2):
            raise ValueError("Indices must be 0, 1, or 2.")
        if index1 == index2:
            raise ValueError("Indices must be different to perform a swap.")

        if deep_copy:
            new_sn = list(copy.deepcopy(self.sn))
            new_sn[index1], new_sn[index2] = new_sn[index2], new_sn[index1]
            return self.__class__(tuple(new_sn))
        else:
            new_sn = list(self.sn)
            new_sn[index1], new_sn[index2] = new_sn[index2], new_sn[index1]
            self.sn = tuple(new_sn)
            return self

    @staticmethod
    def mol_to_pdb(mol: Chem.Mol, filepath: Union[str, Path]) -> Path:
        """
        Convert an RDKit Mol object into a PDB file at the specified path.
        The file will persist until manually deleted.

        Args:
            mol (Chem.Mol): The RDKit molecule to convert to PDB format
            filepath (Union[str, Path]): Full path where the PDB file should be saved

        Returns:
            Path: Path to the created PDB file
        """
        # Convert to Path object if string
        pdb_filepath = Path(filepath)

        # Create parent directories if they don't exist
        pdb_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write the PDB file
        Chem.MolToPDBFile(mol, str(pdb_filepath))

        print(f"PDB file created at: {pdb_filepath}")
        return pdb_filepath

    def _build_rdkit_mol(self, optimize: bool = True) -> Chem.Mol:
        """
        Build an RDKit molecule for the given Glyceride, embed in 3D, and relax.
        Uses kwargs-only ETKDG (ETversion=2) for compatibility with your RDKit.

        Args:
            optimize (bool): Whether to optimize the 3D structure with force fields.
                If False, only embedding is done.

        Returns:
            Chem.Mol: The RDKit molecule with 3D coordinates.
        """
        rw, sn_os, _ = (
            self._build_glycerol_backbone()
        )  # the chem mol func and the indices of the backbone
        for idx, fa in enumerate(self.sn):
            if fa is None:
                continue
            fac = fa.canonical()
            carbonyl_c, _ = self._build_acyl_chain(fac, rw)
            rw.AddBond(sn_os[idx], carbonyl_c, Chem.BondType.SINGLE)  # ester bond

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        if optimize:
            mol = _optimize_mol(mol, confId=-1)

        return mol

    def to_rdkit_mol(self, optimize: bool = True) -> Chem.Mol:
        if optimize:
            cached = getattr(self, "_cached_optimized_rdkit_mol", None)
            if cached is None:
                cached = self._build_rdkit_mol(optimize=True)
                self._cached_optimized_rdkit_mol = cached
            return Chem.Mol(cached)

        cached = getattr(self, "_cached_rdkit_mol", None)
        if cached is None:
            cached = self._build_rdkit_mol(optimize=False)
            self._cached_rdkit_mol = cached
        return Chem.Mol(cached)

    def rdkit_mol_to_gaussian_gjf(
        self,
        mol: Chem.Mol,
        gjf_path: str,
        jobname: str = "glyceride_opt",
        mem: str = "32GB",
        nproc: int = 32,
        chg: int = 0,
        mult: int = 1,
    ) -> None:
        """
        Write a Gaussian .gjf file from an RDKit molecule, matching your
        working Gaussian input style.

        Args:
            mol (Chem.Mol): The RDKit molecule with 3D coordinates.
            gjf_path (str): Output path for the Gaussian .gjf file.
            jobname (str): Title for the Gaussian job (also used in %chk).
            mem (str): Memory allocation for Gaussian (e.g., "32GB").
            nproc (int): Number of processors for Gaussian.
            chg (int): Total charge of the molecule.
            mult (int): Spin multiplicity of the molecule.
        """
        if mol.GetNumConformers() == 0:
            raise RuntimeError(
                "No 3D conformer found; create with to_rdkit_mol(optimize=True)."
            )

        conf = mol.GetConformer()
        chk_name = f"{jobname}.chk"

        lines: list[str] = []

        # Example:
        # %chk=molecule.chk
        # %mem=32GB
        # %NProcShared=32
        lines.append(f"%chk={chk_name}")
        lines.append(f"%mem={mem}")
        lines.append(f"%NProcShared={nproc}")

        # As specified in the paper:
        lines.append(
            "#P B3LYP/6-311G(d,p) EmpiricalDispersion=GD3BJ Opt SCF=Tight Int=UltraFine"
        )

        # Blank line
        lines.append("")

        # Title line
        lines.append(jobname)

        # Blank line before charge/multiplicity
        lines.append("")

        # Charge and multiplicity line, e.g. "0 1"
        lines.append(f"{chg} {mult}")

        # Cartesian coordinates
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            lines.append(
                f"{atom.GetSymbol():<2}  {pos.x: .6f}  {pos.y: .6f}  {pos.z: .6f}"
            )

        # Final blank line
        lines.append("\n")

        # Write to disk
        with open(gjf_path, "w") as f:
            f.write("\n".join(lines))

    def vapor_pressure(self, T_K: float) -> float:
        """
        Estimate glyceride vapor pressure using the Ceriani & Meirelles
        (2004) group-contribution method (Eqs. A.16-A.20, Table A4).
        """
        # Group parameters: (A1k, B1k, C1k, D1k, A2k, B2k, C2k, D2k)
        GP = {
            "CH3": (
                -117.5,
                7232.3,
                -22.7939,
                0.0361,
                0.00338,
                -63.3963,
                -0.00106,
                0.000015,
            ),
            "CH2": (
                8.4816,
                -10987.8,
                1.4067,
                -0.00167,
                -0.00091,
                6.7157,
                0.000041,
                -0.00000126,
            ),
            "CH=CH_cis": (2.4317, 1410.3, 0.7868, -0.004, 0, 0, 0, 0),
            "COO": (1.843, 526.5, 0.6584, -0.00368, 0, 0, 0, 0),
            "OH": (28.4723, -16694, 3.257, 0, 0.00485, 0, 0, 0),
            "glycerol": (688.3, -349293, 122.5, -0.1814, -0.00145, 0, 0, 0),
        }

        fa = [x for x in self.sn if x is not None]
        if not fa:
            raise ValueError("Glyceride has no fatty acid chains")

        n_fa = len(fa)

        # Molecular weight of the glyceride
        Mi = 92.094  # glycerol
        for f in fa:
            fa_mw = 14.027 * f.num_carbons + 32.0 - 2.016 * len(f.db_positions)
            Mi += fa_mw - 18.015  # subtract water per ester bond

        # Total carbons (all FA chains + 3 glycerol carbons)
        N_CT = sum(f.num_carbons for f in fa) + 3
        N_CS = 0  # no substitute fraction for acylglycerols

        # Count functional groups
        total_CH3 = 0
        total_CH2 = 0
        total_db = 0

        for f in fa:
            n_db = len(f.db_positions)
            total_CH3 += 1
            total_CH2 += f.num_carbons - 2 - 2 * n_db
            total_db += n_db

        groups = [
            ("glycerol", 1),
            ("CH3", total_CH3),
            ("CH2", total_CH2),
            ("COO", n_fa),
        ]
        if total_db > 0:
            groups.append(("CH=CH_cis", total_db))

        # Free hydroxyl groups on un-esterified glycerol positions
        n_free_oh = 3 - n_fa
        if n_free_oh > 0:
            groups.append(("OH", n_free_oh))

        # Compound-specific parameters for acylglycerols
        f0, f1, s0, s1 = 0.0, 0.0, 0.0, 0.0

        # Build A', B', C', D' (Eqs. A.17–A.20)
        Ap = Bp = Cp = Dp = 0.0
        for grp, Nk in groups:
            A1k, B1k, C1k, D1k, A2k, B2k, C2k, D2k = GP[grp]
            Ap += Nk * (A1k + Mi * A2k)
            Bp += Nk * (B1k + Mi * B2k)
            Cp += Nk * (C1k + Mi * C2k)
            Dp += Nk * (D1k + Mi * D2k)

        # Q correction terms
        corr = f0 + N_CT * f1
        Ap += 3.4443 * corr + (s0 + N_CS * s1)
        Bp += -499.3 * corr
        Cp += 0.6136 * corr
        Dp += -0.00517 * corr

        # Eq. A.16
        return np.exp(Ap + Bp / (T_K**1.5) - Cp * np.log(T_K) - Dp * T_K)

    def melting_enthalpy(self, polymorph: str) -> float:
        """
        Calculates the melting enthalpy of the glyceride.

        Parameters:
        -----------
            polymorph (str): The polymorph of the pure TAG solid. Must be one
                            of "alpha", "beta_prime", or "beta".

        Returns:
        --------
            float: Melting enthalpy of the pure TAG in the solid phase [kJ/mol].
        """
        if polymorph not in _POLYMORPH_IDX:
            raise NameError(
                "The input polymorph must be one of: alpha, beta_prime, beta"
            )

        j = _POLYMORPH_IDX[polymorph]

        def saturated_melting_enthalpy(glyceride: "Glyceride", polymorph: str) -> float:
            n = sum(glyceride.chain_lengths)
            n1 = glyceride.sn[0].length
            n2 = glyceride.sn[1].length
            n3 = glyceride.sn[2].length
            Q = glyceride.chain_lengths[1]
            P = min(n1, n3)
            R = max(n1, n3)
            x = Q - P
            y = R - P

            k = params["k"][j]
            x0 = params["x0"][j]
            fxy = 2 - np.exp(-(((x - x0) / k) ** 2)) - np.exp(-((y / k) ** 2))

            h = params["h"][j]
            h0 = params["h0"][j]
            hxy = params["hxy"][j]
            hodd = params["hodd"][j]

            return (
                h * n
                + h0
                + hxy * fxy
                + hodd * Glyceride._fodd(n1, n2, n3) * Glyceride._f_beta(polymorph)
            )

        def unsaturated_melting_enthalpy(
            glyceride: "Glyceride", polymorph: str
        ) -> float:
            nO = nE = nJ = nN = 0

            for fa in glyceride.sn:
                if fa is None:
                    continue
                if fa.num_dbs == 1:
                    nO += 1
                elif fa.num_dbs == 2:
                    nJ += 1
                elif fa.num_dbs >= 3:
                    nN += 1

            hf_sat = saturated_melting_enthalpy(glyceride, polymorph)

            hol = params["AE"][j]
            hel = params["Ale"][j]
            hli = params["Al"][j]

            return hf_sat + hol * nO + hel * nE + hli * nJ

        if sum(fa.num_dbs for fa in self.sn) == 0:
            return saturated_melting_enthalpy(self, polymorph)
        else:
            return unsaturated_melting_enthalpy(self, polymorph)

    def melting_temp(self, polymorph: str) -> float:
        """
        Calculates the melting temperature of the glyceride.

        Parameters:
        -----------
            polymorph (str): The polymorph of the pure TAG solid. Must be one
                            of "alpha", "beta_prime", or "beta".

        Returns:
        --------
            float: Melting temperature of the pure TAG [K].
        """
        if polymorph not in _POLYMORPH_IDX:
            raise NameError(
                "The input polymorph must be one of: alpha, beta_prime, beta"
            )

        j = _POLYMORPH_IDX[polymorph]

        def saturated_melting_temp(glyceride: "Glyceride", polymorph: str) -> float:
            n = sum(glyceride.chain_lengths)
            n1 = glyceride.sn[0].length
            n2 = glyceride.sn[1].length
            n3 = glyceride.sn[2].length
            Q = glyceride.chain_lengths[1]
            P = min(n1, n3)
            R = max(n1, n3)
            x = Q - P
            y = R - P
            fodd = Glyceride._fodd(n1, n2, n3)

            # Method 1:
            # Asat = (
            #     params["A0"][j]
            #     + params["Aodd"][j] * fodd
            #     + params["Ax"][j]  * x
            #     + params["Ax2"][j] * x ** 2
            #     + params["Axy"][j] * x * y
            #     + params["Ay"][j]  * y
            #     + params["Ay2"][j] * y ** 2
            # )
            # Bsat = (
            #     params["B0"][j]
            #     + params["Bodd"][j] * fodd
            #     + params["Bx"][j]  * x
            #     + params["Bx2"][j] * x ** 2
            #     + params["Bxy"][j] * x * y
            #     + params["By"][j]  * y
            #     + params["By2"][j] * y ** 2
            # )

            # Method 2:
            hhat = (
                params["h0"][j]
                + (params["hxy"][j] * params["fxy"][j])
                + (
                    params["hodd"][j]
                    * Glyceride._fodd(n1, n2, n3)
                    * Glyceride._f_beta(polymorph)
                )
            )
            shat = (
                params["s0"][j]
                + (params["sxy"][j] * params["fxy"][j])
                + (
                    params["sodd"][j]
                    * Glyceride._fodd(n1, n2, n3)
                    * Glyceride._f_beta(polymorph)
                    * (
                        R
                        * np.log(2)
                        * Glyceride._f_beta(polymorph)
                        * Glyceride._fasym(y)
                    )
                )
            )

            Asat = (hhat / params["h"][j]) - (shat / params["s"][j])
            Bsat = shat / params["s"][j]

            Tinf = params["Tinf"][j]
            return Tinf * (1 + (Asat / n) - ((Asat * Bsat) / (n**2)))

        #TODO
        def unsaturated_melting_temp(glyceride: "Glyceride", polymorph) -> float:
            
            n = sum(glyceride.chain_lengths)
            n1 = glyceride.sn[0].length
            n2 = glyceride.sn[1].length
            n3 = glyceride.sn[2].length

            Q = glyceride.chain_lengths[1]
            P = min(n1, n3)
            R = max(n1, n3)

            x = Q - P
            y = R - P

            fodd = Glyceride._fodd(n1, n2, n3)

            # count unsaturation types
            nO = nE = nJ = nN = 0
            for fa in glyceride.sn:
                if fa is None:
                    continue
                if fa.num_dbs == 1:
                    nO += 1
                elif fa.num_dbs == 2:
                    nJ += 1
                elif fa.num_dbs >= 3:
                    nN += 1

            nOO = max(0, nO - 1)
            nJJ = max(0, nJ - 1)
            nNN = max(0, nN - 1)

            nOJ = nO * nJ
            nON = nO * nN
            nJN = nJ * nN

            # saturated base
            As = (
                params["A0"][j]
                + params["Aodd"][j] * fodd
                + params["Ax"][j] * x
                + params["Ax2"][j] * x**2
                + params["Axy"][j] * x * y
                + params["Ay"][j] * y
                + params["Ay2"][j] * y**2
            )

            Bs = (
                params["B0"][j]
                + params["Bodd"][j] * fodd
                + params["Bx"][j] * x
                + params["Bx2"][j] * x**2
                + params["Bxy"][j] * x * y
                + params["By"][j] * y
                + params["By2"][j] * y**2
            )

            Au = (
                As
                + params["AE"][j] * nO
                + params["Al"][j] * nJ
                + params["Ale"][j] * nN
                + params["AOO"][j] * nOO
                + params["All"][j] * nJJ
                + params["Alele"][j] * nNN
                + params["AOl"][j] * nOJ
                + params["Aole"][j] * nON
                + params["Alle"][j] * nJN
            )

            Bu = (
                Bs
                + params["BO"][j] * nO
                + params["Bl"][j] * nJ
                + params["Ble"][j] * nN
            )

            Tinf = params["Tinf"][j]

            return Tinf * (1 + (Au / n) - ((Au * Bu) / (n**2)))

        if sum(fa.num_dbs for fa in self.sn) == 0:
            return saturated_melting_temp(self, polymorph)
        else:
            return unsaturated_melting_temp(self, polymorph)

    # # find vapor pressure based on temperature
    # def vapor_pressure(self, T) -> float:
    #     # THIS IS STILL WRONG!! TODO: FIND A BETTER MODEL
    #     return np.exp(
    #         (-self.gibbs_of_vaporitzation / (8314 * 298.15))          + (
    #             (self.enthalpy_of_vaporitzation / (8314*298.15)) * ((1 / 298.15) - (1 / T))
    #         )
    #     )

    def _add_branch_methyl(self, rw: Chem.RWMol, carbon_idx: int) -> None:
        """Attach a methyl (-CH3) to the given carbon atom index."""
        c = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(carbon_idx, c, Chem.BondType.SINGLE)

    # how come there is a missing oxygen here?
    def _build_acyl_chain(self, fa: FattyAcid, rw: Chem.RWMol) -> Tuple[int, List[int]]:
        """
        Build an acyl group for the fatty acid into rw:
           O=C(-) — C2 — C3 — ... — Cn
        Returns:
            (carbonyl_C_idx, [C1=carbonyl, C2, ..., Cn] indices)
        """
        # Carbonyl carbon + caarbonyl oxygen (double-bond O)
        c1 = rw.AddAtom(Chem.Atom(6))  # carbonyl carbon (C1)
        o_dbl = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c1, o_dbl, Chem.BondType.DOUBLE)

        # Build the rest of the chain (C2...Cn)
        chain_idx = [c1]
        last = c1
        for i in range(2, fa.length + 1):
            ci = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(last, ci, Chem.BondType.SINGLE)
            chain_idx.append(ci)
            last = ci

        # Branches
        for pos, lbl in fa.branches:
            # Ensure pos maps to chain_idx[pos - 1]
            if lbl.lower() in ("me", "methyl"):
                if 1 <= pos <= fa.length:
                    self._add_branch_methyl(rw, chain_idx[pos - 1])

            else:
                raise NotImplementedError(
                    f"Branch label '{lbl}' not implemented yet (only 'Me')."
                )

        # Double bonds along chain
        # Map positions k (C_k -- C_{k + 1}) to indices (chain_idx[k - 1], chain_idx[k])
        for k, st in zip(fa.db_positions, fa.db_stereo):
            if k < 2:
                # Avoid making the carbonyl single bond a C=C;
                continue
            a = chain_idx[k - 1]
            b = chain_idx[k]
            bond = rw.GetBondBetweenAtoms(a, b)
            if bond is None:
                raise RuntimeError("Internal: expected a bond to set C=C.")
            bond.SetBondType(Chem.BondType.DOUBLE)

            # Assign E/Z stereo if possible
            # We need to pick one neighbor on each side that is not the other double-bond atom.
            # Left neighbors of 'a':
            a_neighbors = [
                nbr.GetIdx()
                for nbr in rw.GetAtomWithIdx(a).GetNeighbors()
                if nbr.GetIdx() != b
            ]
            # Right neighbors of 'b':
            b_neighbors = [
                nbr.GetIdx()
                for nbr in rw.GetAtomWithIdx(b).GetNeighbors()
                if nbr.GetIdx() != a
            ]
            if a_neighbors and b_neighbors:
                # Choose first neighbor on each side for stereo refs
                bond.SetStereoAtoms(a_neighbors[0], b_neighbors[0])
                norm = st.strip().lower()
                if norm in ("z", "cis"):
                    bond.SetStereo(Chem.BondStereo.STEREOZ)
                elif norm in ("e", "trans"):
                    bond.SetStereo(Chem.BondStereo.STEREOE)
                else:
                    pass

        return c1, chain_idx

    # where are the hydrogens?
    def _build_glycerol_backbone(
        self,
    ) -> Tuple[Chem.RWMol, Tuple[int, int, int], List[int]]:
        """
        Build glycerol (as triol) and return:
        rw_mol, (o_sn1, o_sn2, o_sn3), carbon_indices
        Skeleton (numbering of O for clarity):
        HO-CH2-(O2)CH-(O3)CH2-O1H
        We'll keep three hydroxyl oxygens to esterify later.
        """
        # Initialize empty RWMol
        rw = Chem.RWMol()
        # Carbons
        c1 = rw.AddAtom(Chem.Atom(6))  # CH2 (sn-1 carbon)
        c2 = rw.AddAtom(Chem.Atom(6))  # CH (sn-2 carbon)
        c3 = rw.AddAtom(Chem.Atom(6))  # Ch2 (sn-3 carbon)

        # Connect backbone
        rw.AddBond(c1, c2, Chem.BondType.SINGLE)
        rw.AddBond(c2, c3, Chem.BondType.SINGLE)

        # Add Hydroxyls
        o1 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c3, o1, Chem.BondType.SINGLE)
        o2 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c1, o2, Chem.BondType.SINGLE)
        o3 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c2, o3, Chem.BondType.SINGLE)

        return rw, (o2, o3, o1), [c1, c2, c3]

    @staticmethod
    def _fasym(y):
        return 0 if y == 0 else 1

    @staticmethod
    def _fodd(n1, n2, n3):
        return 1 if n1 % 2 == 1 or n2 % 2 == 1 or n3 % 2 == 1 else 0

    @staticmethod
    def _f_beta(polymorph):
        return 1 if polymorph == "beta" else 0

    @property
    def molar_mass(self) -> float:
        """Calculate the molar mass of a glyceride in g/mol"""
        cached = getattr(self, "_cached_molar_mass", None)
        if cached is None:
            mol = self.to_rdkit_mol(optimize=False)
            cached = 0
            for atom in mol.GetAtoms():
                cached += atom.GetMass()
            self._cached_molar_mass = cached
        return cached

    @property
    def num_fatty_acids(self) -> int:
        """Number of fatty acid chains (1, 2, or 3)."""
        return sum(1 for fa in self.sn if fa is not None)

    @property
    def num_carbons(self) -> int:
        """Return the number of carbons in the glyceride molecule"""
        # Add 3 for the glycerol molecule
        return sum(self.chain_lengths) + 3

    # enthalphy of vaportization for mags, dags, tags
    @property
    def enthalpy_of_vaporitzation(self) -> float:
        """
        Returns the enthalpy of vaporitzation

        Parameters: None

        Returns: Enthalpy of vaporization
        """
        fa = [x for x in self.sn if x]
        # mag
        if len(fa) == 1:
            return DELTA_H_MAG + fa[0].num_carbons * 2093479.64 + 31397826.69
        # dag
        elif len(fa) == 2:
            return DELTA_H_DAG + sum(
                fa_i.num_carbons * 2093479.64 + 31397826.69 for fa_i in fa
            )
        # tag
        else:
            return DELTA_H_TAG + sum(
                fa_i.num_carbons * 2093479.64 + 31397826.69 for fa_i in fa
            )

    @property
    def chain_lengths(self) -> Tuple[int, int, int]:
        """Tuple of chain lengths (0 if empty) in sn-1, sn-2, sn-3 order."""

        def L(fa: Optional[FattyAcid]) -> int:
            return fa.length if fa is not None else 0

        return (L(self.sn[0]), L(self.sn[1]), L(self.sn[2]))

    @property
    def name(self) -> str:
        """Generate a standardized name for the glyceride."""

        def fa_name(fa: Optional[FattyAcid]) -> str:
            if fa is None:
                return "EMPTY"
            parts = [f"N{fa.length:02d}", f"D{len(fa.db_positions):02d}"]
            if fa.db_positions:
                pos_stereo = []
                for p, s in zip(fa.db_positions, fa.db_stereo):
                    pos_stereo.append(f"P{p:02d}{s.upper()}")
                parts.extend(pos_stereo)
            for bpos, _ in fa.branches:
                parts.append(f"M{bpos:02d}")  # Only 'Me' supported now
            return "".join(parts)

        if all(fa is None for fa in self.sn):
            return "Glycerol"

        return "G_" + "_".join(fa_name(fa) for fa in self.sn)

    def __eq__(self, other):
        """Equality based on the signature tuple."""
        if not isinstance(other, Glyceride):
            return NotImplemented
        return self.signature_tuple() == other.signature_tuple()

    def __hash__(self):
        """Hash based on the signature tuple."""
        return hash((self.signature_tuple(),))

    def __str__(self):
        lines = []
        for idx, fa in enumerate(self.sn):
            fa_str = "EMPTY" if fa is None else fa.name
            lines.append(f"sn-{idx + 1}: {fa_str}")
        return "\n".join(lines)

    def __gt__(self, other: Glyceride):
        """Greater than based on chain length"""
        if isinstance(other, Glyceride):
            return self.chain_lengths > other.chain_lengths
        else:
            assert TypeError("Must compare two Glyceride objects")

    def __lt__(self, other: Glyceride):
        """Lesser than based on chain length"""
        if isinstance(other, Glyceride):
            return self.chain_lengths < other.chain_lengths
        else:
            assert TypeError("Must compare two Glyceride objects")


class SymmetricGlyceride(Glyceride):
    """Glyceride where sn-1 and sn-3 are considered equivalent for equality/hash.
    Methods
    -------
    signature_tuples: lists the tuples to return the glyceride based on the name

    Property
    --------
    name: generates a name for the glyceride

    """

    def signature_tuple(self) -> tuple:
        """
        Lists the tuples of the glyceride to check if they're symmetrical

        Parameters: None

        Returns: tuple
        """
        fa1, fa2, fa3 = self.sn
        left, right = sorted((fa1, fa3), key=_fa_key)
        parts = []
        for fa in (left, fa2, right):
            if fa is None:
                parts.append(("EMPTY",))
            else:
                fac = fa.canonical()
                parts.append(
                    ("FA", fac.length, fac.db_positions, fac.db_stereo, fac.branches)
                )
        return tuple(parts)

    @property
    def name(self) -> str:
        """
        Generate a standardized name for the glyceride with sn-1 and sn-3
        ordered canonically so symmetric species share the same string.
        """
        fa1, fa2, fa3 = self.sn
        left, right = sorted((fa1, fa3), key=_fa_key)

        def fa_name(fa: Optional[FattyAcid]) -> str:
            if fa is None:
                return "EMPTY"
            # Keep your original fatty-acid naming style:
            parts = [f"N{fa.length:02d}", f"D{len(fa.db_positions):02d}"]
            if fa.db_positions:
                pos_stereo = []
                for p, s in zip(fa.db_positions, fa.db_stereo):
                    pos_stereo.append(f"P{p:02d}{s.upper()}")
                parts.extend(pos_stereo)
            for bpos, blabel in fa.branches:
                parts.append(f"M{bpos:02d}")  # Only 'Me' supported now
            return "".join(parts)

        return "G_" + "_".join([fa_name(left), fa_name(fa2), fa_name(right)])
