from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from pathlib import Path
import numpy as np

GLYCEROL_MOL_MASS = 92.09382 # g/mol for C3H8O3

def _optimize_mol(mol: Chem.Mol, confId: int) -> Chem.Mol:
    """Optimize the 3D structure of an RDKit molecule with ETKDG v2 and force fields."""

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

#TODO: Fix the docstrings in the Glyceride and FattyAcid class
@dataclass(frozen=True)
class FattyAcid:
    """
    Immutable description of a fatty acid chain.

    Attributes
    ----------

    Functions
    ---------
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
        for pos, _label in self.branches:
            if not (1 <= pos <= self.length):
                raise ValueError(
                    f"branch position C{pos} out of range for C{self.length}"
                )
            #TODO: Ensure the branch is valid 

    @classmethod
    def from_name(cls, fa_str: str) -> Optional[FattyAcid]:
        """
        Return the fatty acid using its unique name
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
        return FattyAcid(
            length, tuple(db_positions), tuple(db_stereo), tuple(branches)
        )


    def canonical(self) -> "FattyAcid":
        """
        Returns a canonical (normalized) version of the fatty acid. Ensures that
        object signature is identical for equivalent fatty acids.
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

    def to_rdkit_mol(self, optimize: bool = False) -> Chem.Mol:
        """
        Convert the fatty acid to an RDKit molecule.

        Returns:
            Chem.Mol: The RDKit molecule representing the fatty acid.
        """
        rw = Chem.RWMol()

        # add the carboxyl group
        carboxyl = rw.AddAtom(Chem.Atom(6)) #the atoms
        o_double = rw.AddAtom(Chem.Atom(8))
        o_single = rw.AddAtom(Chem.Atom(8))

        rw.AddBond(carboxyl, o_double, Chem.BondType.DOUBLE)
        rw.AddBond(carboxyl, o_single, Chem.BondType.SINGLE)

        # Build the rest of the chain (C1...Cn)
        chain_idx = []
        last = carboxyl #adds to the chain
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
            b_neighbors = [nbr.GetIdx()
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
    
    # find P at a specific temperature using clausis-clapporon (T2 = 298.15K)
    def vapor_pressure_temp(self, T)-> float:
        return self.vapor_pressure*np.exp((self.enthalpy_of_vaporization/0.008314462618)*((1/T)-(1/298.15)))


    @property
    def molar_mass(self) -> float:
        """Calculate the molar mass of the fatty acid in g/mol"""
        # Build RDkit molecule and sum atomic masses
        mol = self.to_rdkit_mol()
        mass = 0
        for atom in mol.GetAtoms():
            mass += atom.GetMass()

        return mass

    @property 
    def num_carbons(self) -> int:
        """Return the number of carbons in the fatty acid"""
        return self.length
    # need delta H first // in Kj mol^-1 
    # TODO: add citation
    @property
    def enthalpy_of_vaporization(self) -> float:
        if len(self.db_positions) == 0:
            return 5.36*self.num_carbons + 37.1 
        else:
            return 5.91*self.num_carbons + 26.4
        
    # ln(p/p0) where p0 = 101325 pa
    @property
    def ln_vapor_pressure(self)-> float:
        return -1.01*self.num_carbons - 3.2
    
    # need to find p
    @property
    def vapor_pressure(self)-> float:
        return 101325*np.exp(self.num_carbons)
    

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
    


class Glyceride:
    """
    Description of a glyceride (diacylglyceride if a chain is None)

    sn: tuple of three Optional[FattyAcid] in sn-1, sn-2, sn-3 order.
        Use None for an emtpy chain (e.g. diacylglyceride embedding).
    """

    def __init__(
        self, sn: Tuple[Optional[FattyAcid], Optional[FattyAcid], Optional[FattyAcid]]
    ):
        self.sn = sn
        if len(self.sn) != 3:
            raise ValueError("sn must have length 3 (sn-1, sn-2, sn-3)")

    # TODO: Validate from_name and make sure to canonicalize stereochemical names in name
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
        if not name.startswith("G_"):
            raise ValueError(f"Invalid glyceride format: {name}")
        
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
            index (int): Index (0, 1, or 2) to add the fatty acid to.
            fatty_acid (FattyAcid): The fatty acid to add.
            deep_copy (bool): Whether to perform a deep copy of the glyceride.

        Returns:
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

    def to_rdkit_mol(self, optimize: bool = True) -> Chem.Mol:
        """
        Build an RDKit molecule for the given Glyceride, embed in 3D, and relax.
        Uses kwargs-only ETKDG (ETversion=2) for compatibility with your RDKit.

        Args:
            optimize (bool): Whether to optimize the 3D structure with force fields.
                If False, only embedding is done.

        Returns:
            Chem.Mol: The RDKit molecule with 3D coordinates.
        """
        rw, sn_os, _ = self._build_glycerol_backbone()  # the chem mol func and the indices of the backbone
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

    # find vapor pressure based on temperature
    def vapor_pressure(self, T) -> float:
        return np.exp((-self.gibbs_of_vaporitzation/(0.008314*298.15*np.log(10)))+((self.enthalpy_of_vaporitzation/(0.008314*np.log(10)))*((1/298.15)-(1/T))))

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

    @property
    def molar_mass(self) -> float:
        """Calculate the molar mass of a glyceride in g/mol"""
        mol = self.to_rdkit_mol()
        mass = 0
        for atom in mol.GetAtoms():
            mass += atom.GetMass()
        return mass

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
        delta_h_mag = 41.73
        delta_h_dag = 3.486
        delta_h_tag = -34.76
        fa = [x for x in self.sn if x]
        # mag
        if len(fa) == 1:
            return delta_h_mag + fa[0].num_carbons*2.09 + 31.4
        # dag
        elif len(fa) == 2:
            return delta_h_dag + (fa[0].num_carbons*2.09 + 31.4) + (fa[1].num_carbons*2.09 + 31.4)
        # tag
        else:
            return delta_h_tag + (fa[0].num_carbons*2.09 + 31.4) + (fa[1].num_carbons*2.09 + 31.4) + (fa[2].num_carbons*2.09 + 31.4)

    # gibbs free energy of vaporization for mags, dags, tags
    @property
    def gibbs_of_vaporitzation(self) -> float:
        delta_g_mag = -19.86
        delta_g_dag = -46.87
        delta_g_tag = -73.88
        fa = [x for x in self.sn if x]
        # mag
        if len(fa) == 1:
            return delta_g_mag + fa[0].num_carbons*1.66 + 22
        # dag
        elif len(fa) == 2:
            return delta_g_dag + (fa[0].num_carbons*1.66 + 22) + (fa[1].num_carbons*1.66 + 22)
        # tag
        else:
            return delta_g_tag + (fa[0].num_carbons*1.66 + 22) + (fa[1].num_carbons*1.66 + 22) + (fa[2].num_carbons*1.66 + 22)

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
    """Glyceride where sn-1 and sn-3 are considered equivalent for equality/hash."""

    def signature_tuple(self) -> tuple:
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
