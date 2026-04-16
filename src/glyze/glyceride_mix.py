from __future__ import annotations

import numpy as np
from typing import Dict, Mapping, List, Tuple, Union
from glyze.glyceride import Glyceride, FattyAcid
import MDAnalysis as mda
from glyze.packmol import PackmolSimulator
import shutil
import hashlib, re
from rdkit import Chem
from rdkit.Geometry import Point3D
from pathlib import Path
import csv
from collections import Counter


RESNAME_FORBIDDEN = {
    "SOL",
    "WAT",
    "HOH",
    "H2O",
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "ACE",
    "NME",
    "TIP",
    "ION",
}
BOHR_TO_ANG = 0.52917721092
AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1e-24


def _clean_name(name: str) -> str:
    """
    Clean a name string by removing non-alphanumeric characters and converting to uppercase.

    Parameters:
    -----------
    name (str) : 
        The input name string to be cleaned.

    Returns:
    --------
    str :
        The cleaned name string, containing only uppercase alphanumeric characters and spaces.
    """
    return re.sub(r"[^A-Za-z0-9]+", " ", str(name)).upper().strip()


def _sig_letters(name: str) -> str:
    """
    Extract significant letters from a name for resname generation. This is a heuristic to get meaningful initials.

    Parameters:
    -----------
    name (str) :
        The input name string from which to extract significant letters.

    Returns:
    --------
    str :
        A string of significant letters derived from the input name, intended for use in generating a resname.
    """
    toks = [t for t in _clean_name(name).split() if len(t) > 1]
    if not toks:
        toks = [_clean_name(name)]
    initials = "".join(t[0] for t in toks)
    tail = "".join(t[1:] for t in toks)
    return initials + tail


def _hash_letter(name: str, i: int) -> str:
    """
    Generate a deterministic letter based on the name and index, using a hash function to ensure uniqueness.

    Parameters:
    -----------
    name (str) :
        The input name string from which to generate a hash-based letter.
    i (int) :
        The index (0-based) of the letter to generate from the hash.

    Returns:
    --------    
    str :
        A single uppercase letter derived from the hash of the input name and index, intended for use in generating a resname.
    """
    h = hashlib.md5(name.encode()).hexdigest()
    idx = int(h[2 * i : 2 * i + 2], 16) % 26
    return chr(ord("A") + idx)


def make_resname(name: str, taken: set[str]) -> str:
    """
    Generate a unique 3-letter code for a glyceride based on its name, avoiding conflicts with taken codes and forbidden names.

    Parameters:
    -----------
    name (str) :
        The input name string from which to generate a resname.
    taken (set[str]) :
        A set of resnames that are already taken and should be avoided.
    
    Returns:
    --------
    str :
        A unique 3-letter resname derived from the input name, guaranteed not to be in the taken set or forbidden list.
    """
    sig = _sig_letters(name)
    cands = []
    if len(sig) >= 3:
        cands.append(sig[:3])
    for i in range(1, max(1, len(sig) - 2)):
        cands.append(sig[i : i + 3])
    canon = [re.sub(r"[^A-Z]", "", c).ljust(3, "X")[:3] for c in cands]

    h1, h2, h3 = _hash_letter(name, 0), _hash_letter(name, 1), _hash_letter(name, 2)
    canon += [
        (sig[0] if sig else "X") + (sig[1] if len(sig) > 1 else "X") + h1,
        (sig[0] if sig else "X") + h1 + h2,
        h1 + h2 + h3,
    ]
    for c in canon:
        if c and c not in taken and c not in RESNAME_FORBIDDEN:
            taken.add(c)
            return c
    for i in range(3, 10):
        c = "".join(_hash_letter(name, j) for j in range(i, i + 3))
        if c not in taken and c not in RESNAME_FORBIDDEN:
            taken.add(c)
            return c
    raise RuntimeError("Could not generate unique 3-letter code.")


def build_resname_map(glycerides_iter):
    """
    Build a mapping from glyceride objects to unique 3-letter resnames for use in PDB files.

    Parameters:
    -----------
    glycerides_iter (iterable):
        An iterable of glyceride objects for which to generate resnames.
    
    Returns:
    --------
    dict :
        A dictionary mapping each glyceride object to a unique 3-letter resname string.
    """
    taken, mapping = set(), {}
    for g in glycerides_iter:
        # use g.name if present; fall back to class name or repr
        base = getattr(g, "name", None) or g.__class__.__name__ or repr(g)
        mapping[g] = make_resname(base, taken)
    return mapping


def _as_pairs(mix) -> List[Tuple[MixtureComponent, float]]:
    """
    Accept dict-like or iterable of pairs.

    Parameters:
    -----------
    mix (dict or iterable of pairs):
        The input mixture specification, which can be either a dictionary mapping components to quantities or an iterable of (component, quantity) pairs.
    
    Returns:
    List[Tuple[MixtureComponent, float]] :
        A standardized list of (component, quantity) pairs extracted from the input mixture specification.
    """
    if hasattr(mix, "items"):
        return [(k, float(v)) for k, v in mix.items()]
    return [(k, float(v)) for (k, v) in mix]


def _canonical_key(comp: MixtureComponent) -> str:
    """
    Produce a stable canonical identifier. Ensure things are standardized

    Parameters:
    -----------
    comp (MixtureComponent):
        The mixture component for which to generate a canonical key.
    
    Returns:
    --------
    str :
        A canonical string identifier for the given mixture component, suitable for use as a dictionary key.
        This should be the same for components that are considered equivalent (e.g., same glyceride, mirrored).
    """
    if hasattr(comp, "canonical_id"):
        return comp.canonical_id()
    # fallback: class + repr (ok as a stopgap, but not ideal)
    return f"{comp.__class__.__name__}:{repr(comp)}"


def _canonical_component(comp: MixtureComponent) -> MixtureComponent:
    """
    Return a canonicalized object instance (e.g., sorted chains, standardized naming).
    If you already ensure objects are canonical upon construction, this can just return comp.

    Parameters:
    -----------
    comp (MixtureComponent):
        The mixture component to canonicalize.

    Returns:
    --------
    MixtureComponent :
        A canonicalized instance of the mixture component.
    """
    if hasattr(comp, "canonicalize"):
        return comp.canonicalize()
    return comp


def _clip_small_quantity(qty: float, atol: float = 1e-12) -> float:
    """
    Clip extremely small floating-point quantities to exactly zero.
    Any negative quantity is also clipped to zero.

    Parameters:
    -----------
    qty (float):
        The quantity to be clipped.
    atol (float):
        The absolute tolerance below which quantities are considered zero.

    Returns:
    --------
    float :
        The clipped quantity, which is zero if the input is negative or below the specified tolerance,
        and unchanged otherwise.
    """
    qty = float(qty)
    return 0.0 if qty < 0 or abs(qty) <= atol else qty


def water_vapor_pressure(T_K: float) -> float:
    """
    Water vapor pressure via the Antoine equation.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float :
        Vapor pressure in Pa.

    References
    ----------
    Antoine parameters from the NIST Chemistry WebBook (https://webbook.nist.gov)
    """
    T_C = T_K - 273.15
    if T_C < 100:
        A, B, C = 8.07131, 1730.63, 233.426
    else:
        A, B, C = 8.14019, 1810.94, 244.485
    p_mmHg = 10 ** (A - B / (T_C + C))
    return p_mmHg * 133.322  # mmHg -> Pa


def glycerol_vapor_pressure(T_K: float) -> float:
    """
    Glycerol vapor pressure via the Antoine equation.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float :
        Vapor pressure in Pa.
    """
    T_C = T_K - 273.15
    A, B, C = 12.89896, 6240.54, 331.126
    p_mmHg = 10 ** (A - B / (T_C + C))
    return p_mmHg * 133.322  # mmHg -> Pa


class MixtureComponent:
    """
    A wrapper class to represent a component in the glyceride mixture, which can be a Glyceride, FattyAcid, water, or glycerol.

    Attributes:
    -----------
    component (Union[Glyceride, FattyAcid, str]):
        The underlying component, which can be an instance of Glyceride, FattyAcid
        or a string representing water ("H2O") or glycerol ("Glycerol").
    
    Class Methods:
    --------------
    from_name :
        Create a MixtureComponent instance from a string name.
    from_string : 
        Create a MixtureComponent instance from a string representation.
    
    Methods:
    --------
    vapor_pressure :
        Calculate the vapor pressure of the component at a given temperature.

    Properties:
    name :
        Get the name of the component. 
    molar_mass :
        Get the molar mass of the component in g/mol.
    length :
        Get the chain length information of the component, if applicable.
    """

    def __init__(self, component):
        self.component = component

    @classmethod
    def from_name(cls, name: str):
        """
        Create a MixtureComponent instance from a string name. The method determines the type of component based on naming conventions:
        - If the name starts with "G_", it is treated as a Glyceride and parsed accordingly.
        - If the name starts with "N", it is treated as a FattyAcid and parsed accordingly.
        - Otherwise, it is treated as a raw string component (e.g., "H2O" or "Glycerol").

        Parameters
        ----------
        name (str) :
            The name of the component.

        Returns
        -------
        MixtureComponent :
            An instance of MixtureComponent representing the specified component.
        """
        if name.startswith("G_"):
            return cls(Glyceride.from_name(name))
        elif name.startswith("N"):
            return cls(FattyAcid.from_name(name))
        else:
            return cls(name)

    @classmethod
    def from_string(cls, string: str):
        """
        Create a MixtureComponent instance from a string representation. This is similar to from_name but includes validation for specific allowed formats.
        - If the string starts with "G_", it is treated as a Glyceride and parsed accordingly.
        - If the string starts with "N", it is treated as a FattyAcid and parsed accordingly.
        - If the string is "H2O", it is treated as water.
        - If the string is "Glycerol", it is treated as glycerol.
        - Otherwise, a TypeError is raised indicating an invalid component name.

        Parameters:
        -----------
        string (str) : 
            The string representation of the component.

        Returns:
        -------
        MixtureComponent :
            An instance of MixtureComponent representing the specified component.
        """
        if string.startswith("G_"):
            return cls(Glyceride.from_name(string))
        elif string.startswith("N"):
            return cls(FattyAcid.from_name(string))
        elif string == "H2O":
            return cls(string)
        else:
            raise TypeError(
                f"Please enter a valid component name (e.g., G_XXX, NXXX, H2O)"
            )

    def vapor_pressure(self, T: float):
        """
        Calculate the vapor pressure of the component at a given temperature T (in Kelvin).
        
        Parameters:
        -----------
        T (float) : 
            Temperature in Kelvin at which to calculate the vapor pressure.

        Returns:
        --------
        float :
            The vapor pressure of the component at the specified temperature.
        """
        if isinstance(self.component, Glyceride) and self.component.name == "Glycerol":
            return glycerol_vapor_pressure(T)
        vp = getattr(self.component, "vapor_pressure", None)
        if vp is None:
            if self.component == "H2O":
                return water_vapor_pressure(T)
            elif self.component == "Glycerol":
                return glycerol_vapor_pressure(T)
            else:
                raise ValueError(
                    f"Component {self.component} does not have a vapor pressure method."
                )
        return vp(T) if callable(vp) else vp

    @property
    def name(self):
        """The name of the component, if available, otherwise a string representation."""
        if hasattr(self.component, "name"):
            return self.component.name
        return str(self.component)

    @property
    def molar_mass(self):
        """The molar mass of the component in g/mol, if available, otherwise an error."""
        if hasattr(self.component, "molar_mass"):
            return self.component.molar_mass
        elif self.component == "H2O":
            return 18.01528
        else:
            raise ValueError(f"Unknown molar mass for component: {self.component}")

    @property
    def length(self):
        """
        The chain length information of the component, if applicable.
        - For glycerides, this returns a list of (chain_length, count) tuples.
        - For free fatty acids, it returns a single tuple with the number of carbons.
        - For water or glycerol, it returns an empty list.
        """
        if hasattr(self.component, "chain_legnths"):
            # Return a list like [9, 9, 8] to [(9, 2), (8, 1)]
            return list(Counter(self.component.chain_lengths).items())
        elif hasattr(self.component, "num_carbons"):
            return [(self.component.num_carbons, 1)]
        else:
            # Return null list if water or glycerol
            return []


class GlycerideMix:
    """
    A class that represents a mixture of free fatty acids (FFAs), and glycerides (TAGs, DAGs, MAGs) with their respective quantities.
    It provides methods to manipulate the mixture, calculate properties, and build a simulation box using Packmol.

    Attributes:
    -----------
    mix (Dict[MixtureComponent, float]) :
        A dictionary mapping each MixtureComponent to its quantity in the mixture.

    units (str) :
        The units of the quantities (e.g., "Moles", "Mass", "Volume").

    zero_tol (float) :
        A tolerance level for treating small quantities as zero to avoid floating-point issues.
    
    components (List[MixtureComponent]) :
        A list of all unique components in the mixture.
    
    quantities (List[float]) :
        A list of the quantities of each component in the mixture.
    
    glyceride_list (List[Glyceride]) :  
        A list of glyceride components in the mixture.
    
    fa_list (List[FattyAcid]) :
        A list of free fatty acid components in the mixture.
    
    glyceride_indices (List[int]) :
        A list of indices corresponding to glyceride components in the components list.
    
    fa_indices (List[int]) :
        A list of indices corresponding to free fatty acid components in the components list.
    
    mol_list (List[Chem.Mol]) :
        A list of RDKit Mol objects corresponding to the glyceride components, used for building Packmol templates.
    
    _mol_by_glyceride (Dict[Glyceride, Chem.Mol]) :
        A dictionary mapping each glyceride to its corresponding RDKit Mol object for quick access.
    
    index_by_key (Dict[str, int]) :
        A dictionary mapping the canonical key of each component to its index in the components list for efficient lookups.
    
    Class Methods:
    --------------
    from_csv : 
        import a mixture specification from a CSV file.

    Methods:
    --------
    change_qty : 
        change the quantity of a specific component in the mixture.
    
    add_species : 
        add a new component and its quantity to the mixture.
    
    total_quantity : 
        calculate the total quantity of all components in the mixture.
    
    build_simulation_box : 
        pack the mixture into a cubic box at a target density using Packmol, returning an MDAnalysis Universe.
    
    packmol_input_from_mix : 
        generate a Packmol input file content for the current mixture specification.
    
    to_csv : 
        export the mixture specification to a CSV file.

    Properties:
    -----------
        None

    """
    def __init__(
        self, mix, units: str = "Moles", *, sort: bool = True, zero_tol: float = 1e-12
    ):
        """
        Initialize the GlycerideMix with a mixture specification.

        Parameters:
        -----------
            mix (Dict[MixtureComponent, float]) :
                A dictionary mapping each MixtureComponent to its quantity in the mixture.
            units (str) :
                The units of the quantities (e.g., "Moles", "Mass", "Volume").
            sort (bool) :
                Whether to sort the components alphabetically.
            zero_tol (float) :
                A tolerance level for treating small quantities as zero to avoid floating-point issues.
        
        Returns:
        --------
            None
        """
        self.units = units
        self.zero_tol = float(zero_tol)

        # Take in a dictionary or iterable pair to create a list of tuples
        pairs = _as_pairs(mix)

        # Ensure every component is wrapped in MixtureComponent
        pairs = [
            (
                comp if isinstance(comp, MixtureComponent) else MixtureComponent(comp),
                qty,
            )
            for comp, qty in pairs
        ]

        merged_qty: Dict[str, float] = {}
        rep_obj: Dict[str, MixtureComponent] = {}

        # Separate information to ensure information stays ordered
        for comp, qty in pairs:
            comp_c = _canonical_component(comp)
            key = _canonical_key(comp_c)
            merged_qty[key] = _clip_small_quantity(
                merged_qty.get(key, 0.0) + float(qty), atol=self.zero_tol
            )
            rep_obj.setdefault(key, comp_c)

        keys = list(merged_qty.keys())
        if sort:
            keys.sort()

        keys = [k for k in keys if abs(merged_qty[k]) > self.zero_tol]

        self.components: List[MixtureComponent] = [rep_obj[k] for k in keys]
        self.quantities: List[float] = [merged_qty[k] for k in keys]

        # Build the mixture dictionary
        self.mix: Dict[MixtureComponent, float] = {
            rep_obj[k]: merged_qty[k] for k in keys
        }

        self.glyceride_list: List["Glyceride"] = []
        self.fa_list: List["FattyAcid"] = []

        self.glyceride_indices: List[int] = []
        self.fa_indices: List[int] = []

        for i, comp in enumerate(self.components):
            base = comp.component

            if isinstance(base, Glyceride):
                self.glyceride_indices.append(i)
                self.glyceride_list.append(base)

            elif isinstance(base, FattyAcid):
                self.fa_indices.append(i)
                self.fa_list.append(base)

        self.mol_list: List[Chem.Mol] = [
            g.to_rdkit_mol(optimize=True) for g in self.glyceride_list
        ]
        self._mol_by_glyceride: Dict["Glyceride", Chem.Mol] = dict(
            zip(self.glyceride_list, self.mol_list)
        )
        self.index_by_key: Dict[str, int] = {k: i for i, k in enumerate(keys)}

    @classmethod
    def from_csv(cls, csv_path: str):
        """
        Create an instance of GlycerideMix using a csv file

        Parameters:
        -----------
            csv_path (str) : The path to the csv file

        Returns:
        --------
            GlycerideMix :
                An instance of GlycerideMix initialized with the mixture specification from the CSV file.
        """
        mix = []
        units = None

        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = list(row.values())
                name = values[0].strip()
                qty = float(values[1])
                units = values[2].strip()
                mix.append((MixtureComponent.from_string(name), qty))

        if not mix:
            raise ValueError(f"CSV file '{csv_path}' is empty or has no data rows.")
        if units is None:
            raise ValueError(f"Could not determine units from '{csv_path}'.")

        return cls(mix, units)

    def change_qty(self, component: MixtureComponent, new_quantity: float):
        """
        Change the quantity of a component in the mixture.

        Parameters:
        -----------
            component (MixtureComponent) : 
                The component whose quantity is to be changed.
            new_quantity (float) : 
                The new quantity for the component.

        Returns:
        --------
            None
        """
        comp_c = _canonical_component(component)
        key = _canonical_key(comp_c)
        if key not in self.index_by_key:
            raise ValueError(f"Component {component} not in mixture")
        i = self.index_by_key[key]
        clipped_quantity = _clip_small_quantity(new_quantity, atol=self.zero_tol)
        self.quantities[i] = clipped_quantity
        self.mix[self.components[i]] = clipped_quantity

    def add_species(self, species_conc: Tuple[MixtureComponent, float]):
        """
        Add a new species to the mixture.

        Parameters:
        -----------
            species_conc (Tuple[MixtureComponent, float]) :
                The component and its quantity to be added.
        
        Returns:
        --------
            None
        """
        component, quantity = species_conc
        comp_c = _canonical_component(component)
        key = _canonical_key(comp_c)
        if key in self.index_by_key:
            raise ValueError(f"Component {component} already in mixture")
        clipped_quantity = _clip_small_quantity(quantity, atol=self.zero_tol)
        self.components.append(component)
        self.quantities.append(clipped_quantity)
        self.mix[component] = clipped_quantity
        self.index_by_key[key] = len(self.components) - 1

    @staticmethod
    def _integer_counts_from_fractions(
        fracs: Mapping[Glyceride, float], N: int
    ) -> Dict[Glyceride, int]:
        """
        Round fractional allocations to integers while preserving the total N.

        Parameters:
        -----------
        fracs (Mapping[Glyceride, float]) :
            A mapping of glycerides to their fractional allocations (summing to 1).
        N (int) :
            The total number of molecules to allocate.

        Returns:
        --------
        Dict[Glyceride, int] :
            A mapping of glycerides to their integer counts, summing to N.
        """
        raw = {g: fracs[g] * N for g in fracs}
        floors = {g: int(np.floor(raw[g])) for g in fracs}
        deficit = N - sum(floors.values())
        # Distribute remaining molecules to the largest fractional remainders
        remainders = sorted(((raw[g] - floors[g], g) for g in fracs), reverse=True)
        for i in range(deficit):
            _, g = remainders[i]
            floors[g] += 1
        return floors

    def _update_single_mol_from_pdb(
        self,
        mol: Chem.Mol,
        pdb_path: str | Path,
        conf_id: int = -1,
    ) -> Chem.Mol:
        """
        Internal helper: update coordinates of an existing RDKit Mol from a PDB file.

        Parameters
        ----------
        mol (Chem.Mol) :
            Target molecule whose coordinates will be updated in-place.
        pdb_path (str or Path) :
            Path to a single-molecule PDB file with matching atom order.
        conf_id (int) :
            Conformer ID to update (default -1 = last conformer).

        Returns
        -------
        Chem.Mol :
            The same Mol object with updated coordinates.
        """
        pdb_path = str(Path(pdb_path).resolve())

        # Read PDB as RDKit Mol (no need to sanitize if we only want coords)
        pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if pdb_mol is None:
            raise ValueError(f"Failed to read PDB file: {pdb_path}")

        if pdb_mol.GetNumAtoms() != mol.GetNumAtoms():
            raise ValueError(
                f"Atom count mismatch between target mol ({mol.GetNumAtoms()}) "
                f"and PDB ({pdb_mol.GetNumAtoms()}) for file {pdb_path}"
            )

        if pdb_mol.GetNumConformers() == 0:
            raise ValueError(f"PDB mol has no conformers: {pdb_path}")
        if mol.GetNumConformers() == 0:
            # If mol has no conformer yet, copy one from PDB
            new_conf_id = mol.AddConformer(pdb_mol.GetConformer(0), assignId=True)
            return mol  # all coords already set

        # Otherwise, overwrite the specified conformer
        pdb_conf = pdb_mol.GetConformer(0)
        try:
            conf = mol.GetConformer(conf_id)
        except ValueError:
            # If requested conf_id doesn't exist, fall back to last
            conf = mol.GetConformer(mol.GetNumConformers() - 1)

        for i in range(mol.GetNumAtoms()):
            pos = pdb_conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))

        return mol

    def qty_for(self, comp: MixtureComponent) -> float:
        """Quantity lookup after canonicalization.
        Parameters
        ----------
        comp (MixtureComponent) : 
            The component for which to look up the quantity. This will be canonicalized before lookup.
            
        Returns:
        -------
            float  :
                The quantity of the specified component in the mixture, or zero if not present.
        """
        comp_c = _canonical_component(comp)
        key = _canonical_key(comp_c)
        i = self.index_by_key[key]
        return _clip_small_quantity(self.quantities[i], atol=self.zero_tol)

    def update_mols_from_pdbs(
        self,
        pdb_files: List[str | Path],
        conf_id: int = -1,
    ) -> None:
        """
        Update all stored Chem.Mol templates with new coordinates from PDB files.

        Parameters
        ----------
        pdb_files : list of str or Path
            List of PDB file paths. The i-th PDB is assumed to correspond to the
            i-th glyceride in `self.glyceride_list` / `self.mol_list`, and the
            atom ordering must match the existing RDKit Mol.
        conf_id : int
            Conformer ID to update on each Mol (default -1 = last conformer).

        Returns:
        -------
        None.

        Notes
        -----
        - This updates both `self.mol_list[i]` and the corresponding entry
          in `self._mol_by_glyceride` so that all later operations (e.g., building
          Packmol templates) use the new coordinates.
        """
        if len(pdb_files) != len(self.mol_list):
            raise ValueError(
                f"Length mismatch: got {len(pdb_files)} PDB files but "
                f"{len(self.mol_list)} stored molecules."
            )

        for i, (g, pdb_path) in enumerate(zip(self.glyceride_list, pdb_files)):
            mol = self._mol_by_glyceride.get(g, self.mol_list[i])

            updated = self._update_single_mol_from_pdb(
                mol=mol,
                pdb_path=pdb_path,
                conf_id=conf_id,
            )

            # Keep references consistent
            self.mol_list[i] = updated
            self._mol_by_glyceride[g] = updated

    def total_quantity(self) -> float:
        """
        Calculate the total quantity of all glycerides in the mix.
        
        Parameters:
        -----------
            None
            
        Returns:
        --------
        float :
            The total quantity of all components in the mixture, with small values clipped to zero.
        """
        return _clip_small_quantity(sum(self.quantities), atol=self.zero_tol)

    def build_simulation_box(
        self,
        num_molecules: int,
        density_g_per_cm3: float,
        min_dist: float = 2.0,
        seed: int | None = None,
        nloop: int | None = None,
        resname_map: Dict["Glyceride", str] | None = None,
    ) -> mda.Universe:
        """
        Pack a homogeneous mixture of triglycerides into a cubic box at a target density.
        Returns an MDAnalysis.Universe; coordinates in angstroms and unit cell set.

        This version uses PackmolSimulator to:
        - compute the cubic box length from counts + molar masses and density
        - build the Packmol input file
        - execute Packmol directly (handling error 173 gracefully).

        Parameters:
        -----------
        num_molecules (int) :
            The total number of molecules to pack in the box.
        density_g_per_cm3 (float) :
            The target density of the packed box in g/cm^3.
        min_dist (float) :
            The minimum distance between any two atoms in the packed box, in angstroms.
        seed (int or None) :
            An optional random seed for Packmol's random number generator to ensure reproducibility.
        nloop (int or None) :
            An optional parameter to specify the number of iterations for Packmol's optimization loop.
        resname_map (dict or None) :
            An optional mapping from Glyceride objects to 3-letter residue names for the PDB
            templates. If None, a mapping will be generated automatically. If provided, it must include
            entries for all glycerides in the mixture.

        Returns:
        --------
        mda.Universe :
            An MDAnalysis Universe containing the packed box with coordinates in angstroms and unit cell set.
        """
        if not self.mix:
            raise ValueError("GlycerideMix is empty.")

        # self.mix is assumed to be an iterable of (Glyceride, qty: float)
        total_qty = self.total_quantity()
        if total_qty <= 0:
            raise ValueError("Total quantity must be positive.")

        # Stable ordering of glycerides
        mol_fractions = {g: qty / total_qty for g, qty in self.mix.items()}

        # Reuse your existing integerization logic if you have it
        counts_dict = self._integer_counts_from_fractions(mol_fractions, num_molecules)

        # Get aligned counts + molar masses for PackmolSimulator
        counts_list = [int(counts_dict[g]) for g in self.glyceride_list]
        molar_masses = [g.molar_mass for g in self.glyceride_list]  # g/mol

        # Quick sanity check
        if sum(counts_list) != num_molecules:
            raise RuntimeError(
                f"Integerized counts sum to {sum(counts_list)} != num_molecules={num_molecules}."
            )

        # Try to reuse an existing simulator on self if present, otherwise make one.
        packmol_sim = getattr(self, "_packmol_sim", None)
        if packmol_sim is None:
            # This assumes 'packmol' is on PATH; customize if you want a fixed executable path.
            packmol_sim = PackmolSimulator()
            # Optionally stash it for future calls
            # self._packmol_sim = packmol_sim

        L = packmol_sim.estimate_cubic_box_length_from_species(
            counts=counts_list,
            molar_masses_g_per_mol=molar_masses,
            density_g_per_cm3=float(density_g_per_cm3),
        )

        if resname_map is None:
            resname_map = build_resname_map(self.glyceride_list)
        else:
            missing = [g for g in self.glyceride_list if g not in resname_map]
            if missing:
                raise ValueError(
                    f"Missing residue names for glycerides: {missing}. "
                    "Provide a complete resname_map."
                )

        # These will be the 'structure' entries in the Packmol input file.
        template_pdbs: Dict["Glyceride", str] = {}

        for g in self.glyceride_list:
            n = counts_dict[g]
            if n == 0:
                continue

            # Use stored RDKit mol if available; otherwise generate + cache
            try:
                mol = self._mol_by_glyceride[g]
            except KeyError:
                mol = g.to_rdkit_mol(optimize=True)
                self._mol_by_glyceride[g] = mol

            resname = resname_map[g]
            template_pdb = f"packmol_template_{resname}.pdb"
            template_pdbs[g] = template_pdb

            # You already have this utility in your Glyceride class
            Glyceride.mol_to_pdb(mol, template_pdb)

        # Filter out species with zero count for Packmol input
        structure_files = []
        structure_counts = []
        for g in self.glyceride_list:
            n = counts_dict[g]
            if n <= 0:
                continue
            structure_files.append(template_pdbs[g])
            structure_counts.append(int(n))

        if not structure_files:
            raise ValueError("All species ended up with zero count; nothing to pack.")

        extra_lines = []
        if seed is not None:
            extra_lines.append(f"seed {int(seed)}")
        if nloop is not None:
            extra_lines.append(f"nloop {int(nloop)}")

        input_filename = "packmol.inp"
        output_filename = "output.pdb"

        input_path = packmol_sim.build_input_file(
            structure_files=structure_files,
            counts=structure_counts,
            box_lengths=L,  # cubic box
            tolerance=float(min_dist),
            filetype="pdb",
            output=output_filename,
            input_filename=input_filename,
            extra_lines=extra_lines or None,
            workdir=None,  # current working directory
        )
        try:
            result = packmol_sim.execute_packmol(
                input_file=input_path,
                use_instance_path=True,
                cwd=None,
                capture_output=True,
                check=True,
                timeout=None,
                env=None,
                treat_173_as_success=True,
            )
            # You can inspect result.stdout / result.stderr for logging if desired
        except Exception as e:
            # Optional: surface Packmol output in the error message
            raise RuntimeError(f"Packmol run failed: {e}") from e

        out_path = Path(output_filename).resolve()
        forced_path = Path(f"{output_filename}_FORCED").resolve()

        if not out_path.exists() and forced_path.exists():
            shutil.move(forced_path, out_path)

        if not out_path.exists():
            raise FileNotFoundError(
                f"Packmol did not produce '{output_filename}' or '{output_filename}_FORCED'."
            )

        md_system = mda.Universe(str(out_path))

        expected_num_atoms = 0
        for g, n in counts_dict.items():
            if n <= 0:
                continue
            num_atoms_gly = self._mol_by_glyceride[g].GetNumAtoms()
            expected_num_atoms += num_atoms_gly * int(n)

        if len(md_system.atoms) != expected_num_atoms:
            raise ValueError(
                f"Expected {expected_num_atoms} atoms but found {len(md_system.atoms)} "
                f"in packed system."
            )

        md_system.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=float)

        print("Successfully ran Packmol.")
        return md_system

    def packmol_input_from_mix(
        items: List[Dict[str, Union[str, int]]],
        box: Tuple[float, float, float],
        output_pdb="mixture.pdb",
        tolerance=2.0,
    ) -> str:
        """
        Generate a Packmol input file content for the given mixture.

        Parameters
        ----------
        items : list of dict
            Each dict should have keys: 'pdb' (str), 'count' (int
            'resname' (str).
        box : tuple of float
            Box dimensions (Lx, Ly, Lz).
        output_pdb : str
            Output PDB filename.
        tolerance : float
            Minimum distance tolerance for Packmol.

        Returns:
        -------
        str :
            The content of the Packmol input file.
        """
        Lx, Ly, Lz = box
        lines = [f"tolerance {tolerance}", "filetype pdb", f"output {output_pdb}", ""]
        for it in items:
            lines += [
                f"structure {it['pdb']}",
                f"  number {it['count']}",
                f"  resnumbers 1",
                f"  resname {it['resname']}",
                f"  inside box 0. 0. 0. {Lx} {Ly} {Lz}",
                f"end structure",
                "",
            ]
        return "\n".join(lines)

    def to_csv(self, output_path: str):
        """
        Create a csv containing the species names and concetrations

        Paramters:
        ----------
            output_path (str): The path to the written csv file

        Returns:
        --------
            None
        """
        # Create the headers of the csv file
        headers = ["Species", "Concentration", "Units"]

        # Write into the csv file
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            # Make sure to print units
            for key, value in self.mix.items():
                row = (
                    [key.name if isinstance(key, (Glyceride, FattyAcid)) else key]
                    + [value]
                    + [self.units]
                )
                writer.writerow(row)

        print(f"CSV file {output_path} created successfully")

    @property
    def name(self) -> str:
        """
        Generate a standardized name for the glyceride mix using the format:

        N{CC}D{DD}[P{pp}{S}{pp}{S}...][M{pp}...][OH{pp}...]

        N{CC} - Number of Carbons like 06 or 12
        D{DD} - Number of double bonds
        [P{pp}(S){pp{S}...] - double bond position and stereo where S is trans and Z is cis (e.g. 06Z )
        M{pp}... -  Methyl branches at position pp
        OH{pp}... - Hydroxyl branches at position pp.

        Example: N18D1P09Z means 18 carbons, 1 double bond at position 9 with Z (cis) stereo and is oleic acid.

        To represent a glyceride, the following format is used:

        G_fa1_fa2_fa3

        where fa1, fa2, and fa3 are the fatty acids attached to the glycerol backbone with the naming convention above.

        Finally, to represent a mixture of glycerides, the following format is used:

        MIX_G_fa1_fa2_fa3-qty_G_fa1_fa2_fa3-qty...

        where qty is the quantity up to three significant figures of each glyceride in the mixture.
        """
        parts = [f"{glyceride.name}-{qty:.3g}" for glyceride, qty in self.mix.items()]
        return "MIX_" + "_".join(parts)

    @property
    def list_masses(self) -> List[float]:
        """
        List of species masses using units
        """
        masses = []
        if self.units == "Moles":
            masses = [g.molar_mass * qty for g, qty in self.mix.items()]

        if self.units == "Grams":
            masses = [qty for g, qty in self.mix.items()]

        if self.units == "Mass Fractions":
            masses = [qty * self.total_mass for g, qty in self.mix.items()]

        return masses

    @property
    def total_mass(self) -> float:
        """
        Calculate total species mass using units
        """

        if self.units == "Moles":
            total = sum(g.molar_mass * qty for g, qty in self.mix.items())

        if self.units == "Grams":
            total = sum(qty for _, qty in self.mix.items())

        if self.units == "Mass Fractions":
            total = 1.0

        return total

    def __repr__(self):
        """
        Return a string representation of the GlycerideMix instance, showing each glyceride and its quantity in the mixture.
        """
        parts = [f"{glyceride.name}: {qty}" for glyceride, qty in self.mix.items()]
        return "Glyceride_Composition({" + ", ".join(parts) + "})"

    def __str__(self):
        """
        Return a tabulated string representation of the GlycerideMix instance, showing each glyceride and its quantity in the mixture in a table format.
        """
        import tabulate

        table = [[glyceride.name, qty] for glyceride, qty in self.mix.items()]
        return tabulate.tabulate(
            table, headers=["Glyceride", f"Quantity ({self.units})"]
        )
