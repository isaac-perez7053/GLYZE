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
    return re.sub(r"[^A-Za-z0-9]+", " ", str(name)).upper().strip()


def _sig_letters(name: str) -> str:
    toks = [t for t in _clean_name(name).split() if len(t) > 1]
    if not toks:
        toks = [_clean_name(name)]
    initials = "".join(t[0] for t in toks)
    tail = "".join(t[1:] for t in toks)
    return initials + tail


def _hash_letter(name: str, i: int) -> str:
    h = hashlib.md5(name.encode()).hexdigest()
    idx = int(h[2 * i : 2 * i + 2], 16) % 26
    return chr(ord("A") + idx)


def make_resname(name: str, taken: set[str]) -> str:
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
    taken, mapping = set(), {}
    for g in glycerides_iter:
        # use g.name if present; fall back to class name or repr
        base = getattr(g, "name", None) or g.__class__.__name__ or repr(g)
        mapping[g] = make_resname(base, taken)
    return mapping


def _as_pairs(mix) -> List[Tuple[MixtureComponent, float]]:
    """Accept dict-like or iterable of pairs."""
    if hasattr(mix, "items"):
        return [(k, float(v)) for k, v in mix.items()]
    return [(k, float(v)) for (k, v) in mix]


def _canonical_key(comp: MixtureComponent) -> str:
    """
    Produce a stable canonical identifier. Ensure things are standardized
    """
    if hasattr(comp, "canonical_id"):
        return comp.canonical_id()
    # fallback: class + repr (ok as a stopgap, but not ideal)
    return f"{comp.__class__.__name__}:{repr(comp)}"


def _canonical_component(comp: MixtureComponent) -> MixtureComponent:
    """
    Return a canonicalized object instance (e.g., sorted chains, standardized naming).
    If you already ensure objects are canonical upon construction, this can just return comp.
    """
    if hasattr(comp, "canonicalize"):
        return comp.canonicalize()
    return comp

def water_vapor_pressure(T: float, P: float=1.0):
    """
    Calculate water vapor pressure in mmHg using Antoine equation.
    T is temperature in Celsius.
    P is pressure in atm (default 1.0 atm).
    """
    # Antoine constants for water (from NIST)
    A, B, C = 8.07131, 1730.63, 233.426
    # Calculate vapor pressure in mmHg
    log_p = A - B / (T + C)
    p_mmHg = 10 ** log_p
    # Convert to atm if needed
    return p_mmHg / 760.0 if P == 1.0 else p_mmHg

def glycerol_vapor_pressure(T: float, P:float=1.0):
    """
    Calculate glycerol vapor pressure in mmHg using Antoine equation.
    T is temperature in Celsius.
    P is pressure in atm (default 1.0 atm).
    """
    # Antoine constants for glycerol (from NIST)
    A, B, C = 7.28542, 1385.03, 220.79
    # Calculate vapor pressure in mmHg
    log_p = A - B / (T + C)
    p_mmHg = 10 ** log_p
    # Convert to atm if needed
    return p_mmHg / 760.0 if P == 1.0 else p_mmHg

class MixtureComponent: 
    """"""

    def __init__(self, component):
        self.component = component

    @classmethod 
    def from_string(cls, string: str): 
        """"""
        if string.startswith("G_"):
            return cls(Glyceride.from_name(string))
        elif string.startswith("N"):
            return cls(FattyAcid.from_name(string))
        elif string == 'H2O' or string == 'Glycerol':
            return cls(string)
        else:
            raise TypeError(f"Please enter a valid component name (e.g., G_XXX, NXXX, H2O, Glycerol)")
        
    def vapor_pressure(self, T: float, P: float = 1.0):
        if hasattr(self.component, "vapor_pressure"):
            return self.component.vapor_pressure(T, P)
        elif self.component == "H2O":
            return water_vapor_pressure(T, P)
        else:
            return glycerol_vapor_pressure(T, P)
        
    @property
    def name(self):
        if hasattr(self.component, "name"):
            return self.component.name
        return str(self.component)



class GlycerideMix:
    def __init__(self, mix, units: str = "mole", *, sort: bool = True):
        self.units = units

        # Take in a dictionary or iterable pair to create a list of tuples
        pairs = _as_pairs(mix)
        merged_qty: Dict[str, float] = {}
        rep_obj: Dict[str, MixtureComponent] = {}

        # Separate information to ensure information stays ordered
        for comp, qty in pairs:
            comp_c = _canonical_component(comp)
            key = _canonical_key(comp_c)
            merged_qty[key] = merged_qty.get(key, 0.0) + float(qty)
            rep_obj.setdefault(key, comp_c)

        keys = list(merged_qty.keys())
        if sort:
            keys.sort()

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
            if isinstance(comp, Glyceride):
                self.glyceride_indices.append(i)
                self.glyceride_list.append(comp)
            elif isinstance(comp, FattyAcid):
                self.fa_indices.append(i)
                self.fa_list.append(comp)

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
            csv_path (str): The path to the

        Returns:
        --------
            GlycerideMix
        """
        data = []
        try:
            # Open file
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                # DictReader uses the header row for keys
                reader = csv.DictReader(f)
                for row in reader:
                    values = list(row.values())
                    # Extract H20 and Glycerol from the mix first if found
                    if values[0] == "H2O" or values[0] == "Glycerol":
                        data.append(((values[0], values[1]), values[2]))
                    else:
                        data.append(
                            (
                                (
                                    (
                                        Glyceride.from_name(values[0])
                                        if values[0].startswith("G_")
                                        else FattyAcid.from_name(values[0])
                                    ),
                                    values[1],
                                ),
                                values[2],
                            )
                        )

        except FileNotFoundError:
            print(f"Error: The file {csv_path} was not found.")
        mix = [x[0] for x in data]
        return cls(mix, data[0][1])

    def change_qty(self, component: MixtureComponent, new_quantity: float):
        """
        Change the quantity of a component in the mixture.
        
        """
        comp_c = _canonical_component(component)
        key = _canonical_key(comp_c)
        if key not in self.index_by_key:
            raise ValueError(f"Component {component} not in mixture")
        i = self.index_by_key[key]
        self.quantities[i] = new_quantity

    def add_species(self, species_conc: Tuple[MixtureComponent, float]):
        """
        Add a new species to the mixture.

        Parameters:
        -----------
            species_conc (Tuple[MixtureComponent, float]): The component and its quantity to be added.
        """
        component, quantity = species_conc
        comp_c = _canonical_component(component)
        key = _canonical_key(comp_c)
        if key in self.index_by_key:
            raise ValueError(f"Component {component} already in mixture")
        self.components.append(component)
        self.quantities.append(quantity)
        self.index_by_key[key] = len(self.components) - 1

    @staticmethod
    def _integer_counts_from_fractions(
        fracs: Mapping[Glyceride, float], N: int
    ) -> Dict[Glyceride, int]:
        """Round fractional allocations to integers while preserving the total N."""
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
        mol : Chem.Mol
            Target molecule whose coordinates will be updated in-place.
        pdb_path : str or Path
            Path to a single-molecule PDB file with matching atom order.
        conf_id : int
            Conformer ID to update (default -1 = last conformer).

        Returns
        -------
        Chem.Mol
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
        """Quantity lookup after canonicalization."""
        comp_c = _canonical_component(comp)
        key = _canonical_key(comp_c)
        i = self.index_by_key[key]
        return self.quantities[i]

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
        """Calculate the total quantity of all glycerides in the mix."""
        return sum(self.mix.values())

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

        Returns
        -------
        str
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
        if self.units == "mole":
            masses = [
                92.08 if g == "Glycerol" else g.molar_mass * qty
                for g, qty in self.mix.items()
            ]

        if self.units == "gram":
            masses = [qty for g, qty in self.mix.items()]

        if self.units == "mass_fraction":
            masses = [qty * self.total_mass for g, qty in self.mix.items()]

        return masses

    @property
    def total_mass(self) -> float:
        """
        Calculate total species mass using units
        """

        if self.units == "mole":
            total = sum(
                92.08 if g == "Glycerol" else g.molar_mass * qty
                for g, qty in self.mix.items()
            )

        if self.units == "gram":
            total = sum(qty for _, qty in self.mix.items())

        if self.units == "mass_fraction":
            total = 1.0

        return total

    def __repr__(self):
        parts = [f"{glyceride.name}: {qty}" for glyceride, qty in self.mix.items()]
        return "Glyceride_Composition({" + ", ".join(parts) + "})"

    def __str__(self):
        import tabulate

        table = [[glyceride, qty] for glyceride, qty in self.mix.items()]
        return tabulate.tabulate(
            table, headers=["Glyceride", f"Quantity ({self.units})"]
        )
