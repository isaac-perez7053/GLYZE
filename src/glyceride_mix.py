import numpy as np
from typing import Dict, Mapping, List, Tuple
from glyceride import Glyceride
import MDAnalysis as mda
import mdapackmol
import shutil
import os

AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1e-24


class GlycerideMix:
    """
    Represents the composition of glycerides in a mixture.

    Attributes:
        mix (Dict[Glyceride, float]): A dictionary mapping Glyceride objects to their quantities.
        units (str): The units for the quantities (default is "mole").
    """

    def __init__(self, mix: List[Tuple[Glyceride, float]], units: str = "mole"):
        # Turn mix into a dictionary
        self.mix = {i[0]: i[1] for i in mix}
        self.units = units

    def add(self, glyceride: Glyceride, quantity: float):
        """
        Add a glyceride and its quantity to the composition.

        Args:
            glyceride (Glyceride): The glyceride to add.
            quantity (float): The quantity of the glyceride.
        """
        if glyceride in self.mix:
            self.mix[glyceride] += quantity
        else:
            self.mix[glyceride] = quantity

    def total_quantity(self) -> float:
        """
        Calculate the total quantity of all glycerides in the composition.

        Returns:
            float: The total quantity.
        """
        return sum(self.mix.values())

    def build_simulation_box(
        self,
        num_molecules: int,
        density_g_per_cm3: float,
        min_dist: float = 2.0,
        seed: int | None = None,
    ) -> mda.Universe:
        """
        Pack a homogeneous mixture of triglycerides into a cubic box at a target density.

        Args:
            num_molecules: total molecules in the box (all species combined).
            density_g_per_cm3: target bulk density (e.g., 0.9 for many triglyceride oils).
            glyceride_to_universe: function that converts a Glyceride -> MDAnalysis.Universe
                                   (single molecule topology+coords).
            min_dist: Packmol 'tolerance' (angstroms)  minimum allowed interatomic distance.
            seed: random seed forwarded to Packmol for reproducibility.

        Returns:
            MDAnalysis.Universe for the packed system, with unit cell set.
        """
        if not self.mix:
            raise ValueError("GlycerideMix is empty.")

        total_qty = self.total_quantity()
        if total_qty <= 0:
            raise ValueError("Total quantity must be positive.")

        #  Determine integer counts by molar fractions
        mol_fractions = {g: qty / total_qty for g, qty in self.mix.items()}
        counts = self._integer_counts_from_fractions(mol_fractions, num_molecules)

        # Compute box length from target density:
        mass_g = 0.0
        for g, n in counts.items():
            MW = g.molar_mass
            mass_g += n * (MW / AVOGADRO)
        if mass_g <= 0:
            raise ValueError("Computed total mass is non-positive; check molar masses.")
        volume_cm3 = mass_g / float(density_g_per_cm3)
        volume_A3 = volume_cm3 / CM3_PER_A3
        L = float(volume_A3 ** (1.0 / 3.0))

        #  Build Packmol instructions for a cubic box [0, L]^3
        instructions = [f"inside box 0. 0. 0. {L:.6f} {L:.6f} {L:.6f}"]
        # Global options for Packmol (tolerance/min distance and seed)
        packmol_kwargs = {"tolerance": float(min_dist)}
        if seed is not None:
            packmol_kwargs["seed"] = int(seed)

        # Prepare species blocks
        species_blocks = []
        for g, n in counts.items():
            if n == 0:
                continue
            # Turn an rdkit object into a mda object
            u = mda.Universe(g.glyceride_to_rdkit(optimize=True))

            nres = len(u.residues)
            if not nres:
                raise ValueError(
                    "No residues present—cannot label. (Did the RDKit import create residues?)"
                )

            try:
                _ = u.residues.resnames
            except AttributeError:
                u.add_TopologyAttr("resnames", [""] * nres)

            u.residues.resnames = [g.name] * nres
            species_blocks.append(
                mdapackmol.PackmolStructure(u, number=int(n), instructions=instructions)
            )
        # Run Packmol. Returns an MDAnalysis.Universe with merged topology.
        try:
            system = mdapackmol.packmol(species_blocks, tolerance=float(min_dist))
        except ValueError as e:
            msg = str(e)
            if "STOP 173" in msg or "errorcode 173" in msg:
                # Rename Packmol's default output
                if os.path.exists("output.pdb_FORCED"):
                    shutil.move("output.pdb_FORCED", "output.pdb")
                # Load renamed file into MDAnalysis
                system = mda.Universe("output.pdb")
            else:
                raise
        else:
            # Success path (exit code 0): rename output if it exists
            if os.path.exists("output.pdb_FORCED"):
                shutil.move("output.pdb_FORCED", "output.pdb")

        system.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=float)

        # Check that the expected number of atoms is in the box
        expected_num_atoms = 0
        for g in self.mix.keys():
            num_atoms_gly = g.glyceride_to_rdkit().GetNumAtoms()
            expected_num_atoms += num_atoms_gly * counts[g]

        # Ensure number of atoms in the box was met
        if len(system.atoms) != expected_num_atoms:
            raise ValueError("Expected number of atoms in the box was not met")

        for seg in system.segments:
            tag = (seg.segid or "RES")[:3].upper()
        try:
            _ = seg.residues.resnames
        except AttributeError:
            system.add_TopologyAttr("resnames", [""] * len(system.residues))
        seg.residues.resnames = [tag] * len(seg.residues)

        return system

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

    def __repr__(self):
        parts = [f"{glyceride.name}: {qty}" for glyceride, qty in self.mix.items()]
        return "Glyceride_Composition({" + ", ".join(parts) + "})"
