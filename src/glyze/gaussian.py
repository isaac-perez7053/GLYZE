from __future__ import annotations

from ase.io import write, read
from qcforever.gaussian_run import GaussianRunPack
import MDAnalysis as mda
import io
from rdkit import Chem


class GaussianSimulation:
    """
    Class associated with running and handling output from the Gaussian
    software.
    """

    def __init__(self, mol: Chem.Mol):
        self.mol = mol

    def fchk_to_mda(self, fchk: str) -> mda.Universe:
        """
        Convert an fchk file to a pdb

        Parameters:

        Returns:

        """
        atoms = read(fchk, format="gaussian_fchk")
        # Write pdb file into memory
        pdb_file = io.StringIO()
        write(pdb_file, atoms)
        return mda.Universe(pdb_file)

    def optimize(
        self,
        functional: str = "B3LYP-D3",
        basis_set: str = "6-311+G*",
        ncore: int = 4,
        option: str = "opt energy",
        infilename="gaussian_opt.gjf",
        restart: bool = False,
        mem: str = "20GB",
        timexe: int = 17280,
    ) -> dict:
        """
        Optimize molecular geometry

        Parameters:

        Returns:

        """
        gsim = GaussianRunPack.GaussianDFTRun(
            functional=functional,
            basis=basis_set,
            nproc=ncore,
            value=option,
            in_file=infilename,
            restart=restart,
        )
        gsim.mem = mem
        gsim.timexe = timexe
        return gsim.run_gaussian()
