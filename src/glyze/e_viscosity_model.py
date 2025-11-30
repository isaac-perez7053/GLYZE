from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os, subprocess

from glyze.glyceride import Glyceride
from glyze.glyceride_mix import GlycerideMix
from glyze.gromacs import GromacsSimulator, SimPaths
from glyze.analysis import GromacsViscosityAnalysis
from glyze.slurm_file import SlurmFile
from glyze.gromacs_topgen import (
    execute_multiwfn_cm5,
    parse_cm5_from_multiwfn,
    build_top_from_mol_with_charges,
)
from glyze.packmol import PackmolSimulator


@dataclass
class EViscosityModel:
    """
    High-level driver for building a glyceride mixture, assigning CM5 charges
    (via Multiwfn), generating CM5/OPLS topologies, packing a box with Packmol,
    and running a GROMACS workflow (EM, NPT, NVT) in an organized directory
    structure.

    The intended pipeline is:

      1) Multiwfn CM5 + per-species tops       -> run_multiwfn_and_build_tops()
      2) Ensure FF dir + Packmol + init.pdb   -> prepare_initial_configuration()
      3) editconf + system.top                -> build_gromacs_system()
      4) Energy minimization (EM)             -> run_energy_minimization()
      5) NPT equilibration                    -> run_npt_equilibration()
      6) NVT equilibration                    -> run_nvt_equilibration()

    Parameters
    ----------
    mix : GlycerideMix
        Glyceride mixture to simulate. The composition (and ordering of
        `mix.mix.items()`) must be consistent with `fchk_paths` and
        `pdb_paths`.
    paths : SimPaths
        Object describing the root working directory, force-field directory,
        and the standard subdirectory layout (00_build, 10_eq_npt, etc.).
    fchk_paths : list[pathlib.Path]
        List of Gaussian (or compatible) .fchk files, one per glyceride
        species in `mix`. The order must align with `mix.mix.items()`.
        These are used by Multiwfn to compute CM5 charges.
    pdb_paths : list[pathlib.Path]
        List of template PDB structures, one per glyceride species in `mix`,
        in the same order as `fchk_paths` and `mix.mix.items()`. Used as
        geometry input for topology/Packmol building.
    resname_map : dict[Glyceride, str]
        Mapping from each glyceride object in `mix` to a 3-4 character
        residue name (e.g. ``{"N18D01P09Z": "G1", ...}``). These residue
        names are used consistently in the generated PDB/GRO/topology files.

    multiwfn_exe : str or pathlib.Path, optional
        Path to the Multiwfn executable. If ``None``, it is expected that
        ``multiwfn`` is discoverable on ``$PATH``.
    multiwfn_script : str or pathlib.Path, optional
        Path to a Multiwfn batch/script file (e.g. a CM5 automation menu
        script). If ``None``, a default interaction style or script is
        assumed by the CM5/topology stage.
    gmx_bin : str or pathlib.Path, optional
        Path to the GROMACS ``gmx`` executable (e.g.
        ``.../gromacs-2025.3/bin/gmx``). If ``None``, the plain string
        ``"gmx"`` is used and resolution is left to the environment ``$PATH``.
        Passed through to the internal :class:`GromacsSimulator`.
    gmxlib : str or pathlib.Path, optional
        Directory that should be used as ``$GMXLIB`` (typically something
        like ``.../share/gromacs/top``). If ``None``, the simulator will
        either use the current environment setting or attempt to auto-detect
        the correct data directory. Also passed to :class:`GromacsSimulator`.
    packmol_exe : str or pathlib.Path, optional
        Path to the Packmol executable. If ``None``, ``"packmol"`` is assumed
        to be available on ``$PATH``. This is forwarded to the
        :class:`PackmolSimulator` created from ``mix``.

    ff_name : str, optional
        Short name of the force field family to use, default is ``"oplsaa"``.
        This controls the main include in the system topology, e.g.
        ``#include "oplsaa.ff/forcefield.itp"``.

    tops : dict[Glyceride, dict], optional
        Optional pre-populated per-species topology metadata, mapping each
        glyceride to a dictionary with at least an ``"itp"`` key pointing to
        the generated CM5/OPLS .itp file. Normally this is filled by
        :meth:`run_multiwfn_and_build_tops` and you can leave it as the
        default ``{}``.

    Attributes
    ----------
    ff_build_dir : pathlib.Path
        Directory under which CM5/OPLS force-field/topology files are
        assembled. Set during the topology-building stage.
    gmx_sim : GromacsSimulator
        Internal GROMACS driver used for EM, NPT, and NVT stages. Constructed
        lazily once the GROMACS system is ready.
    packmol_sim : PackmolSimulator
        Internal Packmol driver used to build the initial mixture box.

    init_pdb : pathlib.Path or None
        Initial packed configuration in PDB format (after Packmol and any
        post-processing). Produced by :meth:`prepare_initial_configuration`.
    conf_gro : pathlib.Path or None
        Initial GROMACS coordinate file (e.g. from ``gmx editconf``),
        representing the starting configuration with a valid periodic box.
    system_top : pathlib.Path or None
        System topology file (``system.top``) referencing the force field and
        per-species .itp files. Produced by :meth:`build_gromacs_system`.
    em_gro : pathlib.Path or None
        Coordinate file after energy minimization. Produced by
        :meth:`run_energy_minimization`.
    npt_gro : pathlib.Path or None
        Coordinate file after NPT equilibration. Produced by
        :meth:`run_npt_equilibration`.
    nvt_gro : pathlib.Path or None
        Coordinate file after NVT equilibration. Produced by
        :meth:`run_nvt_equilibration`.
    """

    mix: GlycerideMix
    paths: SimPaths
    fchk_paths: List[Path]
    pdb_paths: List[Path]
    resname_map: Dict[Glyceride, str]

    # Executables: if None, rely on PATH; if given paths, they are expanded to abs
    multiwfn_exe: Optional[str | Path] = None
    multiwfn_script: Optional[str | Path] = None  # e.g. "cm5_menu.txt"
    gmx_bin: Optional[str | Path] = None
    gmxlib: Optional[str | Path] = None
    packmol_exe: Optional[str | Path] = None

    ff_name: str = "oplsaa"  # e.g. "oplsaa" -> "oplsaa.ff/forcefield.itp"

    # Internal fields (filled as you go)
    tops: Dict[Glyceride, Dict] = field(default_factory=dict)
    ff_build_dir: Path = field(init=False)
    gmx_sim: GromacsSimulator = field(init=False)
    packmol_sim: PackmolSimulator = field(init=False)

    init_pdb: Optional[Path] = field(default=None, init=False)
    conf_gro: Optional[Path] = field(default=None, init=False)
    system_top: Optional[Path] = field(default=None, init=False)
    em_gro: Optional[Path] = field(default=None, init=False)
    npt_gro: Optional[Path] = field(default=None, init=False)
    nvt_gro: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:

        self.paths.ensure()

        # Ensure lengths line up with the mixture
        n_species = len(self.mix.mix)
        if len(self.fchk_paths) != n_species or len(self.pdb_paths) != n_species:
            raise ValueError(
                f"fchk_paths and pdb_paths must each have length {n_species} "
                f"(number of species in mix)"
            )

        # Update RDKit mol templates from PDB coordinates
        # (ensures the mix._mol_by_glyceride uses the PDB geometries)
        self.mix.update_mols_from_pdbs(pdb_files=self.pdb_paths)

        # Multiwfn / top-build directory: root/glycerides/ff_build
        self.ff_build_dir = self.paths.workdir / "ff_build"
        self.ff_build_dir.mkdir(parents=True, exist_ok=True)

        # otherwise let PackmolSimulator use 'packmol' on PATH.
        if self.packmol_exe is None:
            self.packmol_sim = PackmolSimulator(None)
        else:
            self.packmol_sim = PackmolSimulator(self.packmol_exe)

        setattr(self.mix, "_packmol_sim", self.packmol_sim)

        # GROMACS simulator, which itself respects absolute path or PATH
        self.gmx_sim = GromacsSimulator(
            mix=self.mix,
            paths=self.paths,
            gmx_bin=self.gmx_bin,
            gmxlib=self.gmxlib,
        )

    def run_multiwfn_and_build_tops(self) -> Dict[Glyceride, Dict]:
        """
        For each species in the mix:
          - run Multiwfn CM5 on its fchk (assign charges),
          - attach CM5 charges to RDKit mol,
          - run OPLS atom typing and write:
              prefix.itp, prefix.gro, prefix.top
        All files go into ff_build_dir.

        Populates self.tops as:
          { glyceride: {"itp": Path, "top": Path, "gro": Path, ...}, ... }
        """
        self.tops.clear()

        # Resolve Multiwfn script if provided
        script_path: Optional[Path] = None
        if self.multiwfn_script is not None:
            script_path = Path(self.multiwfn_script).expanduser().resolve()

        # Work in ff_build_dir when writing topology files
        for (glyceride, _), fchk_path in zip(self.mix.mix.items(), self.fchk_paths):
            resname = self.resname_map[glyceride]
            prefix = resname.lower()

            mol = self.mix._mol_by_glyceride[glyceride]

            stdout_path = self.ff_build_dir / f"{prefix}_multiwfn.stdout"

            # Run Multiwfn CM5 analysis
            stdout_path_str = execute_multiwfn_cm5(
                fchk_path=str(Path(fchk_path).expanduser().resolve()),
                input_file=str(script_path) if script_path is not None else None,
                exe=self.multiwfn_exe,
                stdout_path=str(stdout_path),
            )

            # Parse charges
            charges = parse_cm5_from_multiwfn(stdout_path_str)
            if len(charges) != mol.GetNumAtoms():
                raise ValueError(
                    f"{resname}: number of CM5 charges ({len(charges)}) does not "
                    f"match number of atoms in RDKit mol ({mol.GetNumAtoms()})"
                )

            # Attach charges to atoms
            for atom, q in zip(mol.GetAtoms(), charges):
                atom.SetDoubleProp("_Charge", float(q))

            # Build topology files in ff_build_dir
            cwd = os.getcwd()
            try:
                os.chdir(self.ff_build_dir)
                itp_fn, top_fn, gro_fn, types = build_top_from_mol_with_charges(
                    mol=mol,
                    prefix=prefix,
                    resname=resname,
                )
            finally:
                os.chdir(cwd)

            self.tops[glyceride] = {
                "itp": self.ff_build_dir / itp_fn,
                "top": self.ff_build_dir / top_fn,
                "gro": self.ff_build_dir / gro_fn,
                "types": types,
                "resname": resname,
                "prefix": prefix,
            }

        return self.tops

    def prepare_initial_configuration(
        self,
        num_molecules: int,
        density_g_per_cm3: float,
        min_dist: float = 2.0,
        seed: Optional[int] = None,
        nloop: Optional[int] = None,
    ) -> Path:
        """
        - Ensure local forcefield directory (ff_name.ff in build dir).
        - Use GlycerideMix.build_simulation_box (via Packmol) to pack a box.
        - Write init.pdb with TER records and optional resname mappings.

        Returns
        -------
        Path to init.pdb in paths.build.
        """
        # Ensure FF dir
        self.gmx_sim._ensure_forcefield_dir(self.ff_name)

        # Pack mixture (returns MDAnalysis.Universe)
        packmol_mix = self.gmx_sim._build_packmol_mix(
            mix=self.mix,
            num_molecules=num_molecules,
            density_g_per_cm3=density_g_per_cm3,
            resname_map=self.resname_map,
            min_dist=min_dist,
            seed=seed,
            nloop=nloop,
        )

        # Write init.pdb
        self.init_pdb = self.gmx_sim._write_init_pdb(
            packmol_mix=packmol_mix,
            build_dir=self.paths.build,
            resname_map=self.resname_map,
        )

        return self.init_pdb

    def build_gromacs_system(
        self,
        num_molecules: int,
    ) -> Tuple[Path, Path]:
        """
        Use GROMACS editconf to create init.gro from init.pdb and
        build system.top using per-species .itp files (self.tops).

        Requires:
          - prepare_initial_configuration() has been called (init.pdb exists)
          - run_multiwfn_and_build_tops() has been called (self.tops populated)

        Returns
        -------
        (conf_gro, system_top)
        """
        if self.init_pdb is None:
            raise RuntimeError(
                "init_pdb not set. Run prepare_initial_configuration() first."
            )
        if not self.tops:
            raise RuntimeError(
                "tops dict is empty. Run run_multiwfn_and_build_tops() first."
            )

        # editconf: init.pdb -> init.gro (in build dir)
        print(
            "[EViscosityModel.build_gromacs_system] gmx_sim._gmx_bin_path:",
            self.gmx_sim._gmx_bin_path,
        )
        print(
            "[EViscosityModel.build_gromacs_system] gmx_sim._gmxlib_path:",
            self.gmx_sim._gmxlib_path,
        )

        self.conf_gro = Path(
            self.gmx_sim.editconf(
                build_dir=self.paths.build,
                pdb_in=self.init_pdb.name,
                gro_out="init.gro",
            )
        )
        self.conf_gro = (self.paths.build / self.conf_gro).resolve()

        # Build system.top using CM5-based .itp files
        self.system_top = GromacsSimulator.build_system_top(
            build_dir=self.paths.build,
            ff_name=self.ff_name,
            mix=self.mix,
            num_molecules=num_molecules,
            tops=self.tops,
            title="Glyceride mixture (CM5)",
        )

        return self.conf_gro, self.system_top

    def run_energy_minimization(self) -> Path:
        """
        Run energy minimization (grompp + mdrun), producing em.gro in build dir.
        Requires init.gro + system.top in build dir.
        """
        if self.conf_gro is None:
            raise RuntimeError("conf_gro not set. Run build_gromacs_system() first.")
        if self.system_top is None:
            raise RuntimeError("system_top not set. Run build_gromacs_system() first.")

        self.em_gro = self.gmx_sim._run_energy_minimization(
            build_dir=self.paths.build,
        )
        return self.em_gro

    def run_nvt_equilibration(
        self,
        T: float,
        ns: float = 2.0,
        tau_t: float = 1.0,
        slurm: SlurmFile | None = None,
    ) -> Path:
        """
        Run NVT equilibration (thermalize at fixed density).
        Uses npt.gro as input if not specified.
        """
        if self.em_gro is None:
            raise RuntimeError("em_gro not set. Run run_energy_minimization() first.")

        self.nvt_gro = self.gmx_sim.run_nvt_equilibration(
            T=T,
            ns=ns,
            tau_t=tau_t,
            start_gro=self.npt_gro,
            top=self.system_top or (self.paths.build / "system.top"),
            slurm=slurm,
        )
        return self.nvt_gro

    def run_npt_equilibration(
        self,
        T: float,
        P: float,
        ns: float = 2.0,
        tau_t: float = 1.0,
        tau_p: float = 5.0,
        slurm: SlurmFile | None = None,
    ) -> Path:
        """
        Run NPT equilibration (densify). Uses em.gro as input if not specified.
        """
        if self.nvt_gro is None:
            raise RuntimeError("nvt_gro not set. Run run_nvt_equilibration() first.")

        self.npt_gro = self.gmx_sim.run_npt_equilibration(
            T=T,
            P=P,
            ns=ns,
            tau_t=tau_t,
            tau_p=tau_p,
            start_gro=self.nvt_gro,
            top=self.system_top or (self.paths.build / "system.top"),
            slurm=slurm,
        )
        return self.npt_gro

    def run_production(
        self,
        T: float,
        ns: float = 10.0,
    ) -> Path:
        """
        Run a simple NVT production trajectory after NVT equilibration.
        """
        if self.nvt_gro is None:
            raise RuntimeError("nvt_gro not set. Run run_nvt_equilibration() first.")
        return self.gmx_sim.production(T=T, ns=ns)

    def run_pp_viscosity_sweep(
        self,
        T: float,
        ns: float,
        A_min: float,
        A_max: float,
        num_datapoints: int = 6,
        dt: float = 0.001,
        maxwarn: int = 10,
        slurm: SlurmFile | None = None,
    ) -> List[Tuple[float, Path]]:
        """
        Run a periodic-perturbation viscosity calculation for a range of
        cosine accelerations A between A_min and A_max.

        Parameters
        ----------
        T (float)
            Temperature to run the system at
        ns (float)
            Total time to run the simulation
        A_min (float)
            Minimum cos-acceleration amplitude
        A_max (float)
            Maximum cos-acceleration amplitude
        num_datapoints (int)
            Number of cos-acceleration amplitudes to simulate. More will
            likely lead to a much more trustworthy measurement of viscosity
        dt (float)
            Time step for the simulation
        maxwarn (int)
            Max number of warnings that can be raised by grompp
        slurm (SlurmFile)
            Slurm file to be used if you would like to submit jobs using SLURM

        Returns
        -------
        list of (A, edr_path)
            One entry per PP run: the amplitude and the resulting .edr file.
        """
        # Check if the following files exist before running
        if self.npt_gro is None:
            raise RuntimeError("npt_gro not set. Run run_npt_equilibration() first.")
        if self.system_top is None:
            raise RuntimeError("system_top not set. Run build_gromacs_system() first.")
        if num_datapoints < 1:
            raise ValueError("num_datapoints must be >= 1")

        # Generate a list of amplitudes (use single point in the case of A_min == A_max)
        if num_datapoints == 1 or A_min == A_max:
            amplitudes = [0.5 * (A_min + A_max)]
        else:
            step = (A_max - A_min) / (num_datapoints - 1)
            amplitudes = [A_min + i * step for i in range(num_datapoints)]

        results: List[Tuple[float, Path]] = []

        # Run viscosity calculation for various amplitudes
        for A in amplitudes:
            edr_path = self.gmx_sim.run_viscosity_pp(
                T=T,
                ns=ns,
                cos_acceleration=A,
                dt=dt,
                maxwarn=maxwarn,
                start_gro=self.npt_gro,
                top=self.system_top,
                slurm=slurm,
            )
            # Create a list of amplitudes and paths to .edr (results) file
            results.append((A, edr_path))
        return results

    def analyze_results(
        self, pp_runs: List[Tuple[float, Path]], ns_per_A: float, T: float
    ) -> None:
        L_values = []
        inveta_xvg_files = []
        temp_xvg_files = []

        for L, edr_path in pp_runs:
            edr_path = Path(edr_path)
            run_dir = edr_path.parent

            tag = f"{L:.5f}".replace(".", "")  # 0.01000 -> "001000"
            inveta_xvg = run_dir / f"pp_L{tag}_{int(ns_per_A)}ns_visc.xvg"
            temp_xvg = run_dir / f"pp_L{tag}_{int(ns_per_A)}ns_temp.xvg"

            self._extract_pp_xvg_for_run(
                edr_path=edr_path,
                inveta_xvg=inveta_xvg,
                temp_xvg=temp_xvg,
            )

            L_values.append(L)
            inveta_xvg_files.append(inveta_xvg)
            temp_xvg_files.append(temp_xvg)

        analysis = GromacsViscosityAnalysis()

        fit = analysis.analyze_pp_from_xvg(
            L_values=L_values,
            inveta_xvg=inveta_xvg_files,
            temp_xvg=temp_xvg_files,
            tmin_ns=2.0,
            block_ns=3.0,
        )

        print(f"PP viscosity at {T:.1f} K:")
        print(f"  eta0 = {fit.eta0_cP:.3f} ± {fit.eta0_cP_err:.3f} cP")
        print(f"  R^2(eta(L)) = {fit.r2:.4f}")
        if fit.qc_warnings:
            print("QC warnings:")
            for msg in fit.qc_warnings:
                print(f"  - {msg}")
        else:
            print("QC: no warnings")

        print("\nPer-amplitude points:")
        for p in fit.points:
            print(
                f"L={p.L:.5f}, "
                f"eta={p.eta_cP:.3f} pm {p.eta_cP_sem:.3f} cP, "
                f"Tmean={p.temp_mean_K if p.temp_mean_K is not None else float('nan'):.2f} K"
            )

    def _extract_pp_xvg_for_run(
        self,
        edr_path: Path,
        inveta_xvg: Path,
        temp_xvg: Path,
    ) -> None:
        """
        Helper: run `gmx energy` on a PP .edr file to get 1/eta(t) and T(t) XVG.

        inveta_term should be the exact name or index that selects 1/eta
        in the `gmx energy` menu for your PP runs.
        """
        edr_path = Path(edr_path)
        cwd = edr_path.parent
        gmx = self.gmx_sim._gmx_executable()
        env = self.gmx_sim._gmx_env()

        # 1/eta(t)
        proc1 = subprocess.run(
            [gmx, "energy", "-f", edr_path.name, "-o", inveta_xvg.name],
            cwd=cwd,
            env=env,
            input=f"39\n0\n",
            text=True,
            capture_output=True,
        )
        if proc1.returncode != 0:
            raise RuntimeError(
                f"gmx energy for inveta failed on {edr_path}:\n{proc1.stdout}\n{proc1.stderr}"
            )

        # Temperature(t)
        proc2 = subprocess.run(
            [gmx, "energy", "-f", edr_path.name, "-o", temp_xvg.name],
            cwd=cwd,
            env=env,
            input=f"15\n0\n",
            text=True,
            capture_output=True,
        )
        if proc2.returncode != 0:
            raise RuntimeError(
                f"gmx energy for temperature failed on {edr_path}:\n{proc2.stdout}\n{proc2.stderr}"
            )
