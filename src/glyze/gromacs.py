from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re

from glyze.glyceride_mix import GlycerideMix
from MDAnalysis.coordinates.GRO import GROWriter
import MDAnalysis as mda
import subprocess
import os
import shutil
from glyze.utils import replace_resnames_by_chain


@dataclass
class SimPaths:
    root: Path
    ffdir: Path
    name: str = "glycerides"

    def __post_init__(self) -> None:
        # Allow passing strings, '~', and relative paths
        self.root = Path(self.root).expanduser().resolve()
        self.ffdir = Path(self.ffdir).expanduser().resolve()

    @property
    def workdir(self) -> Path:
        return self.root / self.name

    @property
    def build(self) -> Path:
        return self.workdir / "00_build"

    @property
    def npt(self) -> Path:
        return self.workdir / "10_eq_npt"

    @property
    def nvt(self) -> Path:
        return self.workdir / "20_eq_nvt"

    @property
    def prod(self) -> Path:
        return self.workdir / "30_prod"

    @property
    def nemd(self) -> Path:
        return self.workdir / "40_nemd"

    def ensure(self) -> None:
        for d in (self.build, self.npt, self.nvt, self.prod, self.nemd):
            d.mkdir(parents=True, exist_ok=True)


class GromacsSimulator:
    """
    GROMACS workflow using gmxapi:

    1. Take a Packmol PDB (mixture box) and build:
       - init.pdb, conf.gro, topol.top, em.gro, system.top
    2. Equilibrate (NPT -> NVT)
    3. Production + viscosity (Green-Kubo or NEMD)
    """

    def __init__(
        self,
        mix: GlycerideMix,
        paths: SimPaths,
        gmx_bin: str | Path | None = None,
        gmxlib: str | Path | None = None,
    ):
        self.mix = mix
        self.paths = paths
        self.paths.ensure()

        if gmx_bin is None:
            self._gmx_bin_path = None
        else:
            self._gmx_bin_path = Path(gmx_bin).expanduser().resolve()

        if gmxlib is not None:
            self._gmxlib_path = Path(gmxlib).expanduser().resolve()
        else:
            self._gmxlib_path = self._detect_gmxlib_from_executable()

        print("[GromacsSimulator.__init__] using GMXLIB:", self._gmxlib_path)
        print("[GromacsSimulator.__init__] resolved _gmx_bin_path:", self._gmx_bin_path)

    def _gmx_executable(self) -> str:
        """
        Return the executable name/path for GROMACS.

        If a path was provided at init, use that (absolute); otherwise just 'gmx'
        and rely on the environment PATH.
        """
        if self._gmx_bin_path is not None:
            return str(self._gmx_bin_path)
        return "gmx"

    def _detect_gmxlib_from_executable(self) -> Path:
        """
        Detect GMXLIB from the location of the GROMACS 'gmx' executable.

        Requirements:
        1) A valid 'gmx' executable must be found (either from self._gmx_bin_path
            or by searching $PATH).
        2) The inferred GMXLIB directory must contain 'residuetypes.dat'.

        Layout assumed (matches current cluster environment):
            <prefix>/.../bin/gmx        (or similar)
            <prefix>/share/top/residuetypes.dat

        Returns
        -------
        pathlib.Path
            Resolved GMXLIB directory (the directory that contains residuetypes.dat).

        Raises
        ------
        RuntimeError
            If 'gmx' cannot be found, or if 'residuetypes.dat' cannot be located
            in the inferred GMXLIB directory.
        """
        # Find gmx executable
        if getattr(self, "_gmx_bin_path", None) is not None:
            gmx_path = Path(self._gmx_bin_path).expanduser().resolve()
            if not gmx_path.is_file():
                raise RuntimeError(
                    f"gmx_bin was set to '{gmx_path}', but that is not a file. "
                    "Please provide a valid path to the GROMACS 'gmx' executable."
                )
        else:
            gmx_exe = shutil.which("gmx")
            if gmx_exe is None:
                raise RuntimeError(
                    "Could not find 'gmx' on PATH. Either:\n"
                    "  * Install / module-load GROMACS so that 'gmx' is on $PATH, or\n"
                    "  * Pass gmx_bin=... explicitly when constructing GromacsSimulator."
                )
            gmx_path = Path(gmx_exe).resolve()

        # Infer GMXLIB from gmx_path
        # Your working layout: go up 3 levels, then share/top
        prefix = gmx_path.parent.parent.parent
        gmxlib = (prefix / "share" / "top").resolve()
        residuetypes = gmxlib / "residuetypes.dat"

        if not residuetypes.is_file():
            raise RuntimeError(
                "Auto-detection of GMXLIB failed: 'residuetypes.dat' not found.\n"
                f"gmx executable: {gmx_path}\n"
                f"Expected GMXLIB: {gmxlib}\n"
                f"Expected file:   {residuetypes}\n\n"
                "Please either:\n"
                "  * Set gmxlib explicitly when constructing GromacsSimulator, or\n"
                "  * Set the GMXLIB environment variable to a directory that "
                "contains 'residuetypes.dat'."
            )

        return gmxlib

    def _gmx_env(self) -> dict:
        """
        Environment for calling GROMACS through subprocess.
        Ensures GMXLIB is set if a path was provided at init.
        """
        env = os.environ.copy()

        if self._gmxlib_path is not None:
            env["GMXLIB"] = str(self._gmxlib_path)
        else:
            pass

        return env

    @staticmethod
    def _extract_moleculetype_name(itp_path: Path) -> str:
        """
        Parse the first moleculetype name from an .itp file.

        Looks for:

            [ moleculetype ]
            ; comment
            NAME   nrexcl

        and returns NAME (e.g. "TRIOL").
        """
        text = itp_path.read_text().splitlines()
        in_section = False
        for line in text:
            # strip comments
            line_nocom = line.split(";", 1)[0].strip()
            if not line_nocom:
                continue

            if line_nocom.startswith("["):
                # entering / leaving sections
                in_section = "[ moleculetype" in line_nocom.lower()
                continue

            if in_section:
                parts = line_nocom.split()
                if parts:
                    return parts[0]

        raise ValueError(f"Could not find [ moleculetype ] name in {itp_path}")

    @staticmethod
    def build_system_top(
        build_dir: Path,
        ff_name: str,
        mix: GlycerideMix,
        num_molecules: int,
        tops: Dict,
        title: str = "Glyceride mixture",
    ) -> Path:
        """
        Build a GROMACS system topology for the full box.

        Parameters
        ----------
        build_dir : Path
            Directory where system.top will be written (and .itp files copied).
        ff_name : str
            Force field name, e.g. "oplsaa" (expects ff dir "oplsaa.ff").
        mix : GlycerideMix
            The mixture object used to build the Packmol box.
        num_molecules : int
            Total number of molecules in the box (same as used in Packmol).
        tops : dict
            Mapping: glyceride_obj -> dict with at least {"itp": Path(...)}.
        title : str
            Title for the [ system ] section.

        Returns
        -------
        Path to system.top in build_dir.
        """
        build_dir.mkdir(parents=True, exist_ok=True)
        system_top = build_dir / "system.top"

        # Copy per-species .itp into build_dir and record local paths
        local_itps: Dict[object, Path] = {}
        for glyceride, info in tops.items():
            itp_src = Path(info["itp"])
            itp_dst = build_dir / itp_src.name
            if not itp_dst.exists():
                shutil.copy(itp_src, itp_dst)
            local_itps[glyceride] = itp_dst

        # Compute integer counts (same logic as Packmol)
        total_qty = mix.total_quantity()
        if total_qty <= 0:
            raise ValueError("Total quantity in mix must be positive.")

        mol_fractions = {g: qty / total_qty for g, qty in mix.mix.items()}
        counts = mix._integer_counts_from_fractions(mol_fractions, num_molecules)

        # Build lines of system.top
        lines = []
        # include the force field master .itp (assumes e.g. "oplsaa.ff/forcefield.itp")
        lines.append(f'#include "{ff_name}.ff/forcefield.itp"')

        # include each species' CM5-charged topology
        for glyceride, itp_path in local_itps.items():
            lines.append(f'#include "{itp_path.name}"')

        lines.append("")
        lines.append("[ system ]")
        lines.append(title)
        lines.append("")
        lines.append("[ molecules ]")
        lines.append("; Compound    nmols")

        # Determine moleculetype name from each .itp and write counts
        for glyceride, n in counts.items():
            if n == 0:
                continue
            if glyceride not in local_itps:
                raise KeyError(
                    f"No .itp in `tops` for glyceride {glyceride}; "
                    "cannot add it to [ molecules ]."
                )
            itp_path = local_itps[glyceride]
            moleculetype_name = GromacsSimulator._extract_moleculetype_name(itp_path)
            lines.append(f"{moleculetype_name:10s} {int(n)}")

        system_top.write_text("\n".join(lines) + "\n")
        return system_top

    def _ensure_forcefield_dir(self, ff_name: str) -> Path:
        """
        Ensure ff_name.ff exists in build_dir. Returns local ffdir path.
        """
        build_dir = self.paths.build
        build_dir.mkdir(parents=True, exist_ok=True)

        local_ffdir = build_dir / f"{ff_name}.ff"
        if not local_ffdir.exists():
            shutil.copytree(self.paths.ffdir, local_ffdir, dirs_exist_ok=True)
        return local_ffdir

    @staticmethod
    def _ensure_ter_between_residues(pdb_path: Path):
        """
        Insert TER records between residues in a PDB (simple, testable).
        """
        lines = pdb_path.read_text().splitlines()
        out = []
        prev_resid = None
        prev_chain = None
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                resid = line[22:26]  # resSeq
                chain = line[21:22]
                if prev_resid is not None and (
                    resid != prev_resid or chain != prev_chain
                ):
                    out.append("TER")
                prev_resid, prev_chain = resid, chain
                out.append(line)
            else:
                out.append(line)
        if out and out[-1] != "TER":
            out.append("TER")
        pdb_path.write_text("\n".join(out) + "\n")

    @staticmethod
    def make_chain_resname_map(resname_map: Dict[object, str]) -> Dict[str, str]:
        """
        Convert {glyceride_obj: 'RESN'} -> {'A': 'RESN', 'B': 'RESN2', ...}
        """

        def chain_label(n: int) -> str:
            label = ""
            while True:
                n, r = divmod(n, 26)
                label = chr(ord("A") + r) + label
                if n == 0:
                    break
                n -= 1
            return label

        chain_map = {}
        for idx, (_, resname) in enumerate(resname_map.items()):
            chain_map[chain_label(idx)] = resname
        return chain_map

    def _build_packmol_mix(
        self,
        mix,
        num_molecules: int,
        density_g_per_cm3: float,
        resname_map: Dict[object, str] | None = None,
        **kwargs,
    ) -> mda.Universe:
        """
        Call GlycerideMix.build_simulation_box and sanitize residue names.
        Returns an MDAnalysis.Universe.
        """
        packmol_mix = mix.build_simulation_box(
            num_molecules=num_molecules,
            density_g_per_cm3=density_g_per_cm3,
            **({} if resname_map is None else {"resname_map": resname_map}),
            **kwargs,
        )

        # Ensure clean 3-char alphanumeric resnames
        for r in packmol_mix.residues:
            if not r.resname:
                r.resname = "UNK"
            r.resname = re.sub(r"[^A-Za-z0-9]", "", r.resname.upper())[:3] or "UNK"

        return packmol_mix

    def _write_init_pdb(
        self,
        packmol_mix: mda.Universe,
        build_dir: Path,
        resname_map: Dict[object, str] | None = None,
    ) -> Path:
        """
        Write init.pdb from packmol_mix, add TERs, and (optionally) fix resnames
        using chain IDs + resname_map.
        """
        init_pdb = build_dir / "init.pdb"
        with mda.Writer(str(init_pdb), multiframe=False) as w:
            w.write(packmol_mix.atoms)

        # Insert TER records
        self._ensure_ter_between_residues(init_pdb)

        # Optional: remap resnames based on chain ID -> residue name
        if resname_map is not None:
            chain_resname_map = self.make_chain_resname_map(resname_map)
            replace_resnames_by_chain(
                pdb_file=str(init_pdb),
                chain_resname_map=chain_resname_map,
                output_file=str(init_pdb),
            )

        return init_pdb

    @staticmethod
    def pdb_to_gro(input_pdb_file, output_gro_file):
        """
        Converts a .pdb file to a .gro file using MDAnalysis.

        Args:
            input_pdb_file (str): Path to the input .pdb file.
            output_gro_file (str): Path for the output .gro file.
        """
        try:
            # Create a Universe object from the PDB file
            universe = mda.Universe(input_pdb_file)

            # Write the Universe to a GRO file
            with GROWriter(output_gro_file) as writer:
                writer.write(universe)

            print(f"Successfully converted '{input_pdb_file}' to '{output_gro_file}'")

        except Exception as e:
            print(f"Error during conversion: {e}")

    @staticmethod
    def _extract_moleculetype_name(itp_path: Path) -> str:
        """
        Parse the first moleculetype name from an .itp file.
        """
        text = itp_path.read_text().splitlines()
        in_section = False
        for line in text:
            line_nocom = line.split(";", 1)[0].strip()
            if not line_nocom:
                continue
            if line_nocom.startswith("["):
                in_section = "[ moleculetype" in line_nocom.lower()
                continue
            if in_section:
                parts = line_nocom.split()
                if parts:
                    return parts[0]
        raise ValueError(f"Could not find [ moleculetype ] name in {itp_path}")

    def editconf(
        self,
        build_dir: Path,
        pdb_in: str = "init.pdb",
        gro_out: str = "init.gro",
        distance: float = 1.0,
    ):

        env = self._gmx_env()
        gmx = self._gmx_executable()

        cmd = [
            gmx,
            "editconf",
            "-f",
            str(pdb_in),
            "-o",
            str(gro_out),
            "-c",
            "-d",
            str(distance),
            "-bt",
            "cubic",
        ]

        print("[editconf] gmx executable:", gmx)
        print("[editconf] working directory (cwd):", build_dir)
        print("[editconf] GMXLIB in env:", env.get("GMXLIB"))
        print("[editconf] full command:", " ".join(cmd))

        # Optional: verify that residuetypes.dat is actually there
        gmxlib_path = env.get("GMXLIB")
        if gmxlib_path:
            from pathlib import Path

            rt = Path(gmxlib_path) / "residuetypes.dat"
            print("[editconf] residuetypes.dat expected at:", rt)
            print("[editconf] residuetypes.dat exists? ->", rt.is_file())

        # Optional: print gmx --version under the same env
        try:
            version_result = subprocess.run(
                [gmx, "--version"],
                cwd=str(build_dir),
                env=env,
                capture_output=True,
                text=True,
            )
            print("[editconf] gmx --version stdout:\n", version_result.stdout)
            print("[editconf] gmx --version stderr:\n", version_result.stderr)
        except Exception as e:
            print("[editconf] ERROR running 'gmx --version':", e)

        result = subprocess.run(
            cmd,
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            print("editconf stderr:")
            print(result.stderr)
            raise RuntimeError("editconf failed")

        print("editconf stdout:")
        print(result.stdout)

        return gro_out

    @staticmethod
    def build_system_top(
        build_dir: Path,
        ff_name: str,
        mix,
        num_molecules: int,
        tops: Dict,
        title: str = "Glyceride mixture",
    ) -> Path:
        """
        Build system.top from:
        - forcefield name (ff_name, e.g. 'oplsaa')
        - GlycerideMix (mix) + num_molecules
        - per-species tops dict {glyceride: {'itp': Path(...), ...}}
        """
        system_top = build_dir / "system.top"
        system_top.parent.mkdir(parents=True, exist_ok=True)

        # Copy .itp files to build_dir
        import shutil

        local_itps: Dict[object, Path] = {}
        for glyceride, info in tops.items():
            itp_src = Path(info["itp"])
            itp_dst = build_dir / itp_src.name
            if not itp_dst.exists():
                shutil.copy(itp_src, itp_dst)
            local_itps[glyceride] = itp_dst

        # Compute counts just like Packmol
        total_qty = mix.total_quantity()
        if total_qty <= 0:
            raise ValueError("Total quantity in mix must be positive.")

        mol_fractions = {g: qty / total_qty for g, qty in mix.mix.items()}
        counts = mix._integer_counts_from_fractions(mol_fractions, num_molecules)

        lines = []
        lines.append(f'#include "{ff_name}.ff/forcefield.itp"')

        for glyceride, itp_path in local_itps.items():
            lines.append(f'#include "{itp_path.name}"')

        lines.append("")
        lines.append("[ system ]")
        lines.append(title)
        lines.append("")
        lines.append("[ molecules ]")
        lines.append("; Compound    nmols")

        for glyceride, n in counts.items():
            if n == 0:
                continue
            itp_path = local_itps.get(glyceride)
            if itp_path is None:
                raise KeyError(
                    f"No .itp in `tops` for glyceride {glyceride}; "
                    "cannot add it to [ molecules ]."
                )
            moltype = GromacsSimulator._extract_moleculetype_name(itp_path)
            lines.append(f"{moltype:10s} {int(n)}")

        system_top.write_text("\n".join(lines) + "\n")
        return system_top

    def _run_energy_minimization(self, build_dir: Path, maxwarn: int = 6) -> Path:
        """
        Run grompp + mdrun for energy minimization. Returns em.gro.
        """
        em_mdp = build_dir / "em.mdp"
        self._write_em_mdp(em_mdp)
        em_tpr = "em.tpr"
        em_top = "system.top"
        em_gro = "init.gro"

        env = self._gmx_env()
        gmx = self._gmx_executable()

        # grompp for EM
        grompp_cmd = [
            gmx,
            "grompp",
            "-f",
            "em.mdp",
            "-c",
            em_gro,
            "-p",
            em_top,
            "-o",
            em_tpr,
            "-maxwarn",
            str(maxwarn),
        ]
        result = subprocess.run(
            grompp_cmd,
            cwd=str(build_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print(f"Printing grompp_cmd: {grompp_cmd}")
        print("grompp (EM) stdout:\n", result.stdout)
        print("grompp (EM) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"grompp (EM) failed with exit code {result.returncode}")

        # mdrun for EM
        mdrun_cmd = [gmx, "mdrun", "-deffnm", "em"]
        result = subprocess.run(
            mdrun_cmd,
            cwd=str(build_dir),
            env=env,
            capture_output=True,
            text=True,
        )

        print("mdrun (EM) stdout:\n", result.stdout)
        print("mdrun (EM) stderr:\n", result.stderr)
        em_gro = build_dir / "em.gro"
        if not em_gro.exists():
            raise RuntimeError("Energy minimization did not produce em.gro.")
        return em_gro

    def build_system(
        self,
        mix,
        num_molecules: int,
        density_g_per_cm3: float,
        ff_name: str = "oplsaa",
        tops: dict | None = None,
        resname_map: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        High-level pipeline:
        1. Ensure FF dir
        2. Build Packmol mixture (Universe)
        3. Write init.pdb (+ TER + resname fix)
        4. Convert to conf.gro (editconf)
        5. Build system.top from .itp + mix
        6. Run energy minimization

        Returns dict with key paths.
        """
        if tops is None:
            raise ValueError("You must pass `tops` (per-glyceride topology info).")

        build_dir = self.paths.build
        build_dir.mkdir(parents=True, exist_ok=True)

        # Ensure FF dir
        self._ensure_forcefield_dir(ff_name=ff_name)

        # Pack mixture
        packmol_mix = self._build_packmol_mix(
            mix=mix,
            num_molecules=num_molecules,
            density_g_per_cm3=density_g_per_cm3,
            resname_map=resname_map,
            **kwargs,
        )

        # Write init.pdb
        init_pdb = self._write_init_pdb(
            packmol_mix=packmol_mix,
            build_dir=build_dir,
            resname_map=resname_map,
        )

        # PDB -> GRO
        gro_path = self.editconf(
            init_pdb=init_pdb,
            build_dir=build_dir,
        )

        # system.top
        system_top = GromacsSimulator.build_system_top(
            build_dir=build_dir,
            ff_name=ff_name,
            mix=mix,
            num_molecules=num_molecules,
            tops=tops,
            title="Glyceride mixture (CM5)",
        )

        # Energy minimization   
        em_gro = self._run_energy_minimization(
            build_dir=build_dir,
            gro_path=gro_path,
            system_top=system_top,
        )

        return {
            "init_pdb": init_pdb,
            "conf_gro": gro_path,
            "system_top": system_top,
            "em_gro": em_gro,
        }

    def run_npt_equilibration(
        self,
        T: float,
        P: float,
        ns: float = 2.0,
        tau_t: float = 1.0,
        tau_p: float = 5.0,
        start_gro: str | Path | None = None,
        top: str | Path | None = None,
        maxwarn: int = 6,
    ) -> Path:
        """
        Run NPT equilibration (densify).

        Parameters
        ----------
        T : float
            Target temperature (K).
        P : float
            Target pressure (bar).
        ns : float
            Length of NPT run in nanoseconds.
        tau_t : float
            Temperature coupling time constant (ps).
        tau_p : float
            Pressure coupling time constant (ps).
        start_gro : str | Path | None
            Input configuration (.gro). If None, uses build/em.gro.
        top : str | Path | None
            Topology file (.top). If None, uses build/system.top.

        Returns
        -------
        Path
            Path to npt.gro in self.paths.npt.
        """
        npt_dir = self.paths.npt
        npt_dir.mkdir(parents=True, exist_ok=True)

        if start_gro is None:
            start_gro = self.paths.build / "em.gro"
        if top is None:
            top = self.paths.build / "system.top"

        start_gro = Path(start_gro)
        top = Path(top)

        env = self._gmx_env()
        gmx = self._gmx_executable()

        # Write NPT MDP
        npt_mdp = npt_dir / "npt.mdp"
        self._write_npt_mdp(npt_mdp, T=T, P=P, ns=ns, tau_t=tau_t, tau_p=tau_p)
        npt_tpr = npt_dir / "npt.tpr"

        # grompp (NPT)
        grompp_cmd = [
            gmx,
            "grompp",
            "-f",
            npt_mdp.name,
            "-c",
            str(start_gro),
            "-p",
            str(top),
            "-o",
            npt_tpr.name,
            "-maxwarn",
            str(maxwarn),
        ]
        result = subprocess.run(
            grompp_cmd,
            cwd=str(npt_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("grompp (NPT) stdout:\n", result.stdout)
        print("grompp (NPT) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"grompp (NPT) failed with exit code {result.returncode}"
            )

        # mdrun (NPT)
        mdrun_cmd = [gmx, "mdrun", "-deffnm", "npt"]
        result = subprocess.run(
            mdrun_cmd,
            cwd=str(npt_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("mdrun (NPT) stdout:\n", result.stdout)
        print("mdrun (NPT) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"mdrun (NPT) failed with exit code {result.returncode}")

        npt_gro = npt_dir / "npt.gro"
        if not npt_gro.exists():
            raise RuntimeError("NPT equilibration did not produce npt.gro.")

        return npt_gro

    def run_nvt_equilibration(
        self,
        T: float,
        ns: float = 2.0,
        tau_t: float = 1.0,
        start_gro: str | Path | None = None,
        top: str | Path | None = None,
        maxwarn: int = 6,
    ) -> Path:
        """
        Run NVT equilibration (thermalize at fixed density).

        Parameters
        ----------
        T : float
            Target temperature (K).
        ns : float
            Length of NVT run in nanoseconds.
        tau_t : float
            Temperature coupling time constant (ps).
        start_gro : str | Path | None
            Input configuration (.gro). If None, uses npt/npt.gro.
        top : str | Path | None
            Topology file (.top). If None, uses build/system.top.

        Returns
        -------
        Path
            Path to nvt.gro in self.paths.nvt.
        """
        nvt_dir = self.paths.nvt
        nvt_dir.mkdir(parents=True, exist_ok=True)

        if start_gro is None:
            start_gro = self.paths.npt / "npt.gro"
        if top is None:
            top = self.paths.build / "system.top"

        start_gro = Path(start_gro)
        top = Path(top)

        env = self._gmx_env()
        gmx = self._gmx_executable()

        # Write NVT MDP
        nvt_mdp = nvt_dir / "nvt.mdp"
        self._write_nvt_mdp(nvt_mdp, T=T, ns=ns, tau_t=tau_t)
        nvt_tpr = nvt_dir / "nvt.tpr"

        # grompp (NVT)
        grompp_cmd = [
            gmx,
            "grompp",
            "-f",
            nvt_mdp.name,
            "-c",
            str(start_gro),
            "-p",
            str(top),
            "-o",
            nvt_tpr.name,
            "-maxwarn",
            str(maxwarn),
        ]
        result = subprocess.run(
            grompp_cmd,
            cwd=str(nvt_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("grompp (NVT) stdout:\n", result.stdout)
        print("grompp (NVT) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"grompp (NVT) failed with exit code {result.returncode}"
            )

        # mdrun (NVT)
        mdrun_cmd = [gmx, "mdrun", "-deffnm", "nvt"]
        result = subprocess.run(
            mdrun_cmd,
            cwd=str(nvt_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("mdrun (NVT) stdout:\n", result.stdout)
        print("mdrun (NVT) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"mdrun (NVT) failed with exit code {result.returncode}")

        nvt_gro = nvt_dir / "nvt.gro"
        if not nvt_gro.exists():
            raise RuntimeError("NVT equilibration did not produce nvt.gro.")

        return nvt_gro

    def equilibrate(
        self,
        T: float,
        P: float,
        npt_ns: float = 2.0,
        nvt_ns: float = 2.0,
        tau_t: float = 1.0,
        tau_p: float = 5.0,
    ) -> dict:
        """
        Convenience wrapper: NPT (densify) -> NVT (thermalize).
        """
        npt_gro = self.run_npt_equilibration(
            T=T,
            P=P,
            ns=npt_ns,
            tau_t=tau_t,
            tau_p=tau_p,
        )
        nvt_gro = self.run_nvt_equilibration(
            T=T,
            ns=nvt_ns,
            tau_t=tau_t,
            start_gro=npt_gro,
        )
        top = self.paths.build / "system.top"
        return {"npt_gro": npt_gro, "nvt_gro": nvt_gro, "top": top}

    def production(self, T: float, ns: float = 10.0, maxwarn: int = 6) -> Path:
        """
        Simple NVT production (for later analysis if desired).
        """
        prod_dir = self.paths.prod
        prod_dir.mkdir(parents=True, exist_ok=True)

        start = self.paths.nvt / "nvt.gro"
        top = self.paths.build / "system.top"
        env = self._gmx_env()
        gmx = self._gmx_executable()

        mdp = prod_dir / "prod.mdp"
        self._write_prod_nvt_mdp(mdp, T=T, ns=ns, write_press=True)

        tpr = prod_dir / "prod.tpr"

        grompp_cmd = [
            gmx,
            "grompp",
            "-f",
            mdp.name,
            "-c",
            str(start),
            "-p",
            str(top),
            "-o",
            tpr.name,
            "-maxwarn",
            str(maxwarn),
        ]
        result = subprocess.run(
            grompp_cmd,
            cwd=str(prod_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("grompp (prod) stdout:\n", result.stdout)
        print("grompp (prod) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"grompp (prod) failed with exit code {result.returncode}"
            )

        mdrun_cmd = [gmx, "mdrun", "-deffnm", "prod"]
        result = subprocess.run(
            mdrun_cmd,
            cwd=str(prod_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        print("mdrun (prod) stdout:\n", result.stdout)
        print("mdrun (prod) stderr:\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"mdrun (prod) failed with exit code {result.returncode}"
            )

        return prod_dir / "prod.edr"

    def viscosity_green_kubo(self) -> Path:
        """
        Post-process equilibrium production with Green-Kubo.
        Placeholder: implement your own viscosity calculator here.
        """
        pass

    def viscosity_nemd(self, T: float, shear_rate_ps: float, ns: float = 5.0) -> Path:
        """
        NEMD viscosity via imposed shear.
        Placeholder: implement your own NEMD scheme / analysis here.
        """
        pass

    # MDP file writers

    def _write_em_mdp(self, path: Path) -> None:
        path.write_text(
            "; Energy minimization\n"
            "integrator  = steep\n"
            "nsteps      = 5000\n"
            "emtol       = 1000.0\n"
            "emstep      = 0.01\n"
            "cutoff-scheme = Verlet\n"
            "coulombtype = PME\n"
            "rcoulomb    = 1.2\n"
            "rvdw        = 1.2\n"
            "constraints = h-bonds\n"
        )

    def _write_npt_mdp(
        self, path: Path, T: float, P: float, ns: float, tau_t: float, tau_p: float
    ) -> None:
        dt = 0.002  # ps
        nsteps = int((ns * 1e6) / (dt * 1e3))  # ns -> steps at 2 fs
        path.write_text(
            f"; NPT equilibration\n"
            f"integrator       = md\n"
            f"dt               = {dt}\n"
            f"nsteps           = {nsteps}\n"
            f"tcoupl           = v-rescale\n"
            f"tc-grps          = System\n"
            f"tau_t            = {tau_t}\n"
            f"ref_t            = {T}\n"
            f"pcoupl           = C-rescale\n"
            f"pcoupltype       = isotropic\n"
            f"tau_p            = {tau_p}\n"
            f"ref_p            = {P}\n"
            f"compressibility  = 4.5e-5\n"
            f"constraints      = h-bonds\n"
            f"constraint-algorithm = lincs\n"
            f"cutoff-scheme    = Verlet\n"
            f"coulombtype      = PME\n"
            f"rcoulomb         = 1.2\n"
            f"rvdw             = 1.2\n"
            f"nstxout-compressed = 1000\n"
            f"nstenergy        = 1000\n"
        )

    def _write_nvt_mdp(self, path: Path, T: float, ns: float, tau_t: float) -> None:
        dt = 0.002
        nsteps = int((ns * 1e6) / (dt * 1e3))
        path.write_text(
            f"; NVT equilibration\n"
            f"integrator     = md\n"
            f"dt             = {dt}\n"
            f"nsteps         = {nsteps}\n"
            f"tcoupl         = v-rescale\n"
            f"tc-grps        = System\n"
            f"tau_t          = {tau_t}\n"
            f"ref_t          = {T}\n"
            f"constraints    = h-bonds\n"
            f"cutoff-scheme  = Verlet\n"
            f"coulombtype    = PME\n"
            f"rcoulomb       = 1.2\n"
            f"rvdw           = 1.2\n"
            f"nstxout-compressed = 1000\n"
            f"nstenergy      = 1000\n"
        )

    def _write_prod_nvt_mdp(
        self, path: Path, T: float, ns: float, write_press: bool = True
    ) -> None:
        dt = 0.002
        nsteps = int((ns * 1e6) / (dt * 1e3))
        extras = ""
        if write_press:
            extras = (
                "nstcalcenergy = 1\n"
                "nstenergy     = 100\n"
                "nstxout-compressed = 1000\n"
                "; ensure pressure tensor is saved (GROMACS saves in .edr)\n"
            )
        path.write_text(
            f"; Production NVT (for Green-Kubo)\n"
            f"integrator     = md\n"
            f"dt             = {dt}\n"
            f"nsteps         = {nsteps}\n"
            f"tcoupl         = v-rescale\n"
            f"tc-grps        = System\n"
            f"tau_t          = 1.0\n"
            f"ref_t          = {T}\n"
            f"constraints    = h-bonds\n"
            f"cutoff-scheme  = Verlet\n"
            f"coulombtype    = PME\n"
            f"rcoulomb       = 1.2\n"
            f"rvdw           = 1.2\n"
            f"{extras}"
        )

    def _write_nemd_shear_mdp(
        self, path: Path, T: float, ns: float, shear_rate_ps: float
    ) -> None:
        """
        Example NEMD via periodic box deformation (simple Couette-like shear).
        You *must* validate physics for your system; this is a scaffold.
        """
        dt = 0.002
        nsteps = int((ns * 1e6) / (dt * 1e3))
        # GROMACS 'deform' deforms the box every step (units: 1/ps). Example shears xy.
        # Turn off pressure coupling or use anisotropic schemes carefully during shear.
        path.write_text(
            f"; NEMD shear deformation (scaffold)\n"
            f"integrator     = md\n"
            f"dt             = {dt}\n"
            f"nsteps         = {nsteps}\n"
            f"tcoupl         = v-rescale\n"
            f"tc-grps        = System\n"
            f"tau_t          = 1.0\n"
            f"ref_t          = {T}\n"
            f"pcoupl         = no\n"
            f"constraints    = h-bonds\n"
            f"cutoff-scheme  = Verlet\n"
            f"coulombtype    = PME\n"
            f"rcoulomb       = 1.2\n"
            f"rvdw           = 1.2\n"
            f"; Box deformation: shear in xy; set others to 0.0\n"
            f"deform         = {shear_rate_ps} 0.0 0.0  0.0 0.0 0.0\n"
            f"nstxout-compressed = 2000\n"
            f"nstenergy      = 500\n"
        )
