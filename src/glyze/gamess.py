from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional, Dict, Any

from rdkit import Chem
from glyze.slurm_header import SlurmHeader
from glyze.slurm_file import SlurmFile
import textwrap


class GAMESSSimulation:
    """
    Minimal, standalone wrapper to run GAMESS directly from an RDKit Mol.

      1) Builds a GAMESS $DATA input from the RDKit conformer
      2) Writes job.inp in a scratch directory
      3) Copies it to the generic INPUT file (F05) OR does so inside a SLURM script
      4) Sets INPUT/PUNCH/SCR/USERSCR/GMSPATH/AUXDATA
      5) Runs gamess.00.x directly (locally or under SLURM)
      6) Optionally parses the final TOTAL ENERGY from the log

    Example (local)
    ---------------
    sim = GAMESSSimulation(mol)
    res = sim.optimize(
        functional="B3LYP",
        basis_set="6-31G",
        charge=0,
        multiplicity=1,
    )
    print(res["energy_au"], res["logfile"])

    Example (SLURM)
    ---------------
    from symmstate.slurm.slurm_header import SlurmHeader

    header = SlurmHeader(
        job_name="gamess_test",
        partition="compute",
        ntasks=8,
        time="24:00:00",
        output="slurm-%j.out",
        error="slurm-%j.err",
    )

    sim = GAMESSSimulation(mol)
    res = sim.optimize(
        functional="B3LYP",
        basis_set="6-31G",
        charge=0,
        multiplicity=1,
        use_slurm=True,
        slurm_header=header,
        slurm_num_procs=8,
        slurm_monitor=False,  # set True if you want to block + parse energy
    )
    print(res["slurm_job_id"], res["batch_script"])
    """

    def __init__(
        self,
        mol: Chem.Mol,
        gamess_root: str = "/hopper/home/iperez/gamess",
        scratch_parent: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule with a valid 3D conformer.
        gamess_root : str
            Path where GAMESS is installed; must contain gamess.00.x and auxdata/.
        scratch_parent : str, optional
            Parent directory for scratch subdirectories. If None, uses /tmp.
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("RDKit Mol has no conformers; generate 3D coords first.")

        self.mol = mol
        self.gamess_root = os.path.abspath(os.path.expanduser(gamess_root))
        self.gamess_exe = os.path.join(self.gamess_root, "gamess.00.x")
        self.scratch_parent = scratch_parent or "/tmp"

        if not os.path.isfile(self.gamess_exe):
            raise FileNotFoundError(
                f"GAMESS executable not found at: {self.gamess_exe}"
            )
        if not os.access(self.gamess_exe, os.X_OK):
            raise PermissionError(
                f"GAMESS executable is not executable: {self.gamess_exe}"
            )

        self.auxdata_dir = os.path.join(self.gamess_root, "auxdata")
        if not os.path.isdir(self.auxdata_dir):
            print(
                f"[GAMESSSimulation] Warning: auxdata directory not found at "
                f"{self.auxdata_dir}. Some features may fail."
            )

        self.rungms = os.path.join(self.gamess_root, "rungms")
        if not os.path.isfile(self.rungms):
            raise FileNotFoundError(f"rungms script not found at: {self.rungms}")

    def _basis_block(self, basis_set: str) -> str:
        """
        Very minimal mapping from human basis name to GAMESS $BASIS.

        Currently implemented:
            - "6-31G"
            - "6-31G*" / "6-31G(d)" / "6-31G(D)"

        Extend this as needed.
        """
        b = basis_set.upper().replace(" ", "")
        if b == "6-31G":
            # Standard 6-31G
            return " $BASIS GBASIS=N31 NGAUSS=6 $END"
        elif b in ("6-31G*", "6-31G(D)", "6-31G(D)"):
            # 6-31G* -> add d on heavy atoms
            return " $BASIS GBASIS=N31 NGAUSS=6 NDFUNC=1 $END"
        else:
            raise ValueError(
                f"Basis mapping not implemented for '{basis_set}'. "
                "Edit _basis_block to add more."
            )

    def _contrl_block(
        self,
        functional: str,
        charge: int,
        multiplicity: int,
        run_type: str = "OPTIMIZE",
    ) -> str:
        """
        Build a $CONTRL block with DFT.

        NOTE: GAMESS keyword is RUNTYP, not RUNTYPE.
        """
        scftyp = "RHF" if multiplicity == 1 else "UHF"
        fct = functional.upper()
        return (
            f" $CONTRL SCFTYP={scftyp} RUNTYP={run_type} "
            f"DFTTYP={fct} ICHARG={charge} MULT={multiplicity} $END"
        )

    def _system_block(self, mwords: int = 200) -> str:
        return f" $SYSTEM MWORDS={mwords} $END"

    def _scf_block(self) -> str:
        return " $SCF DIRSCF=.TRUE. $END"

    def _statpt_block(self, opttol: float = 1.0e-5, nstep: int = 50) -> str:
        return f" $STATPT OPTTOL={opttol:.1e} NSTEP={nstep} $END"

    def _data_block(self, title: str = "Generated by GAMESSSimulation") -> str:
        """
        Build the $DATA block from the RDKit conformer.

        Uses:
            line 1: $DATA
            line 2: title
            line 3: symmetry (C1)
            remaining lines: "Sym  Z  x  y  z"
        """
        conf = self.mol.GetConformer()
        lines = []
        lines.append(" $DATA")
        lines.append(title)
        lines.append("C1")

        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            sym = atom.GetSymbol()
            Z = atom.GetAtomicNum()
            lines.append(
                f"{sym:2s} {float(Z):6.1f} {pos.x:15.8f} "
                f"{pos.y:15.8f} {pos.z:15.8f}"
            )

        lines.append(" $END")
        return "\n".join(lines)

    def _build_input(
        self,
        functional: str,
        basis_set: str,
        charge: int,
        multiplicity: int,
        run_type: str = "OPTIMIZE",
        mwords: int = 200,
        opttol: float = 1.0e-5,
        nstep: int = 50,
    ) -> str:
        """
        Assemble a complete GAMESS input file as a string.
        """
        blocks = [
            self._contrl_block(functional, charge, multiplicity, run_type),
            self._basis_block(basis_set),
            self._system_block(mwords=mwords),
            self._scf_block(),
            self._statpt_block(opttol=opttol, nstep=nstep),
            self._data_block(),
        ]
        return "\n".join(blocks) + "\n"

    def optimize(
        self,
        functional: str = "B3LYP",
        basis_set: str = "6-31G",
        charge: int = 0,
        multiplicity: int = 1,
        jobname: Optional[str] = None,
        mwords: int = 200,
        opttol: float = 1.0e-5,
        nstep: int = 50,
        keep_scratch: bool = True,
        use_slurm: bool = False,
        slurm_header: Optional["SlurmHeader"] = None,
        slurm_num_procs: int = 8,
        slurm_batch_name: Optional[str] = None,
        slurm_log_name: Optional[str] = None,
        slurm_monitor: bool = False,
        slurm_check_time: int = 60,
    ) -> Dict[str, Any]:
        """
        Run a geometry optimization with GAMESS directly.

        Parameters
        ----------
        functional : str
            DFT functional name (GAMESS DFTTYP).
        basis_set : str
            Basis (currently supports 6-31G and 6-31G*).
        charge : int
            Total molecular charge (ICHARG).
        multiplicity : int
            Spin multiplicity (MULT).
        jobname : str, optional
            Base name for the GAMESS job (input/log/punch). If None, a
            temporary name is generated.
        mwords : int
            MWORDS in $SYSTEM.
        opttol : float
            OPTTOL in $STATPT.
        nstep : int
            NSTEP in $STATPT.
        keep_scratch : bool
            If True, leave the scratch directory for inspection and
            return its path. If False, delete it after parsing.

        SLURM options
        -------------
        use_slurm : bool
            If True, submit the GAMESS job through SLURM using SlurmFile
            instead of running it directly.
        slurm_header : SlurmHeader, optional
            A SlurmHeader instance describing #SBATCH directives. If None,
            a minimal default header is created.
        slurm_num_procs : int
            Number of MPI processes to request in SlurmFile (ntasks).
        slurm_batch_name : str, optional
            Name of the batch script file (inside scratch_dir). Defaults to
            "<jobname>.slurm".
        slurm_log_name : str, optional
            Name of the GAMESS log file. Defaults to "<jobname>.log".
        slurm_monitor : bool
            If True, call SlurmFile.monitor_jobs() and, once finished,
            parse the GAMESS log to extract energy.
        slurm_check_time : int
            Polling interval (seconds) for SlurmFile.monitor_jobs().

        Returns
        -------
        dict

        For local runs (use_slurm=False):
            {
                "mode": "local",
                "returncode": int,
                "energy_au": float or None,
                "scratch_dir": str,
                "input_file": str,
                "logfile": str,
                "punch_file": str,
                "stdout_tail": str,
                "aborted": bool,
            }

        For SLURM runs (use_slurm=True):
            If slurm_monitor=False (submit and return immediately):
                {
                    "mode": "slurm",
                    "slurm_job_id": str or None,
                    "batch_script": str,
                    "scratch_dir": str,
                    "input_file": str,
                    "logfile": str,
                    "punch_file": str,
                    "returncode": None,
                    "energy_au": None,
                    "stdout_tail": "",
                    "aborted": False,
                }

            If slurm_monitor=True (submit, monitor, then parse log):
                Same keys as local run plus "mode": "slurm" and
                "slurm_job_id" and "batch_script".
        """

        # Create a scratch directory
        scratch_parent = os.path.abspath(os.path.expanduser(self.scratch_parent))
        os.makedirs(scratch_parent, exist_ok=True)
        scratch_dir = tempfile.mkdtemp(prefix="gamess_job_", dir=scratch_parent)

        if jobname is None:
            jobname = "job"

        inp_path = os.path.join(scratch_dir, f"{jobname}.inp")

        if slurm_log_name is None:
            slurm_log_name = f"{jobname}.log"
        log_path = os.path.join(scratch_dir, slurm_log_name)

        punch_path = os.path.join(scratch_dir, f"{jobname}.dat")

        # Write the input file
        inp_text = self._build_input(
            functional=functional,
            basis_set=basis_set,
            charge=charge,
            multiplicity=multiplicity,
            run_type="OPTIMIZE",
            mwords=mwords,
            opttol=opttol,
            nstep=nstep,
        )

        with open(inp_path, "w") as f:
            f.write(inp_text)

        # For debugging
        print(f"Content of {inp_path}\n{inp_text}")

        # Common result skeleton
        result: Dict[str, Any] = {
            "mode": "slurm" if use_slurm else "local",
            "returncode": None,
            "energy_au": None,
            "scratch_dir": scratch_dir,
            "input_file": inp_path,
            "logfile": log_path,
            "punch_file": punch_path,
            "stdout_tail": "",
            "aborted": False,
        }

        if use_slurm:
            if SlurmFile is None or SlurmHeader is None:
                raise RuntimeError(
                    "SLURM support requested (use_slurm=True) but "
                    "SlurmHeader/SlurmFile could not be imported."
                )

            if slurm_batch_name is None:
                slurm_batch_name = f"{jobname}.slurm"
            batch_script_path = os.path.join(scratch_dir, slurm_batch_name)

            if slurm_header is None:
                slurm_header = SlurmHeader(
                    job_name=jobname,
                    partition="main",
                    ntasks=slurm_num_procs,
                    time="48:00:00",
                    output="slurm-%j.out",
                    error="slurm-%j.err",
                    additional_lines=[
                        "#SBATCH --nodes=1",
                        "#SBATCH --mem=60gb",
                    ],
                )

            # SLURM will start the job in your current working directory,
            # so we explicitly cd into the scratch_dir before calling rungms.
            mpi_template = textwrap.dedent(
                f"""\
                module purge
                module load openmpi/4.1.6-gcc-11.4.1

                export OMP_NUM_THREADS=1

                # Go to the scratch directory where job.inp lives
                cd {scratch_dir}

                # GAMESS environment expected by rungms
                export GMSPATH={self.gamess_root}
                export SCR={scratch_dir}
                export USERSCR={scratch_dir}
                export AUXDATA={self.auxdata_dir}

                # Run GAMESS via rungms; rungms will take care of INPUT/PUNCH
                {self.rungms} {jobname} 00 {{num_procs}} > {{log}} 2>&1
                """
            )

            slurm_file = SlurmFile(
                slurm_header=slurm_header,
                raw_header=None,
                num_processors=slurm_num_procs,
                mpi_command_template=mpi_template,
            )

            # We  write job.inp as '{jobname}.inp' in scratch_dir,
            # which you already did earlier as 'inp_path'.
            # Now write the batch script there too:
            orig_cwd = os.getcwd()
            try:
                os.chdir(scratch_dir)
                batch_script = slurm_file.write(
                    input_file=inp_path,  # not used in template, but required
                    log_file=log_path,  # this becomes {log}
                    batch_name=os.path.basename(batch_script_path),
                    extra_commands=None,
                )
            finally:
                os.chdir(orig_cwd)

            batch_script_full = os.path.join(
                scratch_dir, os.path.basename(batch_script)
            )
            print(f"[GAMESSSimulation] Wrote SLURM batch script: {batch_script_full}")

            job_id = slurm_file.submit_job(batch_script_full)
            result["slurm_job_id"] = job_id
            result["batch_script"] = batch_script_full

            if slurm_monitor and job_id is not None:
                print(f"[GAMESSSimulation] Waiting for SLURM job {job_id}...")
                slurm_file.wait(check_time=slurm_check_time, check_once=False)

                # Parse GAMESS log produced by rungms
                energy = None
                stdout_tail = ""
                aborted = False
                try:
                    with open(log_path, "r") as f:
                        lines = f.readlines()
                    stdout_tail = "".join(lines[-50:])
                    for line in lines:
                        if "ABNORMALLY" in line:
                            aborted = True
                        if "TOTAL ENERGY =" in line:
                            parts = line.replace("D", "E").split()
                            for token in reversed(parts):
                                try:
                                    energy = float(token)
                                    break
                                except ValueError:
                                    continue
                except Exception as e:
                    stdout_tail = f"<could not read logfile after SLURM run: {e}>"

                result["energy_au"] = energy
                result["stdout_tail"] = stdout_tail
                result["aborted"] = aborted

            if not keep_scratch:
                try:
                    shutil.rmtree(scratch_dir)
                except Exception as e:
                    print(
                        f"[GAMESSSimulation] Warning: could not remove {scratch_dir}: {e}"
                    )

            return result
