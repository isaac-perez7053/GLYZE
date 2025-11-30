#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from glyze.glyceride import FattyAcid, Glyceride
from glyze.glyceride_mix import GlycerideMix
from glyze.gromacs import SimPaths
from glyze.e_viscosity_model import EViscosityModel
from glyze.slurm_header import SlurmHeader
from glyze.slurm_file import SlurmFile


"""
    Build a SlurmFile. Example attached below

        #!/bin/bash
        #SBATCH --job-name=cosaccL0.01
        #SBATCH --partition=main
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=80
        #SBATCH --mem=60gb
        #SBATCH --time=48:00:00
        #SBATCH --output=pp_L001_30ns.log
        #SBATCH --hint=nomultithread
        ...

        srun --cpu-bind=cores gmx mdrun \
          -deffnm pp_L001_30ns \
          -ntmpi ${NTMPI} -ntomp 1 -npme ${NPME} \
          -pin on -dlb yes -nb cpu
"""

header = SlurmHeader(
    job_name="cosacc",     
    partition="main",
    nodes=1,
    ntasks=1,
    cpus_per_task=80,
    time="48:00:00",
    output="gmx_pp_%j.out",    

    additional_lines=[
        "#SBATCH --mem=60gb",
        "#SBATCH --hint=nomultithread",
    ],
    shell_lines=[
        "module purge",
        "export OMP_NUM_THREADS=1",
        "unset OMP_PROC_BIND",
        "unset OMP_PLACES",
        "export GMX_CPUAFFINITY=1",
    ],
)

mpi_command_template = (
    "srun --cpu-bind=cores gmx mdrun "
    "-deffnm {input_file} "
    "-ntmpi {num_procs} -ntomp 1 -npme 10 "
    "-pin on -dlb yes -nb cpu > {log}"
)

slurm = SlurmFile(
    slurm_header=header,
    raw_header=None,
    num_processors=80,
    mpi_command_template=mpi_command_template,
)


def main():

    palmitin = FattyAcid(length=16)
    tripalmitin = Glyceride(sn=(palmitin, palmitin, palmitin))

    olein = FattyAcid(length=18, db_positions=(9,), db_stereo=("Z",))
    triolein = Glyceride(sn=(olein, olein, olein))

    mix = GlycerideMix(mix=[(triolein, 0.5), (tripalmitin, 0.5)])

    fchk_paths = [
        Path("fchks/G_N18D01P09Z_N18D01P09Z_N18D01P09Z.fchk"),
        Path("fchks/G_N16D00_N16D00_N16D00_opt.fchk"),
    ]
    pdb_paths = [
        Path("fchks/G_N18D01P09Z_N18D01P09Z_N18D01P09Z.pdb"),
        Path("fchks/G_N16D00_N16D00_N16D00_opt.pdb"),
    ]
    resname_map = {triolein: "TRIO", tripalmitin: "TRIP"}

    paths = SimPaths(root=Path("runs/"), ffdir=Path("oplsaa.ff"), name=mix.name)

    model = EViscosityModel(
        mix=mix,
        paths=paths,
        fchk_paths=fchk_paths,
        pdb_paths=pdb_paths,
        resname_map=resname_map,
        # multiwfn_exe="multiwfn",
        # multiwfn_script="cm5_menu.txt",
        # gmx_bin="gmx",
        # gmxlib="/path/to/share/gromacs/top",
        # packmol_exe="packmol",
    )

    model.run_multiwfn_and_build_tops()
    model.prepare_initial_configuration(num_molecules=20, density_g_per_cm3=0.5)
    model.build_gromacs_system(num_molecules=20)
    model.run_energy_minimization()
    model.run_npt_equilibration(T=300.0, P=1.0, ns=2.0, slurm=slurm)
    model.run_nvt_equilibration(T=300.0, ns=2.0, slurm=slurm)

    # Run viscosity sweep 
    pp_runs = model.run_pp_viscosity_sweep(
        T=300.0,
        ns=30.0,
        A_min=0.010,
        A_max=0.020,
        num_datapoints=6,
        slurm=slurm,
    )

    # Analyze the final results
    model.analyze_results(pp_runs=pp_runs, ns_per_A=30.0, T=300.0)


if __name__ == "__main__":
    main()


