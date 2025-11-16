from pathlib import Path
from glyze import * 

# Define your species
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

paths = SimPaths(root=Path(f"runs/"), ffdir=Path("oplsaa.ff"), name=mix.name)

model = EViscosityModel(
    mix=mix,
    paths=paths,
    fchk_paths=fchk_paths,
    pdb_paths=pdb_paths,
    resname_map=resname_map,
    multiwfn_exe="multiwfn",
    multiwfn_script="cm5_menu.txt",
    gmx_bin=None,  # or explicit path to gmx
    gmxlib=None,  # or explicit GMXLIB
    packmol_exe=None,  # or explicit path
)

# 1) CM5 + tops
model.run_multiwfn_and_build_tops()

# 2) Packmol + init.pdb
model.prepare_initial_configuration(num_molecules=20, density_g_per_cm3=0.5)

# 3) editconf + system.top
model.build_gromacs_system(num_molecules=20)

# 4) Energy minimization
model.run_energy_minimization()

# 5) NVT
model.run_nvt_equilibration(T=300.0, P=1.0, ns=2.0)

# 6) NPT
model.run_npt_equilibration(T=300.0, ns=2.0)

# Production (if desired)
prod_edr = model.run_production(T=300.0, ns=10.0)


