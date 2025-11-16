from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

# Physical constants
AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1.0e-24


class PackmolSimulator:
    """
    Small helper around Packmol: manages the executable path, builds input
    files, computes integer species counts from relative concentrations, and
    executes Packmol.
    """

    def __init__(self, packmol_path: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        packmol_path
            Path to the Packmol executable. Can be relative and/or use '~'.
            If None, 'packmol' from the environment PATH will be used.
        """
        if packmol_path is None:
            self.packmol_executable: Optional[Path] = None
        else:
            self.packmol_executable = Path(packmol_path).expanduser().resolve()

    def estimate_cubic_box_length(
        self,
        total_molecules: int,
        density_g_per_cm3: float = 0.9,
        *,
        average_molar_mass_g_per_mol: Optional[float] = None,
        total_mass_g: Optional[float] = None,
    ) -> float:
        """
        Estimate cubic box length L (Å) from number of molecules and density.

        Physics: rho = m / V, V = L^3.

        You must provide either:
          - total_mass_g directly, OR
          - average_molar_mass_g_per_mol, in which case:
                m = (N * MW) / N_A.
        """
        if total_molecules <= 0:
            raise ValueError("total_molecules must be positive.")
        if density_g_per_cm3 <= 0:
            raise ValueError("density_g_per_cm3 must be positive.")

        # Determine total mass (g)
        if total_mass_g is None:
            if average_molar_mass_g_per_mol is None:
                raise ValueError(
                    "You must provide either total_mass_g or "
                    "average_molar_mass_g_per_mol."
                )
            total_mass_g = (
                float(total_molecules) * float(average_molar_mass_g_per_mol) / AVOGADRO
            )
        else:
            total_mass_g = float(total_mass_g)

        if total_mass_g <= 0:
            raise ValueError("Computed total_mass_g must be positive.")

        # Volume in cm^3 from density
        volume_cm3 = total_mass_g / float(density_g_per_cm3)

        # Convert to ang^3 and then to box length
        volume_A3 = volume_cm3 / CM3_PER_A3
        L = float(volume_A3 ** (1.0 / 3.0))
        return L

    def estimate_cubic_box_length_from_species(
        self,
        counts: Sequence[int],
        molar_masses_g_per_mol: Sequence[float],
        density_g_per_cm3: float = 0.9,
    ) -> float:
        """
        Convenience variant: compute L (Å) directly from per-species counts
        and molar masses.

        For each species i:
            m_i = n_i * MW_i / N_A  [g]
        and  m_total = sum_i m_i.
        Then:
            rho = m_total / V,  V = L^3.

        Parameters
        ----------
        counts
            Number of molecules of each species (same order as molar_masses).
        molar_masses_g_per_mol
            Molar mass for each species in g/mol, same length as counts.
        density_g_per_cm3
            Target density (default 0.9 g/cm^3 for glycerides).

        Returns
        -------
        float
            Box length L in Å for a cubic box.
        """
        if len(counts) != len(molar_masses_g_per_mol):
            raise ValueError(
                "counts and molar_masses_g_per_mol must have the same length."
            )

        if not counts:
            raise ValueError("counts must be a non-empty sequence.")

        if any(n < 0 for n in counts):
            raise ValueError("All counts must be non-negative.")

        if all(n == 0 for n in counts):
            raise ValueError("At least one species must have a positive count.")

        if any(MW <= 0 for MW in molar_masses_g_per_mol):
            raise ValueError("All molar masses must be positive.")

        total_molecules = int(sum(counts))

        # Total mass in grams
        total_mass_g = 0.0
        for n, MW in zip(counts, molar_masses_g_per_mol):
            if n == 0:
                continue
            total_mass_g += float(n) * float(MW) / AVOGADRO

        return self.estimate_cubic_box_length(
            total_molecules=total_molecules,
            density_g_per_cm3=density_g_per_cm3,
            total_mass_g=total_mass_g,
        )

    def execute_packmol(
        self,
        input_file: Union[str, Path],
        use_instance_path: bool = True,
        cwd: Optional[Union[str, Path]] = None,
        capture_output: bool = True,
        check: bool = True,
        timeout: Optional[float] = None,
        env: Optional[dict] = None,
        treat_173_as_success: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Execute Packmol on a given input file.

        Uses either:
        - the stored absolute path to the executable (if provided), or
        - 'packmol' from the environment PATH.

        Parameters
        ----------
        input_file
            Path to the Packmol input file.
        use_instance_path
            If True and an instance path was provided, use it; else use 'packmol'.
        cwd
            Working directory for the Packmol run.
        capture_output
            If True, capture stdout/stderr in the CompletedProcess.
        check
            If True, raise on non-zero exit code, *except* error 173 if
            `treat_173_as_success` is True.
        timeout
            Optional timeout in seconds.
        env
            Optional environment dict.
        treat_173_as_success
            If True, treat Packmol runs that end with error/STOP 173 as
            “successful” and do not raise, even if returncode != 0.
            (You can then handle `output.pdb_FORCED` etc. at a higher level.)

        Returns
        -------
        subprocess.CompletedProcess
            The result from subprocess.run (possibly with non-zero returncode).
        """
        input_file = Path(input_file)

        # Decide which executable to use
        if use_instance_path and self.packmol_executable is not None:
            exe = str(self.packmol_executable)
        else:
            exe = "packmol"  # rely on environment PATH

        if cwd is not None:
            cwd = Path(cwd).expanduser().resolve()

        # Run without automatic check so we can treat 173 specially
        with input_file.open("rb") as f_in:
            result = subprocess.run(
                [exe],
                stdin=f_in,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=str(cwd) if cwd is not None else None,
                check=False,  # we'll handle error checking manually
                timeout=timeout,
                env=env,
            )

        # Manual "check" logic with special handling for error 173
        if check and result.returncode != 0:
            # Default: assume not a 173 situation
            is_173 = False

            if capture_output:
                # Safely decode; ignore errors to avoid Unicode issues
                stdout_text = (
                    result.stdout.decode(errors="ignore") if result.stdout else ""
                )
                stderr_text = (
                    result.stderr.decode(errors="ignore") if result.stderr else ""
                )
                combined = (stdout_text + "\n" + stderr_text).lower()

                # Heuristics for detecting Packmol's error 173 messages
                if (
                    "stop 173" in combined
                    or "error 173" in combined
                    or "errorcode 173" in combined
                ):
                    is_173 = True

            if not (treat_173_as_success and is_173):
                # Behave like check=True: raise CalledProcessError
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=[exe],
                    output=result.stdout,
                    stderr=result.stderr,
                )

        # Either success, or tolerated error 173
        return result

    def compute_particle_counts(
        self,
        total_molecules: int,
        concentrations: Sequence[float],
    ) -> List[int]:
        """
        Convert relative concentrations into integer counts that sum to
        `total_molecules`.
        """
        if total_molecules <= 0:
            raise ValueError("total_molecules must be positive.")

        if not concentrations:
            raise ValueError("concentrations must be a non-empty sequence.")

        if any(c < 0 for c in concentrations):
            raise ValueError("All concentrations must be non-negative.")

        total_conc = float(sum(concentrations))
        if total_conc <= 0.0:
            raise ValueError("Sum of concentrations must be positive.")

        # Normalize to fractions
        fractions = [c / total_conc for c in concentrations]

        # Ideal real-valued counts
        real_counts = [f * total_molecules for f in fractions]

        # Base counts via floor
        base_counts = [int(math.floor(x)) for x in real_counts]
        assigned = sum(base_counts)
        remainder = total_molecules - assigned

        if remainder < 0:
            raise RuntimeError("Remainder negative; check concentrations logic.")

        if remainder > 0:
            residuals = [x - b for x, b in zip(real_counts, base_counts)]
            order = sorted(
                range(len(residuals)),
                key=lambda i: residuals[i],
                reverse=True,
            )
            for i in order[:remainder]:
                base_counts[i] += 1

        if sum(base_counts) != total_molecules:
            raise RuntimeError(
                "Integerization error: counts do not sum to total_molecules."
            )

        return base_counts

    def build_input_file(
        self,
        structure_files: Sequence[Union[str, Path]],
        counts: Sequence[int],
        box_lengths: Union[float, Sequence[float]],
        tolerance: float = 2.0,
        filetype: str = "pdb",
        output: str = "output.pdb",
        input_filename: Union[str, Path] = "packmol.inp",
        extra_lines: Optional[Iterable[str]] = None,
        workdir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Build a Packmol input file (tolerance, filetype, structure blocks, output).
        """
        if len(structure_files) != len(counts):
            raise ValueError("structure_files and counts must have the same length.")

        if isinstance(box_lengths, (int, float)):
            Lx = Ly = Lz = float(box_lengths)
        else:
            bl = list(box_lengths)
            if len(bl) != 3:
                raise ValueError(
                    "box_lengths must be a single float (cubic) or an "
                    "iterable of three floats (Lx, Ly, Lz)."
                )
            Lx, Ly, Lz = map(float, bl)

        if workdir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(workdir).expanduser().resolve()

        input_path = base_dir / input_filename
        output_str = str(output)

        # Normalize structure paths
        structure_paths: List[str] = []
        for sf in structure_files:
            p = Path(sf).expanduser()
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            structure_paths.append(str(p))

        lines: List[str] = []

        # Global settings
        lines.append(f"tolerance {float(tolerance)}")
        lines.append("")
        lines.append(f"filetype {filetype}")
        lines.append("")

        # Extra keywords (optional)
        if extra_lines is not None:
            for line in extra_lines:
                line = str(line).rstrip("\n")
                if line:
                    lines.append(line)
            lines.append("")

        # Per-structure blocks
        for struct, n in zip(structure_paths, counts):
            if n <= 0:
                continue
            lines.append(f"structure {struct}")
            lines.append(f"  number {int(n)}")
            lines.append(f"  inside box 0. 0. 0. {Lx:.6f} {Ly:.6f} {Lz:.6f}")
            lines.append("end structure")
            lines.append("")

        # Output line
        lines.append(f"output {output_str}")
        lines.append("")

        input_path.write_text("\n".join(lines))
        return input_path


if __name__ == "__main__":
    sim = PackmolSimulator("~/bin/packmol")

    pdb_files = ["MDAPMinput0.pdb", "MDAPMinput1.pdb"]

    # Say you already computed integer counts from your mix
    concentrations = [1.0, 1.0]  # relative concentrations
    total_mols = 20

    counts = sim.compute_particle_counts(total_mols, concentrations)
    molar_masses = [860.0, 890.0]  # g/mol for each glyceride species (example)

    L = sim.estimate_cubic_box_length_from_species(
        counts=counts,
        molar_masses_g_per_mol=molar_masses,
        density_g_per_cm3=0.9,
    )

    inp = sim.build_input_file(
        structure_files=pdb_files,
        counts=counts,
        box_lengths=L,
        tolerance=2.0,
        filetype="pdb",
        output="output.pdb",
    )

    result = sim.execute_packmol(inp)
    print("Return code:", result.returncode)
    print(result.stdout.decode() if result.stdout else "")
