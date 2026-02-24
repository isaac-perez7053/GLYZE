from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional, List, Dict, Tuple, Union
import re
from pathlib import Path
from typing import Callable
import os
import subprocess
import numpy as np
import warnings

Number = Union[int, float]
PathLike = Union[str, Path]


@dataclass
class PPViscosityPoint:
    """Single PP run at one acceleration amplitude L."""

    L: float
    inv_eta: float
    inv_eta_sem: float
    eta_cP: float
    eta_cP_sem: float
    temp_mean_K: Optional[float] = None
    temp_drift_K_per_ns: Optional[float] = None
    source_xvg: Optional[Path] = None
    temp_xvg: Optional[Path] = None


@dataclass
class PPViscosityFitResult:
    """Summary of a PP viscosity fit over several amplitudes."""

    points: List[PPViscosityPoint]

    # Fit in eta(L) = eta0 + m L, with eta in cP
    eta0_cP: float
    eta0_cP_err: float
    slope_cP_per_L: float
    slope_cP_per_L_err: float
    r2: float

    # Leave-one-out (“dropout”) intercepts in cP
    loo_eta0_cP: List[float]
    loo_eta0_cP_err: List[float]

    qc_warnings: List[str]

    # Convenience flag
    passed_qc: bool


class GromacsViscosityAnalysis:
    """
    Analysis utilities for GROMACS Periodic Perturbation (PP) viscosity runs.

    The main entrypoint for viscosity is :meth:`analyze_pp_from_xvg`, which:

    - reads a set of inverse-viscosity `.xvg` files (and optional temperature files),
    - block-averages each time series to obtain mean 1/inv_eta and SEM,
    - converts to viscosity eta in cP (with propagated SEM),
    - fits a weighted linear model eta(L) = eta0 + m L,
    - performs a leave-one-out (“dropout”) analysis for eta0, and
    - runs a few sanity / QC checks (temperature stability, fit quality, etc.).
    """

    def analyze_pp_temperature_dir(
        self,
        pp_temp_dir: PathLike,
        *,
        inveta_glob: str = "*visc.xvg",
        temp_suffix: str = "_temp.xvg",
        amplitude_parser: Optional[Callable[[Path], float]] = None,
        tmin_ns: float = 0.0,
        tmax_ns: Optional[float] = None,
        signal_col: int = 1,
        temp_col: int = 1,
        block_ns: Optional[float] = None,
        nblocks: Optional[int] = None,
        target_temperature_K: float = 300.0,
        temp_tolerance_K: float = 0.2,
        max_temp_drift_K_per_ns: float = 0.1,
        max_rel_sem_inveta: float = 0.04,
        r2_threshold: float = 0.95,
        max_dropout_delta_pct: float = 10.0,
    ) -> PPViscosityFitResult:
        """
        Analyze PP viscosity runs under a given temperature directory.

        Parameters
        ----------
        pp_temp_dir
            Directory for a given temperature, e.g. ".../04_prod/pp_300K".
            All subdirectories / files beneath this will be scanned for
            inverse-viscosity XVG files matching ``inveta_glob``.

        inveta_glob
            Glob pattern (relative to ``pp_temp_dir``) that selects 1/eta .xvg
            files. Default assumes files like "*visc.xvg".

        temp_suffix
            Suffix used to derive the temperature .xvg path from the 1/eta path.
            For example, if an inverse-viscosity file is "pp_A0p0125_30ns_visc.xvg"
            and ``temp_suffix="_temp.xvg"``, the temperature file is assumed to be
            "pp_A0p0125_30ns_temp.xvg" in the same directory.

        amplitude_parser
            Optional function that takes a Path and returns L (float).
            If None, a default parser (:meth:`_infer_pp_amplitude_from_path`)
            is used, which expects something like "A_0p0125" or "L_0p0125"
            in either the directory name or filename.

        All remaining keyword arguments are passed through to
        :meth:`analyze_pp_from_xvg`.

        Returns
        -------
        PPViscosityFitResult
            Same structured result as :meth:`analyze_pp_from_xvg`.
        """
        base = Path(pp_temp_dir)
        inv_paths = sorted(base.rglob(inveta_glob))

        if not inv_paths:
            raise RuntimeError(
                f"No inverse-viscosity XVG files found in {base} "
                f"with pattern '{inveta_glob}'."
            )

        if amplitude_parser is None:
            amplitude_parser = self._infer_pp_amplitude_from_path

        L_values: List[float] = []
        inveta_xvg: List[Path] = []
        temp_xvg: List[Path] = []

        for inv_path in inv_paths:
            inv_path = inv_path.resolve()

            # Infer amplitude L
            L = float(amplitude_parser(inv_path))

            # Construct matching temperature path
            name = inv_path.name
            if name.endswith("visc.xvg"):
                temp_name = name.replace("visc.xvg", temp_suffix.lstrip("_"))
            else:
                # Fallback: just replace suffix with temp_suffix
                stem = inv_path.stem
                temp_name = stem + temp_suffix

            temp_path = inv_path.with_name(temp_name)

            if not temp_path.exists():
                raise RuntimeError(
                    f"Expected temperature XVG file '{temp_path}' "
                    f"for inverse-viscosity file '{inv_path}', but it does not exist."
                )

            L_values.append(L)
            inveta_xvg.append(inv_path)
            temp_xvg.append(temp_path)

        return self.analyze_pp_from_xvg(
            L_values=L_values,
            inveta_xvg=inveta_xvg,
            temp_xvg=temp_xvg,
            tmin_ns=tmin_ns,
            tmax_ns=tmax_ns,
            signal_col=signal_col,
            temp_col=temp_col,
            block_ns=block_ns,
            nblocks=nblocks,
            target_temperature_K=target_temperature_K,
            temp_tolerance_K=temp_tolerance_K,
            max_temp_drift_K_per_ns=max_temp_drift_K_per_ns,
            max_rel_sem_inveta=max_rel_sem_inveta,
            r2_threshold=r2_threshold,
            max_dropout_delta_pct=max_dropout_delta_pct,
        )

    def analyze_pp_from_xvg(
        self,
        L_values: Sequence[float],
        inveta_xvg: Sequence[PathLike],
        temp_xvg: Optional[Sequence[PathLike]] = None,
        *,
        tmin_ns: float = 0.0,
        tmax_ns: Optional[float] = None,
        signal_col: int = 1,
        temp_col: int = 1,
        block_ns: Optional[float] = None,
        nblocks: Optional[int] = None,
        target_temperature_K: float = 300.0,
        temp_tolerance_K: float = 0.2,
        max_temp_drift_K_per_ns: float = 0.1,
        max_rel_sem_inveta: float = 0.04,
        r2_threshold: float = 0.95,
        max_dropout_delta_pct: float = 10.0,
    ) -> PPViscosityFitResult:
        """
        Analyze PP viscosity from lists of amplitudes and 1/eta `.xvg` files.

        Parameters
        ----------
        L_values
            Sequence of PP acceleration amplitudes L (e.g. in nm*ps^-2),
            one per simulation.

        inveta_xvg
            Sequence of inverse-viscosity `.xvg` files produced by ``gmx energy``.
            Must be the same length as ``L_values``; ``inveta_xvg[i]`` is the
            file corresponding to amplitude ``L_values[i]``.

        temp_xvg
            Optional sequence of temperature `.xvg` files, one per amplitude.
            If provided, must have the same length as ``L_values``; the
            i-th temperature file will be associated with the i-th L and
            inverse-viscosity file.

        tmin_ns, tmax_ns
            Analysis window in ns.  All samples with time < ``tmin_ns`` are
            discarded (e.g. to skip warm-in); if ``tmax_ns`` is not ``None``,
            samples with time > ``tmax_ns`` are discarded as well.

        signal_col
            Column index in the signal `.xvg` files for 1/eta values
            (0 = time, 1 = first y-column, etc.).

        temp_col
            Column index in the temperature `.xvg` files for T[K].

        block_ns, nblocks
            Block-averaging settings for 1/eta(t):

            - If ``block_ns`` is given, the trajectory window is divided into
              as many blocks of length ``block_ns`` as will fit.
            - If ``nblocks`` is given instead, the window is divided into
              ``nblocks`` equal-length blocks.
            - If both are ``None``, a default of 6 blocks is used.

            For each block, the mean 1/eta is computed; the overall mean and SEM
            are taken over the block means.

        target_temperature_K
            Expected target temperature (e.g. 300 K).  Used only for QC.

        temp_tolerance_K
            QC threshold for temperature stability.  If any PP run has a mean
            temperature |T_mean - target_temperature_K| larger than this
            tolerance, a warning is recorded.

        max_temp_drift_K_per_ns
            QC threshold for |dT/dt| from a linear fit T(t) = a + b t.  If any
            run has |b| > ``max_temp_drift_K_per_ns``, a warning is recorded.

        max_rel_sem_inveta
            QC threshold on the per-amplitude relative SEM of 1/eta.  If any run
            has ``inv_eta_sem / |inv_eta|`` larger than this fraction, a
            warning is recorded.

        r2_threshold
            Minimal acceptable R^2 for the linear regression eta(L) = eta0 + m L.
            If the fitted R^2 is below this threshold, a warning is recorded.

        max_dropout_delta_pct
            QC threshold on intercept stability under leave-one-out.  If any
            leave-one-out eta0 deviates from the full-fit eta0 by more than this
            percentage, a warning is recorded.

        Returns
        -------
        PPViscosityFitResult
            Structured result containing per-amplitude points, the fitted eta0 and
            slope (with uncertainties), leave-one-out intercepts, and a list of
            QC warnings.  The ``passed_qc`` boolean is ``True`` if no warnings
            were generated.
        """
        # Basic input validation
        if len(L_values) != len(inveta_xvg):
            raise ValueError(
                f"L_values and inveta_xvg must have the same length "
                f"(got {len(L_values)} vs {len(inveta_xvg)})."
            )
        if temp_xvg is not None and len(temp_xvg) != len(L_values):
            raise ValueError(
                f"When provided, temp_xvg must have the same length as L_values "
                f"(got {len(temp_xvg)} vs {len(L_values)})."
            )

        # Prepare points
        points: List[PPViscosityPoint] = []

        for i, (L, sig_file) in enumerate(zip(L_values, inveta_xvg)):
            L = float(L)
            sig_path = Path(sig_file)

            # Parse inverse-viscosity time series and block-average
            t_ns, y = self._parse_xvg(sig_path, ycol=signal_col)
            inv_eta_mean, inv_eta_sem, _, _ = self._block_stats(
                t_ns,
                y,
                tmin_ns=tmin_ns,
                tmax_ns=tmax_ns,
                block_ns=block_ns,
                nblocks=nblocks,
            )

            # Optional temperature stats
            temp_mean = None
            temp_drift = None
            temp_path: Optional[Path] = None

            if temp_xvg is not None:
                temp_path = Path(temp_xvg[i])
                tT_ns, T = self._parse_xvg(temp_path, ycol=temp_col)
                temp_mean, temp_drift = self._temp_stats(
                    tT_ns, T, tmin_ns=tmin_ns, tmax_ns=tmax_ns
                )

            # Convert to viscosity eta with error propagation; then to cP
            eta_SI = 1.0 / inv_eta_mean
            eta_SI_sem = inv_eta_sem / (inv_eta_mean**2)
            eta_cP = 1000.0 * eta_SI
            eta_cP_sem = 1000.0 * eta_SI_sem

            points.append(
                PPViscosityPoint(
                    L=L,
                    inv_eta=inv_eta_mean,
                    inv_eta_sem=inv_eta_sem,
                    eta_cP=eta_cP,
                    eta_cP_sem=eta_cP_sem,
                    temp_mean_K=temp_mean,
                    temp_drift_K_per_ns=temp_drift,
                    source_xvg=sig_path,
                    temp_xvg=temp_path,
                )
            )

        # Sort by L
        points.sort(key=lambda p: p.L)

        # Assemble arrays for fit
        L_vals = np.array([p.L for p in points], dtype=float)
        eta_vals = np.array([p.eta_cP for p in points], dtype=float)
        eta_sem_vals = np.array([p.eta_cP_sem for p in points], dtype=float)

        # Weighted linear regression: eta(L) = a + b L
        a, b, a_err, b_err, r2 = self._weighted_linreg(L_vals, eta_vals, eta_sem_vals)

        eta0_cP = float(a)
        eta0_cP_err = float(a_err)
        slope = float(b)
        slope_err = float(b_err)

        # Leave-one-out intercepts
        loo_eta0: List[float] = []
        loo_eta0_err: List[float] = []
        if len(points) >= 3:
            for i in range(len(points)):
                mask = np.ones(len(points), dtype=bool)
                mask[i] = False
                a_i, _, a_i_err, _, _ = self._weighted_linreg(
                    L_vals[mask],
                    eta_vals[mask],
                    eta_sem_vals[mask],
                )
                loo_eta0.append(float(a_i))
                loo_eta0_err.append(float(a_i_err))

        qc_messages: List[str] = []

        #  R^2 gate check
        if r2 < r2_threshold:
            qc_messages.append(
                f"Linear eta(L) fit has R^2={r2:.3f}, below threshold {r2_threshold:.3f}."
            )

        # Temperature stability and drift check
        for p in points:
            if p.temp_mean_K is not None:
                if abs(p.temp_mean_K - target_temperature_K) > temp_tolerance_K:
                    qc_messages.append(
                        f"L={p.L:g}: mean T={p.temp_mean_K:.3f} K deviates from "
                        f"target {target_temperature_K:.3f} K by more than "
                        f"{temp_tolerance_K:.3f} K."
                    )
            if p.temp_drift_K_per_ns is not None:
                if abs(p.temp_drift_K_per_ns) > max_temp_drift_K_per_ns:
                    qc_messages.append(
                        f"L={p.L:g}: |dT/dt|={p.temp_drift_K_per_ns:.4f} K/ns exceeds "
                        f"threshold {max_temp_drift_K_per_ns:.4f} K/ns."
                    )

        # Relative SEM on 1/eta
        for p in points:
            rel_sem = p.inv_eta_sem / abs(p.inv_eta)
            if rel_sem > max_rel_sem_inveta:
                qc_messages.append(
                    f"L={p.L:g}: relative SEM(1/eta)={100*rel_sem:.1f}% exceeds "
                    f"{100*max_rel_sem_inveta:.1f}%."
                )

        # Dropout stability of eta0
        if loo_eta0 and eta0_cP != 0.0:
            deltas_pct = [
                abs(eta_i - eta0_cP) / abs(eta0_cP) * 100.0 for eta_i in loo_eta0
            ]
            max_delta = max(deltas_pct)
            if max_delta > max_dropout_delta_pct:
                qc_messages.append(
                    f"Leave-one-out eta0 varies by up to {max_delta:.1f}% "
                    f"(threshold {max_dropout_delta_pct:.1f}%)."
                )

        # Physical sanity: positive viscosity
        if eta0_cP <= 0.0:
            qc_messages.append(
                f"Extrapolated eta0={eta0_cP:.3f} cP is non-positive; fit is not physical."
            )

        # Emit runtime warnings (does NOT stop the script)
        for msg in qc_messages:
            warnings.warn(msg, RuntimeWarning)

        passed_qc = len(qc_messages) == 0

        return PPViscosityFitResult(
            points=points,
            eta0_cP=eta0_cP,
            eta0_cP_err=eta0_cP_err,
            slope_cP_per_L=slope,
            slope_cP_per_L_err=slope_err,
            r2=r2,
            loo_eta0_cP=loo_eta0,
            loo_eta0_cP_err=loo_eta0_err,
            qc_warnings=qc_messages,
            passed_qc=passed_qc,
        )

    def extract_energy_terms(
        self,
        edr_file: PathLike,
        terms: Optional[Sequence[str]] = None,
        *,
        gmx_command: str = "gmx",
        time_unit: str = "ps",
        working_directory: Optional[PathLike] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run ``gmx energy`` on an ``.edr`` file and return selected time series.

        Parameters
        ----------
        edr_file
            Path to the GROMACS energy file (.edr).
        terms
            Sequence of energy terms to extract. Each entry can be either
            a menu index (e.g. "3") or a substring of the term label
            (e.g. "Temperature", "1/Viscosity"). If None, all terms in the
            menu are extracted.
        gmx_command
            Command used to invoke GROMACS, e.g. "gmx" or an absolute path.
        time_unit
            "ps" or "ns"; controls the time column of the returned arrays.
            Internally, .xvg files are parsed as ns and converted here.
        working_directory
            Directory from which to run ``gmx energy``. If None, uses the
            directory containing ``edr_file``.
        extra_args
            Optional extra arguments to append to the ``gmx energy`` command.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from term label (as printed by ``gmx energy``) to a
            2-column NumPy array of shape (N, 2) with columns [time, value].
        """
        # Find the .edr file path
        edr_path = Path(edr_file).expanduser().resolve()

        if working_directory is None:
            workdir = edr_path.parent
            edr_arg = edr_path.name
        else:
            workdir = Path(working_directory).expanduser().resolve()
            # Use absolute path to the .edr if we're not in its directory
            edr_arg = str(edr_path)

        if extra_args is None:
            extra_args = []

        env = os.environ.copy()
        probe_out = workdir / "touch.xvg"
        probe_cmd = [
            gmx_command,
            "energy",
            "-f",
            edr_arg,
            "-o",
            probe_out.name,
            *extra_args,
        ]

        # Extract gmx energy options by running program
        probe_proc = subprocess.run(
            probe_cmd,
            cwd=str(workdir),
            env=env,
            input="0\n",  # immediately quit without selecting terms
            text=True,
            capture_output=True,
        )

        if probe_proc.returncode != 0 and not probe_proc.stdout:
            raise RuntimeError(
                f"Failed to run 'gmx energy' to probe terms:\n"
                f"STDOUT:\n{probe_proc.stdout}\nSTDERR:\n{probe_proc.stderr}"
            )

        menu: Dict[int, str] = {}
        for line in probe_proc.stdout.splitlines():
            m = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
            if m:
                idx = int(m.group(1))
                label = m.group(2).strip()
                menu[idx] = label

        # Clean up the temporary probe file if it exists
        try:
            probe_out.unlink()
        except FileNotFoundError:
            pass

        if not menu:
            raise RuntimeError(
                "Could not parse energy term menu from 'gmx energy' output. "
                "Check that gmx_command is correct and the .edr file is valid."
            )

        if terms is None:
            requested_indices = sorted(menu.keys())
        else:
            requested_indices: List[int] = []
            for t in terms:
                s = str(t).strip()
                if s.isdigit():
                    idx = int(s)
                    if idx not in menu:
                        raise ValueError(
                            f"Requested index {idx} not found in energy term menu."
                        )
                    requested_indices.append(idx)
                else:
                    # substring match on label (case-insensitive)
                    matches = [i for i, lbl in menu.items() if s.lower() in lbl.lower()]
                    if not matches:
                        raise ValueError(
                            f"No energy term label matching '{s}' found in menu."
                        )
                    if len(matches) > 1:
                        raise ValueError(
                            f"Ambiguous energy term '{s}' matches multiple labels: "
                            + ", ".join(f"{i}:{menu[i]}" for i in matches)
                        )
                    requested_indices.append(matches[0])

        if time_unit not in {"ps", "ns"}:
            raise ValueError("time_unit must be 'ps' or 'ns'.")

        data: Dict[str, np.ndarray] = {}

        for idx in requested_indices:
            label = menu[idx]
            # Use pregenerated name for the output file
            safe_label = re.sub(r"\s+", "_", label)
            out_name = f"{edr_path.stem}_{safe_label}.xvg"
            out_path = workdir / out_name

            cmd = [
                gmx_command,
                "energy",
                "-f",
                edr_arg,
                "-o",
                out_name,
                *extra_args,
            ]

            # Select the desired term index, then finish with 0
            selection = f"{idx}\n0\n"
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                input=selection,
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"'gmx energy' failed for term {idx} ({label}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            # Reuse existing XVG parser (returns time in ns)
            t_ns, y = self._parse_xvg(out_path, ycol=1)
            if time_unit == "ps":
                t = t_ns * 1000.0
            else:
                t = t_ns

            arr = np.column_stack((t, y))
            data[label] = arr

        return data

    @staticmethod
    def _parse_xvg(filepath: PathLike, ycol: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a GROMACS .xvg file and return (t_ns, y) arrays."""
        path = Path(filepath)
        t: List[float] = []
        y: List[float] = []
        time_units: Optional[str] = None  # 'ps' or 'ns'

        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                if line.startswith(("@", "#")):
                    ls = line.lower()
                    if "xaxis" in ls and "label" in ls:
                        if "(ns)" in ls:
                            time_units = "ns"
                        elif "(ps)" in ls:
                            time_units = "ps"
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) <= ycol:
                    continue
                try:
                    t_val = float(parts[0])
                    y_val = float(parts[ycol])
                except ValueError:
                    continue
                t.append(t_val)
                y.append(y_val)

        if not t:
            raise RuntimeError(f"No numeric data found in {path}")

        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if time_units is None:
            time_units = "ps"
        if time_units == "ps":
            t_ns = t_arr * 0.001
        else:
            t_ns = t_arr

        return t_ns, y_arr

    @staticmethod
    def _block_stats(
        t_ns: np.ndarray,
        y: np.ndarray,
        *,
        tmin_ns: float,
        tmax_ns: Optional[float],
        block_ns: Optional[float],
        nblocks: Optional[int],
    ) -> Tuple[float, float, int, float]:
        """
        Block-average y(t) over [tmin_ns, tmax_ns] and return mean & SEM of block means.
        """
        mask = t_ns >= tmin_ns
        if tmax_ns is not None:
            mask &= t_ns <= tmax_ns

        t2 = t_ns[mask]
        y2 = y[mask]
        if t2.size < 2:
            raise RuntimeError("Not enough points after tmin/tmax filtering.")

        tstart = t2.min()
        tstop = t2.max()
        total = tstop - tstart
        if total <= 0:
            raise RuntimeError("Non-positive analysis window; check tmin/tmax.")

        if block_ns is not None and nblocks is not None:
            raise ValueError("Use either block_ns or nblocks, not both.")

        if block_ns is None and nblocks is None:
            nblocks = 6

        if block_ns is not None:
            nblocks = max(int(np.floor(total / block_ns)), 1)
        else:
            block_ns = total / nblocks

        means: List[float] = []
        for i in range(nblocks):
            a = tstart + i * block_ns
            b = tstart + (i + 1) * block_ns if i < nblocks - 1 else tstop
            m = (t2 >= a) & (t2 <= b)
            yy = y2[m]
            if yy.size == 0:
                continue
            means.append(float(np.mean(yy)))

        if not means:
            raise RuntimeError("No data in any block; check settings.")

        means_arr = np.asarray(means, dtype=float)
        inv_eta_mean = float(np.mean(means_arr))
        if means_arr.size > 1:
            sem = float(np.std(means_arr, ddof=1) / np.sqrt(means_arr.size))
        else:
            sem = float("nan")

        return (
            inv_eta_mean,
            sem,
            len(means_arr),
            block_ns,
        )  # last is effective block size

    @staticmethod
    def _temp_stats(
        t_ns: np.ndarray,
        T: np.ndarray,
        *,
        tmin_ns: float,
        tmax_ns: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return (mean T, dT/dt in K/ns) over the analysis window."""
        mask = t_ns >= tmin_ns
        if tmax_ns is not None:
            mask &= t_ns <= tmax_ns

        tt = t_ns[mask]
        TT = T[mask]
        if tt.size < 2:
            return None, None

        meanT = float(np.mean(TT))
        A = np.vstack([tt, np.ones_like(tt)]).T
        # Fit T = a + b t; b has units K/ns
        b, a = np.linalg.lstsq(A, TT, rcond=None)[0]
        drift = float(b)
        return meanT, drift

    @staticmethod
    def _weighted_linreg(
        x: np.ndarray,
        y: np.ndarray,
        ysem: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """
        Weighted linear regression y = a + b x with uncertainties on y.

        Returns (a, b, a_err, b_err, R^2).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ysem = np.asarray(ysem, dtype=float)

        # If any SEM is NaN or zero, fall back to unweighted least squares
        if not np.all(np.isfinite(ysem)) or np.any(ysem <= 0):
            # Create matrix of points to fit line to
            A = np.vstack([np.ones_like(x), x]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

            # Extract coefficients and fit line
            a, b = coeffs
            yfit = a + b * x

            # Calculate associated R^2 value
            ss_res = float(np.sum((y - yfit) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            n = len(x)
            s2 = ss_res / max(n - 2, 1)
            cov = s2 * np.linalg.inv(A.T @ A)
            a_err = float(np.sqrt(cov[0, 0]))
            b_err = float(np.sqrt(cov[1, 1]))
            return float(a), float(b), a_err, b_err, r2

        # Calculate the weight based off of the ysem value
        w = 1.0 / (ysem**2)
        W = np.sum(w)
        xbar = np.sum(w * x) / W
        ybar = np.sum(w * y) / W
        Sxx = np.sum(w * (x - xbar) ** 2)
        Sxy = np.sum(w * (x - xbar) * (y - ybar))
        b = Sxy / Sxx
        a = ybar - b * xbar

        yfit = a + b * x
        ss_res = float(np.sum(w * (y - yfit) ** 2))
        ss_tot = float(np.sum(w * (y - ybar) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        dof = max(len(x) - 2, 1)
        s2 = ss_res / dof
        a_err = float(np.sqrt(s2 * (1.0 / W + xbar**2 / Sxx)))
        b_err = float(np.sqrt(s2 / Sxx))

        return float(a), float(b), a_err, b_err, float(r2)

    @staticmethod
    def _infer_pp_amplitude_from_path(path: Path) -> float:
        """
        Infer the PP acceleration amplitude L from a file path.

        This assumes a directory / filename convention like:

            .../pp_300K/A_0p0125/pp_A0p0125_30ns_visc.xvg
            .../pp_300K/L_0p0100/pp_L0p0100_30ns_visc.xvg

        i.e. somewhere in the parents or the stem we have a token like
        'A_0p0125' or 'L_0p0125', where 'p' stands for '.'.

        If you use a different convention, either adapt this function or
        pass a custom amplitude_parser into analyze_pp_temperature_dir().
        """
        path = Path(path)

        # Try parent directories: look for A_0p0125 / L_0p0125
        for parent in [path.parent, *path.parents]:
            m = re.search(r"(?:A|L)_([0-9]+p[0-9]+)", parent.name)
            if m:
                token = m.group(1)
                return float(token.replace("p", "."))

        # Try the filename stem: e.g. pp_A0p0125_30ns_visc.xvg
        m = re.search(r"(?:A|L)_?([0-9]+p[0-9]+)", path.stem)
        if m:
            token = m.group(1)
            return float(token.replace("p", "."))

        # Last-resort: look for a bare float-ish token like '0.0125'
        m = re.search(r"([0-9]*\.[0-9]+)", path.stem)
        if m:
            return float(m.group(1))

        raise ValueError(
            f"Could not infer PP amplitude L from path '{path}'. "
            "Adjust _infer_pp_amplitude_from_path() or pass a custom parser."
        )


def main():
    """
    Example use of the analyzer
    """
    analysis = GromacsViscosityAnalysis()

    L = [0.010, 0.0125, 0.015, 0.0175]
    inveta_xvg = [
        "pp_L001_30ns_visc.xvg",
        "pp_L00125_30ns_visc.xvg",
        "pp_L0015_30ns_visc.xvg",
        "pp_L00175_30ns_visc.xvg",
    ]
    temp_xvg = [
        "pp_L001_30ns_temp.xvg",
        "pp_L00125_30ns_temp.xvg",
        "pp_L0015_30ns_temp.xvg",
        "pp_L00175_30ns_temp.xvg",
    ]

    result = analysis.analyze_pp_from_xvg(
        L_values=L,
        inveta_xvg=inveta_xvg,
        temp_xvg=temp_xvg,
        tmin_ns=2.0,
        block_ns=3.0,
    )

    print(result.eta0_cP, result.eta0_cP_err, result.qc_warnings)


if __name__ == "__main__":
    main()
