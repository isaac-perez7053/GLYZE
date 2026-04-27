"""
References
----------
    Teles dos Santos, M., Gerbaud, V., Le Roux, G.A.C. (2013). Modeling and
    simulation of melting curves and chemical interesterification of binary blends
    of vegetable oils. Chemical Engineering Science, 87, 14-22.
"""

from __future__ import annotations

import sys
import warnings
import pandas as pd
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
    Equation,
    Model,
    Sum,
    Sense,
)

from gamspy.math import log

from glyze.glyceride import Glyceride
from glyze.glyceride_mix import GlycerideMix, MixtureComponent
import numpy


# TODO: Add unsaturated contribution
R = 8.314
# Canonical list of all phase labels used throughout the model
PHASES = ["liquid", "alpha", "beta_prime", "beta"]


class DSC:
    """ """

    _cache = {}

    @staticmethod
    def g(T: float, mix: GlycerideMix, polymorph: str) -> float:
        """
        Calculate the intensive Gibbs energy g^j [J/mol] for phase j of the mixture
        evaluated at temperature T.

        For the liquid phase (Eq. 6 in DosSantos et al.)
        For a solid polymorph j (Eq. 8)

        Parameters
        ----------
        T : float
            Temperature [K].
        mix : GlycerideMix
            Mixture whose TAG components carry melting data.
        polymorph : str
            One of "liquid", "alpha", "beta_prime", "beta".

        Returns
        -------
        float
            Intensive Gibbs energy of the phase [J/mol].
        """
        if polymorph not in PHASES:
            raise ValueError(
                f"Ensure that the polymorph is either liquid, beta, beta_prime, or alpha! "
                f"Got '{polymorph}'."
            )

        # Mole fractions: quantities stored in GlycerideMix are already in moles
        # TODO: Ensure that the mixture is indeed in moles
        total_moles = sum(mix.mix.values())

        # Calculate mole fractions
        x = {comp: qty / total_moles for comp, qty in mix.mix.items()}

        if polymorph == "liquid":
            # Eq. 6: ideal liquid phase
            return (
                R
                * T
                * sum(xi * (numpy.log(xi) if xi > 0 else 0.0) for xi in x.values())
            )

        # Solid phases: Eq. 8
        g_solid = 0.0
        for i, (comp, xi) in enumerate(x.items()):
            if xi <= 0.0:
                continue
            glyceride = comp.component if isinstance(comp, MixtureComponent) else comp
            if not isinstance(glyceride, Glyceride):
                # Non-TAG components (water, glycerol) are fully liquid; skip in solid.
                continue

            dH = glyceride.melting_enthalpy(polymorph)
            Tm = glyceride.melting_temp(polymorph)

            # Convert dH from kJ/mol -> J/mol if needed.  The params in glyceride.py
            # yield values already in kJ/mol consistent with the Seilert PII-DL model,
            # so we multiply by 1000 here.
            dH_J = dH * 1_000.0

            mu_solid = dH_J * (T / Tm - 1.0)  # chemical potential [J/mol]
            g_solid += xi * (
                mu_solid + R * T * numpy.log(DSC.gamma(mix, i, T, polymorph) * xi)
            )

        return g_solid

    @staticmethod
    def gamma(mix: GlycerideMix, i: int, T: float, polymorph: str):
        """
        Calculate the activity coefficient of the ith TAG with polymorph (alpha, beta_prime, beta)

        Parameters
        ----------

        Returns
        -------
        """
        # Assume ideal conditions for the alpha phase
        if polymorph == "alpha":
            return 1.0
        else:
            gE, A = DSC._excess_gibbs_free_energy(mix, T, polymorph)
            n = len(mix.mix.keys())
            return numpy.e ** (
                (-gE + sum(A[i][j] for j in range(n) if j != i)) / (R * T)
            )

    @staticmethod
    def chemical_potential(T: float, glyceride: Glyceride, polymorph: str) -> float:
        """
        Calculate the chemical potential mu_i^{0,j} [J/mol] of a pure TAG i in
        state j at temperature T (Eq. 7 in DosSantos et al.).

        For the liquid reference state the chemical potential is zero by definition.

        Parameters
        ----------
        T : float
            Temperature [K].
        glyceride : Glyceride
            Pure TAG whose melting properties are needed.
        polymorph : str
            One of "liquid", "alpha", "beta_prime", "beta".

        Returns
        -------
        float
            Chemical potential [J/mol].
        """
        if polymorph == "liquid":
            return 0.0
        elif polymorph in ["alpha", "beta_prime", "beta"]:
            # melting_enthalpy returns kJ/mol; convert to J/mol
            dH_J = glyceride.melting_enthalpy(polymorph) * 1_000.0
            Tm = glyceride.melting_temp(polymorph)
            return dH_J * (T / Tm - 1.0)
        else:
            raise ValueError(
                "Ensure that the polymorph is either liquid, beta, beta_prime, or alpha!"
            )

    @staticmethod
    def _minimize_gibbs_two_phase(
        T: float,
        glyceride_mix: GlycerideMix,
        solid_polymorph: str = None,
        warm_start: dict | None = None,
    ) -> dict:
        """
        Minimize the Gibbs Free Energy for a two-phase (liquid + one solid
        polymorph) sub-problem at temperature T.

        Parameters
        ----------
        T : float
            Temperature [K].
        glyceride_mix : GlycerideMix
            TAG mixture with molar quantities.
        solid_polymorph : str
            One of "alpha", "beta_prime", "beta".
        warm_start : dict or None
            Optional mapping ``{(tag_name, phase): moles}`` from the previous
            temperature step.

        Returns
        -------
        dict
            Keys: "n_ij", "SFC", "status", "objective".
        """
        total_moles = sum(glyceride_mix.mix.values())

        all_above_Tm = True
        for comp in glyceride_mix.mix.keys():
            glyceride = comp.component if isinstance(comp, MixtureComponent) else comp
            if not isinstance(glyceride, Glyceride):
                continue
            if len([x for x in comp.component.sn if x is not None]) != 3:
                continue
            if solid_polymorph is not None:
                Tm = glyceride.melting_temp(solid_polymorph)
                if T <= Tm:
                    all_above_Tm = False
                    break
            else:
                # If no solid polymorph specified, check if T is above all melting points
                for polymorph in ["alpha", "beta_prime", "beta"]:
                    Tm = glyceride.melting_temp(polymorph)
                    if T <= Tm:
                        all_above_Tm = False
                        break
                if not all_above_Tm:
                    break

        if all_above_Tm:
            equilibrium = {}
            for comp, qty in glyceride_mix.mix.items():
                equilibrium[(comp.name, "liquid")] = qty
                if solid_polymorph:
                    equilibrium[(comp.name, solid_polymorph)] = 0.0
                else:
                    for polymorph in ["alpha", "beta_prime", "beta"]:
                        equilibrium[(comp.name, polymorph)] = 0.0
            # Compute the all-liquid objective: sum_i n_i * RT * log(x_i)
            obj_val = 0.0
            for comp, qty in glyceride_mix.mix.items():
                xi = qty / total_moles
                if xi > 0:
                    obj_val += qty * R * T * numpy.log(xi)
            return {
                "n_ij": equilibrium,
                "SFC": 0.0,
                "status": "OptimalSolution",
                "objective": obj_val,
            }

        if solid_polymorph: 
            phases = ["liquid", solid_polymorph]
        else:
            phases = PHASES

        m = Container()

        tag_names = [comp.name for comp in glyceride_mix.mix.keys()]

        nc = Set(
            container=m,
            name="nc",
            records=tag_names,
            description="Number of Components (TAG species)",
        )
        np_set = Set(
            container=m,
            name="np",
            records=phases,
            description="Phases (liquid + one solid polymorph)",
        )

        # Total moles of each TAG i  (n_i)
        n_total_records = pd.DataFrame(
            [(comp.name, qty) for comp, qty in glyceride_mix.mix.items()],
            columns=["tag", "n_total"],
        )
        n_total = Parameter(
            container=m,
            name="n_total",
            domain=[nc],
            records=n_total_records,
            description="Total moles of TAG i across all phases",
        )

        # Pure-component chemical potentials mu_i^{0,j}(T) [J/mol]
        mu_records_full = DSC._build_mu_records(T, glyceride_mix)
        mu_records = mu_records_full[mu_records_full["phase"].isin(phases)]
        mu_pure = Parameter(
            container=m,
            name="mu_pure",
            domain=[nc, np_set],
            records=mu_records,
            description="Pure-component chemical potential of TAG i in phase j [J/mol]",
        )

        gamma_records_full = DSC._build_gamma_records(T, glyceride_mix)
        gamma_records = gamma_records_full[gamma_records_full["phase"].isin(phases)]
        gamma_param = Parameter(
            container=m,
            name="gamma_param",
            domain=[nc, np_set],
            records=gamma_records,
            description="Activity coefficient of TAG i in phase j",
        )

        # n_ij[i, j] – decision variables: moles of TAG i in phase j
        n_ij = Variable(
            container=m,
            name="n_ij",
            domain=[nc, np_set],
            type="Positive",
            description="Moles of TAG i in phase j",
        )

        n_ij.lo[nc, np_set] = 1e-12

        # Eq. 2: conservation of TAG i across all phases
        n_consv = Equation(
            container=m,
            name="n_consv",
            domain=[nc],
            description="Conservation of TAG moles across phases",
        )
        n_consv[nc] = Sum(np_set, n_ij[nc, np_set]) == n_total[nc]

        # Upper bound: n_ij[i,j] <= n_total[i]
        n_upper = Equation(
            container=m,
            name="n_upper",
            domain=[nc, np_set],
            description="Upper bound: moles of TAG i in phase j <= total moles of TAG i",
        )
        n_upper[nc, np_set] = n_ij[nc, np_set] <= n_total[nc]

        # Total moles per phase
        n_total_phase = Variable(
            container=m,
            name="n_total_phase",
            domain=[np_set],
            type="Positive",
            description="Total moles in phase j",
        )

        n_total_phase_def = Equation(
            container=m,
            name="n_total_phase_def",
            domain=[np_set],
            description="Total moles per phase",
        )
        n_total_phase_def[np_set] = n_total_phase[np_set] == Sum(nc, n_ij[nc, np_set])

        # Temperature scalar
        T_param = Parameter(
            container=m,
            name="T_param",
            records=T,
            description="Temperature [K]",
        )

        # Gas constant scalar
        R_param = Parameter(
            container=m,
            name="R_param",
            records=R,
            description="Gas constant [J/(mol K)]",
        )

        # Objective expression, equation 4
        obj_expr = Sum(
            (nc, np_set),
            n_ij[nc, np_set]
            * (
                mu_pure[nc, np_set]
                + R_param
                * T_param
                * log(
                    gamma_param[nc, np_set] * n_ij[nc, np_set] / n_total_phase[np_set]
                )
            ),
        )

        gibbs_free_energy = Model(
            m,
            name="GibbsFreeEnergy",
            equations=[n_consv, n_upper, n_total_phase_def],
            problem="NLP",
            sense=Sense.MIN,
            objective=obj_expr,
        )

        # Initialize initial mixture with something nonzero to ensure
        # log does not blow up
        init_mix = 1e-6

        for comp in glyceride_mix.mix.keys():
            for phase in phases:
                key = (comp.name, phase)
                if warm_start is not None and key in warm_start:
                    n_ij.l[comp.name, phase] = max(warm_start[key], init_mix)
                else:
                    if phase == "liquid":
                        n_ij.l[comp.name, phase] = glyceride_mix.mix[comp]
                    else:
                        n_ij.l[comp.name, phase] = init_mix

        for phase in phases:
            phase_total = sum(
                (
                    max(warm_start.get((comp.name, phase), init_mix), init_mix)
                    if warm_start is not None
                    else (glyceride_mix.mix[comp] if phase == "liquid" else init_mix)
                )
                for comp in glyceride_mix.mix.keys()
            )
            n_total_phase.l[phase] = phase_total

        # Solve
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gibbs_free_energy.solve(output=sys.stdout)
        except:
            pass

        # Extract results
        results_n_ij = n_ij.records

        total_solid_moles = 0.0
        total_liquid_moles = 0.0
        equilibrium = {}
        if results_n_ij is not None:
            for _, row in results_n_ij.iterrows():
                tag = row["nc"]
                phase = row["np"]
                level = row["level"]
                equilibrium[(tag, phase)] = level
                if phase == "liquid":
                    total_liquid_moles += level
                else:
                    total_solid_moles += level

        SFC = total_solid_moles / max(total_moles, 1e-12)

        liquid_fraction = total_liquid_moles / max(total_moles, 1e-12)
        SFC_TOL = 1e-4
        if liquid_fraction < SFC_TOL:
            SFC = 1.0
        elif SFC < SFC_TOL:
            SFC = 0.0

        try:
            obj_val = gibbs_free_energy.objective_value
        except Exception:
            obj_val = float("inf")

        return {
            "n_ij": equilibrium,
            "SFC": SFC,
            "status": gibbs_free_energy.status if equilibrium else "FailedSolve",
            "objective": obj_val if obj_val is not None else float("inf"),
        }

    @staticmethod
    def minimize_gibbs(
        T: float,
        glyceride_mix: GlycerideMix,
        warm_start: dict | None = None,
        two_phases: bool = True,
    ) -> dict:
        """
        Minimize the total Gibbs Free Energy of the TAG mixture at temperature T
        using GAMSPy and return the equilibrium mole numbers and the SFC.

        Following DosSantos et al. (2013), three separate two-phase
        (liquid + one solid polymorph) sub-problems are solved, and the
        result with the lowest Gibbs energy is selected.

        Parameters
        ----------
        T : float
            Temperature [K].
        glyceride_mix : GlycerideMix
            TAG mixture with molar quantities.
        warm_start : dict or None
            Optional mapping ``{(tag_name, phase): moles}`` from the previous
            temperature step (i.e. the ``"n_ij"`` field returned by a prior
            call to this method).  When supplied, these values are used as the
            initial point for ``n_ij`` instead of the default
            all-liquid initialization.

        Returns
        -------
        dict
            Keys:
            - "n_ij"  : dict mapping (tag_name, phase) -> equilibrium moles
            - "SFC"   : Solid Fat Content as a mass fraction [0, 1]
            - "status": solver status string
        """
        solid_polymorphs = ["alpha", "beta_prime", "beta"]

        best_result = None
        best_obj = float("inf")

        if two_phases:
            for polymorph in solid_polymorphs:
                result = DSC._minimize_gibbs_two_phase(
                    T,
                    glyceride_mix,
                    polymorph,
                    warm_start=warm_start,
                )
                obj = result.get("objective", float("inf"))
                # Grab the result with the lowest Gibbs energy among the three polymorph sub-problems
                if obj is not None and obj < best_obj:
                    best_obj = obj
                    best_result = result

            if best_result is None:
                best_result = result

        else:
            # Solve the full four-phase problem with all polymorphs simultaneously
            best_result = DSC._minimize_gibbs_two_phase(
                T,
                glyceride_mix,
                solid_polymorph=None,
                warm_start=warm_start,
            )

        return {
            "n_ij": best_result["n_ij"],
            "SFC": best_result["SFC"],
            "status": best_result["status"],
        }

    @staticmethod
    def compute_sfc_curve(
        glyceride_mix: GlycerideMix,
        T_start_C: float = -30.0,
        T_end_C: float = 60.0,
        dT_C: float = 1.0,
        two_phases: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the full SFC vs. temperature melting curve for the given mixture
        (solid -> liquid direction).

        Parameters
        ----------
        glyceride_mix : GlycerideMix
            TAG mixture with molar quantities.
        T_start_C : float
            Starting temperature [C].
        T_end_C : float
            Final temperature [C].
        dT_C : float
            Temperature step size [C].

        Returns
        -------
        pd.DataFrame
            Columns: ["T_C", "T_K", "SFC", "solver_status"].
        """
        rows = []
        T_C = T_start_C
        prev_n_ij: dict | None = None  # warm-start from the previous step
        prev_SFC = None
        # Ensure the units of the mixture are in moles!
        glyceride_mix = glyceride_mix.change_units("Moles")
        while T_C <= T_end_C:
            T_K = T_C + 273.15
            # if prev_SFC is not None and prev_n_ij is not None:
            #     if prev_SFC < 0.05:
            #         result = {
            #             "n_ij": prev_n_ij,
            #             "SFC": 0.0,
            #             "status": "OptimalSolution",
            #         }
            # else:
            #     result = DSC.minimize_gibbs(T_K, glyceride_mix, warm_start=prev_n_ij)

            result = DSC.minimize_gibbs(T_K, glyceride_mix, warm_start=prev_n_ij, two_phases=two_phases)
            sfc = result["SFC"]
            rows.append(
                {
                    "T_C": T_C,
                    "T_K": T_K,
                    "SFC": sfc,
                    "solver_status": result["status"],
                }
            )

            # Carry the equilibrium moles forward as the next warm-start
            prev_n_ij = result["n_ij"]

            # Stop scanning once completely melted (SFC = 0)
            # if sfc <= 0.0:
            #     break

            T_C += dT_C

        return pd.DataFrame(rows)

    @staticmethod
    def compute_sfc_hysteresis(
        glyceride_mix: GlycerideMix,
        T_start_C: float = -30.0,
        T_end_C: float = 60.0,
        dT_C: float = 1.0,
        two_phases: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the full SFC hysteresis loop for the given mixture.

        Two sequential scans are performed, each warm-starting every step from
        the previous step's equilibrium solution (Section 3.2, DosSantos et al.):

        * **Heating scan** (solid -> liquid): temperature increases from
          T_start_C to T_end_C (or until SFC = 0).
        * **Cooling scan** (liquid -> solid): temperature decreases from
          T_end_C back to T_start_C (or until SFC = 1).

        Parameters
        ----------
        glyceride_mix : GlycerideMix
            TAG mixture with molar quantities.
        T_start_C : float
            Lowest temperature of the loop [C].
        T_end_C : float
            Highest temperature of the loop [C].
        dT_C : float
            Absolute temperature step size [C]; must be positive.

        Returns
        -------
        pd.DataFrame
            Columns: ["T_C", "T_K", "SFC", "solver_status", "scan_direction"].
            ``scan_direction`` is ``"heating"`` or ``"cooling"``.
        """
        if dT_C <= 0:
            raise ValueError("dT_C must be a positive step size.")

        rows = []

        # Heating scan
        prev_n_ij: dict | None = None
        prev_SFC = None
        T_C = T_start_C
        # Ensure the units of the mixture are in moles!
        glyceride_mix = glyceride_mix.change_units("Moles")
        while T_C <= T_end_C:
            T_K = T_C + 273.15
            if prev_SFC is not None and prev_n_ij is not None and prev_SFC < 0.05:
                result = {
                    "n_ij": prev_n_ij,
                    "SFC": 0.0,
                    "status": "OptimalSolution",
                    "scan_direction": "heating",
                }
            else:
                result = DSC.minimize_gibbs(T_K, glyceride_mix, warm_start=prev_n_ij, two_phases=two_phases)
            # result = DSC.minimize_gibbs(T_K, glyceride_mix, warm_start=prev_n_ij)
            sfc = result["SFC"]
            prev_n_ij = result["n_ij"]
            prev_SFC = result["SFC"]
            rows.append(
                {
                    "T_C": T_C,
                    "T_K": T_K,
                    "SFC": sfc,
                    "solver_status": result["status"],
                    "scan_direction": "heating",
                }
            )

            # if sfc <= 0.0:
            #     # Fully melted – record final temperature for the cooling seed
            #     T_end_actual = T_C
            #     break

            T_C += dT_C
        else:
            T_end_actual = T_end_C

        # Cooling Scan
        T_C = T_end_actual
        result = {
            "n_ij": prev_n_ij,
            "SFC": 0.0,
            "status": "OptimalSolution",
            "scan_direction": "cooling",
        }
        prev_n_ij = result["n_ij"]
        prev_SFC = result["SFC"]
        sfc = result["SFC"]
        rows.append(
            {
                "T_C": T_C,
                "T_K": T_K,
                "SFC": sfc,
                "solver_status": result["status"],
                "scan_direction": "cooling",
            }
        )
        T_C -= dT_C

        while T_C >= T_start_C:
            T_K = T_C + 273.15
            if prev_SFC is not None and prev_n_ij is not None and prev_SFC > 0.97:
                result = {
                    "n_ij": prev_n_ij,
                    "SFC": 1.0,
                    "status": "OptimalSolution",
                    "scan_direction": "cooling",
                }
            else:
                result = DSC.minimize_gibbs(T_K, glyceride_mix, warm_start=prev_n_ij)

            prev_n_ij = result["n_ij"]
            prev_SFC = result["SFC"]
            sfc = result["SFC"]
            rows.append(
                {
                    "T_C": T_C,
                    "T_K": T_K,
                    "SFC": sfc,
                    "solver_status": result["status"],
                    "scan_direction": "cooling",
                }
            )

            # if sfc >= 1.0:
            #     break

            T_C -= dT_C

        return pd.DataFrame(rows)

    @staticmethod
    def plot_results(results, hysteresis: bool = False, return_fig=False):
        """
        Plot SFC vs. temperature from a DataFrame or list of dicts.

        Parameters
        ----------
        results : pd.DataFrame or list of dicts
            Output from ``compute_sfc_curve`` or ``compute_sfc_hysteresis``.
        hysteresis : bool
            When True the DataFrame is expected to contain a
            ``"scan_direction"`` column and both heating / cooling branches
            are drawn on the same axes with distinct colours.
        """
        import plotly.express as px
        import plotly.graph_objects as go

        df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)

        if df.empty:
            print("DataFrame is empty. No data to plot.")
            return

        if hysteresis and "scan_direction" in df.columns:
            fig = go.Figure()
            colors = {"heating": "#d62728", "cooling": "#1f77b4"}
            for direction, group in df.groupby("scan_direction"):
                fig.add_trace(
                    go.Scatter(
                        x=group["T_C"],
                        y=group["SFC"],
                        mode="lines+markers",
                        name=direction.capitalize(),
                        line=dict(color=colors.get(direction, None)),
                    )
                )
            fig.update_layout(
                title="SFC Hysteresis Loop (Heating & Cooling)",
                xaxis_title="Temperature (°C)",
                yaxis_title="Solid Fat Content (SFC)",
                yaxis=dict(range=[0.0, 1.0]),
                template="plotly_white",
                legend_title="Scan direction",
            )
        else:
            fig = px.line(
                df,
                x="T_C",
                y="SFC",
                markers=True,
                title="Solid Fat Content vs Temperature",
                labels={"T_C": "Temperature (°C)", "SFC": "Solid Fat Content (SFC)"},
            )
            fig.update_layout(yaxis=dict(range=[0.0, 1.0]), template="plotly_white")

        if return_fig:
            return fig
        else:
            fig.show()

    @staticmethod
    def _build_mu_records(T: float, glyceride_mix: GlycerideMix) -> pd.DataFrame:
        """
        Build a DataFrame of chemical potential records suitable for a GAMSPy
        Parameter with domain [nc, np].

        Each row has the form (tag_name, phase_label, mu_value).

        Parameters
        ----------
        T : float
            Temperature [K].
        glyceride_mix : GlycerideMix
            Mixture whose TAG components carry melting data.

        Returns
        -------
        pd.DataFrame
            Columns: ["tag", "phase", "mu"].
        """
        rows = []
        for comp in glyceride_mix.mix.keys():
            glyceride = comp.component if isinstance(comp, MixtureComponent) else comp
            if not isinstance(glyceride, Glyceride):
                # Non-TAG species (water, glycerol) treated as purely liquid
                for phase in PHASES:
                    rows.append((comp.name, phase, 0.0))
                continue
            elif len([x for x in comp.component.sn if x is not None]) != 3:
                # Treat DAGS and MAGS as purely liquid
                for phase in PHASES:
                    rows.append((comp.name, phase, 0.0))
                continue

            for phase in PHASES:
                mu_val = DSC.chemical_potential(T, glyceride, phase)
                rows.append((comp.name, phase, mu_val))

        return pd.DataFrame(rows, columns=["tag", "phase", "mu"])

    def _build_delta_H_records(glyceride_mix: GlycerideMix) -> pd.DataFrame:
        """
        Build a DataFrame of melting-enthalpy records [kJ/mol] for each
        (TAG, polymorph) pair, suitable for a GAMSPy Parameter with domain [nc, np].

        Parameters
        ----------
        glyceride_mix : GlycerideMix
            Mixture whose TAG components carry melting data.

        Returns
        -------
        pd.DataFrame
            Columns: ["tag", "phase", "delta_H"].
        """
        rows = []
        for comp in glyceride_mix.mix.keys():
            glyceride = comp.component if isinstance(comp, MixtureComponent) else comp
            for phase in PHASES:
                if phase == "liquid" or not isinstance(glyceride, Glyceride):
                    dH = 0.0
                # Treat MAG and DAG species as liquid
                elif len([x for x in comp.component.sn if x is not None]) != 3:
                    dH = 0.0
                else:
                    dH = glyceride.melting_enthalpy(phase)  # kJ/mol
                rows.append((comp.name, phase, dH))

        return pd.DataFrame(rows, columns=["tag", "phase", "delta_H"])

    # ------------------------------------------------------------------------------
    # Margulles isomorphism correlation model to calculate the activity coefficient
    # ------------------------------------------------------------------------------

    @staticmethod
    def _build_gamma_records(T: float, glyceride_mix: GlycerideMix) -> pd.DataFrame:
        """
        Build a DataFrame of activity coefficient records for each (TAG, phase)
        pair, suitable for a GAMSPy Parameter with domain [nc, np].

        The alpha and liquid phases are ideal (gamma = 1.0). For beta_prime and
        beta, gamma is computed via the 2-suffix Margules model.

        Parameters
        ----------
        T : float
            Temperature [K].
        glyceride_mix : GlycerideMix
            Mixture whose TAG components carry melting data.

        Returns
        -------
        pd.DataFrame
            Columns: ["tag", "phase", "gamma"].
        """
        rows = []
        for i, comp in enumerate(glyceride_mix.mix.keys()):
            for phase in PHASES:
                if phase == "liquid" or not isinstance(comp.component, Glyceride):
                    gamma_val = 1.0
                # Treat MAG and DAG species as liquid
                elif len([x for x in comp.component.sn if x is not None]) != 3:
                    gamma_val = 1.0
                else:
                    gamma_val = DSC.gamma(glyceride_mix, i, T, phase)
                rows.append((comp.name, phase, gamma_val))

        return pd.DataFrame(rows, columns=["tag", "phase", "gamma"])

    @staticmethod
    def _excess_gibbs_free_energy(mix: GlycerideMix, T: float, polymorph: str):
        if polymorph not in ["beta_prime", "beta"]:
            raise ValueError("The polymorph must either be beta_prime or beta!")
        comp_list = list(mix.mix.keys())
        A = numpy.zeros(shape=(len(comp_list), len(comp_list)))
        if (mix, T, polymorph) not in DSC._cache:

            # Create molar fractions list
            molar_fractions = [i / mix.total_quantity() for i in mix.quantities]
            for i in range(len(comp_list)):
                for j in range(i + 1, len(comp_list)):

                    # Grab fatty acids
                    nCi = comp_list[i].component.sn
                    nCj = comp_list[j].component.sn

                    # Calcluate isomorphism factor
                    v_non = sum(abs(nCi[k].length - nCj[k].length) for k in range(3))
                    v_0 = sum(min(nCi[k].length, nCj[k].length) for k in range(3))
                    epsilon = 1 - (v_non / v_0)

                    # Fit isomorphism factor to linear models
                    if polymorph == "beta":
                        A[i][j] = (
                            0 if epsilon > 0.93 else R * T * (-19.5 * epsilon + 18.2)
                        )
                    else:
                        A[i][j] = (
                            0 if epsilon > 0.98 else R * T * (-35.8 * epsilon + 35.9)
                        )

            # Calculate excess gibbs free energy
            gE = 0
            for i in range(len(comp_list)):
                for j in range(i + 1, len(comp_list)):
                    A[j][i] = A[i][j]
                    gE += A[i][j] * molar_fractions[i] * molar_fractions[j]

            # Update class cache
            DSC._cache[(mix, T, polymorph)] = (gE, A)

        return DSC._cache[(mix, T, polymorph)]
