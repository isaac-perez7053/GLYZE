from __future__ import annotations

import numpy as np
from typing import Tuple
from glyze.glyceride import Glyceride, FattyAcid, FattyAcid
from glyze.glyceride_mix import GlycerideMix
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

# TODO: Implement custom k-list and list reactions
AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1e-24


@dataclass
class PKineticSim:
    """
    Simulator of fat formulation chemical kinetics via mass-action ODEs.

    Attributes
    ----------
    species_names : list
        List of species names.
    react_stoic : np.ndarray
        Reactant stoichiometry matrix R (ns, nr).
    prod_stoic : np.ndarray
        Product stoichiometry matrix P (ns, nr).
    init_state : np.ndarray
        Initial state vector (ns,).
    """

    species_names: list
    react_stoic: np.ndarray  # R (ns, nr)
    prod_stoic: np.ndarray  # P (ns, nr)
    init_state: np.ndarray  # (ns,)
    k_det: np.ndarray  # (nr,)
    rxn_names: list
    chem_flag: bool
    overall_order: float | None = None

    def S(self) -> np.ndarray:
        return self.prod_stoic - self.react_stoic

    def alpha(self) -> np.ndarray:
        """
        Kinetic orders alpha(ns, nr).
        If overall_order is None -> alpha = R (standard mass-action).
        Else for each reaction j, scale alpha[:,j] = R[:,j] * (overall_order / sum(R[:,j])),
        preserving relative contributions but forcing the desired total order.
        """
        R = self.react_stoic
        if self.overall_order is None:
            return R
        alpha = R.astype(float).copy()
        sums = np.sum(R, axis=0)  # (nr,)
        # avoid divide-by-zero for empty columns
        scale = np.where(sums > 0, self.overall_order / sums, 1.0)
        alpha = alpha * scale  # broadcast over rows
        return alpha

    def rates(self, x: np.ndarray) -> np.ndarray:
        """
        r_j(x) = k_j * sigma_i x_i^{alpha_ij}, robust at x_i=0.
        """
        alpha = self.alpha()  # (ns, nr)
        k = self.k_det  # (nr,)
        x_clip = np.clip(x, 0.0, None)
        xpow = np.power(x_clip[:, None], alpha)  # (ns, nr); 0**0 -> 1
        term = np.prod(xpow, axis=0)  # (nr,)
        return k * term

    def rhs(self, t, x):
        return self.S() @ self.rates(x)

    def solve(
        self,
        t_span: Tuple[float, float],
        t_eval: np.ndarray | None = None,
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ):
        """
        Integrate the ODE system.
        - Default method 'LSODA'
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 400)
        sol = solve_ivp(
            self.rhs,
            t_span,
            self.init_state,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            vectorized=False,
        )
        self.sol = sol
        return sol

    def plot(self, sol, show_species: list[str] | None = None, figsize=(10, 6)):
        """
        Plot specified species (or all if None)
        """
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        idxs = (
            range(len(self.species_names))
            if show_species is None
            else [self.species_names.index(nm) for nm in show_species]
        )

        plt.figure(figsize=figsize)
        for i in idxs:
            plt.plot(sol.t, sol.y[i, :], label=self.species_names[i])
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.legend(loc="best", ncols=2)
        plt.tight_layout()
        plt.show()

    def _top_species_indices(self, sol, n: int = 12):
        """Pick top-N species by max concentration over time (to declutter)."""
        maxvals = np.nanmax(sol.y, axis=1)
        order = np.argsort(maxvals)[::-1]
        return order[:n]

    def plot_interactive(
        self,
        sol,
        show_species: list[str] | None = None,
        top_n: int | None = 12,
        return_fig: bool = False,
    ):
        """
        Interactive Plotly plot.

        Parameters:
            TODO: Write the docstring
        """
        import plotly.graph_objects as go

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        ns = len(self.species_names)
        if show_species is None:
            idxs = range(ns) if top_n is None else self._top_species_indices(sol, top_n)
        else:
            idxs = [self.species_names.index(nm) for nm in show_species]

        fig = go.Figure()
        for i in idxs:
            fig.add_trace(
                go.Scatter(
                    x=sol.t,
                    y=sol.y[i, :],
                    mode="lines",
                    name=self.species_names[i],
                    hovertemplate=f"{self.species_names[i]}<br>t=%{{x:.3g}}, c=%{{y:.5g}}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Concentration",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=60, r=20, t=30, b=70),
        )
        # range slider + y-scale buttons
        fig.update_xaxes(rangeslider=dict(visible=True))
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    showactive=True,
                    x=0.5,
                    y=-0.2,
                    xanchor="center",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Linear",
                            method="relayout",
                            args=[{"yaxis.type": "linear"}],
                        ),
                        dict(
                            label="Log", method="relayout", args=[{"yaxis.type": "log"}]
                        ),
                    ],
                )
            ]
        )

        if return_fig:
            return fig
        import plotly.io as pio

        pio.renderers.default = "browser"
        fig.show()

    def to_csv(self, output_path: str):
        """
        Create a csv dumping the results of the simulation.

        Paramters:
        ----------
            output_path (str): The path to the written csv file

        Returns:
        --------
            None
        """
        # Create the headers of the csv file
        headers = [
            "Species",
            "Initial Concentration",
            "Final Concentration",
            "Initial Mass (g)",
            "Final Mass (g)",
            "Mass Fraction",
            "Carbon Number",
        ]

        # Basic preconditions
        if not hasattr(self, "sol"):
            raise ValueError(
                "No solution available. Run `solve` before exporting to CSV."
            )
        if not hasattr(self, "species_names"):
            raise ValueError("`species_names` not set on simulator.")
        if not hasattr(self, "init_state"):
            raise ValueError("`init_state` not set on simulator.")

        ns = len(self.species_names)
        if len(self.init_state) != ns:
            raise ValueError("Length of `init_state` does not match number of species")
        if self.sol.y.shape[0] != ns:
            raise ValueError("Solution shape does not match number of species")

        initial_moles = list(self.init_state)
        final_moles = [float(x) for x in self.sol.y[:, -1]]

        # compute masses and carbon numbers
        rows = []
        total_final_mass = 0.0
        for i, name in enumerate(self.species_names):
            molar_mass = None
            carbon_number = ""

            # handle special species
            lname = str(name).lower()
            if lname in ("glycerol",):
                molar_mass = 92.094
                carbon_number = 3
                display_name = "Glycerol"
            elif lname in ("h2o", "water"):
                molar_mass = 18.01528
                carbon_number = 0
                display_name = "H2O"
            else:
                # decide whether glyceride or fatty acid by name pattern
                try:
                    if str(name).count("_") == 3:
                        obj = Glyceride.from_name(name)
                    else:
                        obj = FattyAcid.from_name(name)
                except Exception:
                    obj = None

                if obj is not None:
                    molar_mass = obj.molar_mass
                    carbon_number = obj.num_carbons
                    display_name = obj.name
                else:
                    display_name = name

            imoles = float(initial_moles[i])
            fmoles = float(final_moles[i])
            imass = imoles * molar_mass if molar_mass is not None else ""
            fmass = fmoles * molar_mass if molar_mass is not None else ""

            if isinstance(fmass, (int, float)):
                total_final_mass += fmass

            rows.append(
                {
                    "Species": display_name,
                    "Initial Concentration": imoles,
                    "Final Concentration": fmoles,
                    "Initial Mass (g)": imass,
                    "Final Mass (g)": fmass,
                    "Mass Fraction": 0.0,  # placeholder, fill later
                    "Carbon Number": carbon_number,
                }
            )

        # compute mass fractions
        for r in rows:
            fmass = r["Final Mass (g)"]
            if isinstance(fmass, (int, float)) and total_final_mass > 0:
                r["Mass Fraction"] = fmass / total_final_mass
            else:
                r["Mass Fraction"] = 0.0

        # write CSV
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(
                    [
                        r["Species"],
                        r["Initial Concentration"],
                        r["Final Concentration"],
                        r["Initial Mass (g)"],
                        r["Final Mass (g)"],
                        r["Mass Fraction"],
                        r["Carbon Number"],
                    ]
                )

    @property
    def glyceride_mix(self):
        """
        Glyceride mixture output
        """
        if hasattr(self, "sol"):
            if hasattr(self, "species_names"):
                list_of_gly = [
                    (
                        Glyceride.from_name(x)
                        if x.count("_") == 3
                        else FattyAcid.from_name(x)
                    )
                    for x in self.species_names[1:]
                    if x not in ("H2O", "Glycerol")
                ]
                # Grab the final concentration of every species and create a list of them that
                # correspond to the list of glycerides

                # Cut off glycerol and H2O to match list_of_gly
                list_of_conc = [
                    x[-1]
                    for i, x in enumerate(self.sol.y[1:], 1)
                    if self.species_names[i] not in ("H2O", "Glycerol")
                ]

                # Append together glycerol and water
                list_of_components = ["Glycerol", "H2O"]
                list_of_all_conc = [self.sol.y[0][-1], self.sol.y[1][-1]]
                list_of_components += list_of_gly
                list_of_all_conc += list_of_conc

                return GlycerideMix(
                    mix=[x for x in zip(list_of_components, list_of_all_conc)]
                )
            else:
                raise ValueError("Please define the name of the species")
        else:
            raise ValueError("Please run the solver before grabbing the results")
