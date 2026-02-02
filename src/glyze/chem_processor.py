from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Set
from glyze.glyceride import Glyceride, FattyAcid, SymmetricGlyceride
from glyze.glyceride_mix import GlycerideMix
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ordered_set import OrderedSet

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
    overall_order: float | None = None  #

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


class ChemReactSim:
    """
    Core library for simulating the chemical processing of lipids
    """

    @staticmethod
    def species_space(list_of_fa: List[FattyAcid]) -> Set[Glyceride]:
        """
        Build the list of unique glyceride species from a list of fatty acids.
        The list includes monoglycerides, diglycerides, and triglycerides. Furthermore,
        the species treat positions 1 and 3 of the fatty acids as equivalent (i.e., 1,2,3 and 3,2,1 are the same).

        Parameters:
            list_of_fa (List[FattyAcid]): List of fatty acid objects.

        Returns:
            Set[Glyceride]: List of unique glyceride species represented as strings.
        """
        unique_species = OrderedSet()

        # Generate monoglycerides
        for fa in list_of_fa:
            mg = Glyceride((fa, None, None))
            unique_species.add(mg)
            mg = Glyceride((None, fa, None))
            unique_species.add(mg)

        # Generate diglycerides
        for i in range(len(list_of_fa)):
            for j in range(i, len(list_of_fa)):
                fa1 = list_of_fa[i]
                fa2 = list_of_fa[j]
                dg1 = Glyceride((fa1, fa2, None))
                dg2 = Glyceride((fa2, fa1, None))
                unique_species.add(dg1)
                unique_species.add(dg2)

        # Generate triglycerides
        for i in range(len(list_of_fa)):
            for j in range(i, len(list_of_fa)):
                for k in range(j, len(list_of_fa)):
                    fa1 = list_of_fa[i]
                    fa2 = list_of_fa[j]
                    fa3 = list_of_fa[k]
                    tg1 = Glyceride((fa1, fa2, fa3))
                    tg2 = Glyceride((fa1, fa3, fa2))
                    tg3 = Glyceride((fa3, fa1, fa2))
                    unique_species.update([tg1, tg2, tg3])

        return unique_species

    @staticmethod
    def random_esterification(list_of_fa: List[FattyAcid]):
        """
        Perform esterification of glycerol with given fatty acids.

        Args:
            glycerol (Glycerol): The glycerol molecule.
            fatty_acids (list of FattyAcid): List of fatty acids to esterify with.

        Returns:
            Glyceride_Composition: The resulting glyceride composition.
        """
        # Implementation of esterification logic
        pass

    @staticmethod
    def p_kinetic_esterification(
        list_of_fa: List[FattyAcid],
        initial_conc: List[int],
        k_calc: str = "permutation",
        chem_flag=False,
    ) -> PKineticSim:
        """
        Will simulate the batch reaction for the given glyceride species.

        Parameters:
            list_of_fa (List[FattyAcid]): List of fatty acid objects.
            initial_conc (List[int]): List of initial concentrations for each fatty acid.
            k_calc (str): Method to calculate rate constants. Options are "permutation" or
                            "random". Default is "permutation".
            chem_flag (bool): If True, divide by avogadro's number while calculating stochastic rate constants.

        Returns:
            Simulation: Cayenne Simulation object representing the batch reaction.
        """

        react_stoic = []
        prod_stoic = []
        rxn_names: List[str] = []
        ks = []
        # First add glyceride and fatty acid species to the reactants

        if len(initial_conc) - 1 != len(list_of_fa):
            raise ValueError("initial_conc must have the same length as list_of_fa")

        # Build all unique species once, and REUSE them when building reactions
        unique_mags = OrderedSet()
        mag_lookup: Dict[Tuple[str, str], SymmetricGlyceride] = (
            {}
        )  # (pos_tag, fa.name) -> MAG

        for fa in list_of_fa:
            # Count for fa attachment in middle and end positions
            # Formated as [glyceride, fatty acid]
            # (We will now encode these counts in full-length rows via _add_rxn.)
            # Generate monoglycerides and rxn_names
            mg_end = SymmetricGlyceride((fa, None, None))
            mg_mid = SymmetricGlyceride((None, fa, None))
            unique_mags.add(mg_end)
            unique_mags.add(mg_mid)
            mag_lookup[("end", fa.name)] = mg_end
            mag_lookup[("mid", fa.name)] = mg_mid

        unique_dags = OrderedSet()
        dag_lookup: Dict[Tuple[str, str, int], SymmetricGlyceride] = (
            {}
        )  # (mag.name, fa.name, index) -> DAG

        for mag in unique_mags:
            # Generate all the diglycerides form the monoglycerides
            for fa in list_of_fa:
                # Count for fa attachment in middle and end positions
                # Formated as [monoglyceride, fatty acid]
                # Generate diglycerides and rxn_names
                # If we have mag as (fa1, None, None), we can form (fa1, fa2, None)
                # or (fa1, None, fa2)
                if mag.sn[0] is not None:
                    dg1 = mag.add_fatty_acid(index=1, fatty_acid=fa)
                    dg2 = mag.add_fatty_acid(index=2, fatty_acid=fa)
                    unique_dags.add(dg1)
                    unique_dags.add(dg2)
                    dag_lookup[(mag.name, fa.name, 1)] = dg1
                    dag_lookup[(mag.name, fa.name, 2)] = dg2
                # Else if we have mag as (None, fa1, None), we can only form (fa2, fa1, None)
                else:
                    dg1 = mag.add_fatty_acid(index=0, fatty_acid=fa)
                    unique_dags.add(dg1)
                    dag_lookup[(mag.name, fa.name, 0)] = dg1

        unique_tgs = OrderedSet()
        tg_lookup: Dict[Tuple[str, str], SymmetricGlyceride] = (
            {}
        )  # (dag.name, fa.name) -> TG

        for dag in unique_dags:
            for fa in list_of_fa:
                # Count for fa attachment in middle and end positions
                # Formated as [glyceride, fatty acid]
                # Generate triglycerides and rxn_names
                # Fill the fatty acid in the empty position knowing that the first
                # position must be filled already
                if dag.sn[1] is None:
                    tg1 = dag.add_fatty_acid(index=1, fatty_acid=fa)
                elif dag.sn[2] is None:
                    tg1 = dag.add_fatty_acid(index=2, fatty_acid=fa)
                else:
                    raise ValueError("Unexpected diglyceride structure")
                unique_tgs.add(tg1)
                tg_lookup[(dag.name, fa.name)] = tg1

        unique_species = list(OrderedSet.union(unique_mags, unique_dags, unique_tgs))
        base_gly = "Glycerol"
        fa_names = [fa.name for fa in list_of_fa]
        gly_names = [specie.name for specie in unique_species]
        species_names = [base_gly, *fa_names, *gly_names]
        species_idx = {nm: i for i, nm in enumerate(species_names)}
        ns = len(species_names)

        # Generate monoglycerides and rxn_names (reuse prebuilt MAGs)
        for fa in list_of_fa:
            mg_end = mag_lookup[("end", fa.name)].name
            mg_mid = mag_lookup[("mid", fa.name)].name
            # Double count for position 1 and 3 equivalence
            ChemReactSim._add_rxn(
                react_stoic,
                prod_stoic,
                rxn_names,
                ks,
                species_idx,
                reactants=[base_gly, fa.name],
                products=[mg_end],
                k=2.0,
                name=f"Glyceride + {fa.name} => {mg_end}",
            )
            ChemReactSim._add_rxn(
                react_stoic,
                prod_stoic,
                rxn_names,
                ks,
                species_idx,
                reactants=[base_gly, fa.name],
                products=[mg_mid],
                k=1.0,
                name=f"Glyceride + {fa.name} => {mg_mid}",
            )

        # Build DAG reactions (reuse prebuilt DAGs)
        for mag in unique_mags:
            for fa in list_of_fa:
                if mag.sn[0] is not None:
                    dg1 = dag_lookup[(mag.name, fa.name, 1)].name
                    dg2 = dag_lookup[(mag.name, fa.name, 2)].name
                    ChemReactSim._add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg1],
                        k=1.0,
                        name=f"{mag.name} + {fa.name} => {dg1}",
                    )
                    ChemReactSim._add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg2],
                        k=1.0,
                        name=f"{mag.name} + {fa.name} => {dg2}",
                    )
                else:
                    dg1 = dag_lookup[(mag.name, fa.name, 0)].name
                    # two equivalent ends on MAG(mid)
                    ChemReactSim._add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg1],
                        k=2.0,
                        name=f"Glyceride + {fa.name} => {dg1}",
                    )

        # Build TG reactions (reuse prebuilt TGs)
        for dag in unique_dags:
            for fa in list_of_fa:
                # Fill the fatty acid in the empty position knowing that the first
                # position must be filled already
                tg1 = tg_lookup[(dag.name, fa.name)].name
                ChemReactSim._add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks,
                    species_idx,
                    reactants=[dag.name, fa.name],
                    products=[tg1],
                    k=1.0,
                    name=f"{dag.name} + {fa.name} => {tg1}",
                )

        # Initial state vector
        init_state = np.zeros(len(species_names), dtype=float)
        init_state[0] = initial_conc[0]  # Glycerol initial concentration
        for fa, c0 in zip(list_of_fa, initial_conc[1:]):
            init_state[species_idx[fa.name]] = float(c0)

        # Convert list of column vectors into full (ns, nr) matrices
        react_stoic = (
            np.hstack(react_stoic)
            if len(react_stoic)
            else np.zeros((ns, 0), dtype=float)
        )
        prod_stoic = (
            np.hstack(prod_stoic) if len(prod_stoic) else np.zeros((ns, 0), dtype=float)
        )

        ks = np.asarray(ks, dtype=float)

        # sanity checks
        ns = len(species_names)
        nr = len(rxn_names)
        assert react_stoic.shape == (
            ns,
            nr,
        ), f"react_stoic shape {react_stoic.shape} != (ns, nr)=({ns}, {nr})"
        assert prod_stoic.shape == (
            ns,
            nr,
        ), f"prod_stoic shape {prod_stoic.shape} != (ns, nr)=({ns}, {nr})"
        assert ks.shape == (nr,), f"k_det shape {ks.shape} != (nr,)={nr}"
        assert init_state.shape == (
            ns,
        ), f"init_state shape {init_state.shape} != (ns,)={ns}"

        print("Species index mapping:")
        for i, nm in enumerate(species_names):
            print(f"  [{i:2d}] {nm}")
        print()
        print("First few reactions and stoichiometry rows:")
        for i in range(min(5, len(rxn_names))):
            print(f"{i:3d}: {rxn_names[i]}")
            print("    Reactants:", np.where(react_stoic.T[i] != 0)[0])
            print("    Products: ", np.where(prod_stoic.T[i] != 0)[0])
        print()
        np.set_printoptions(linewidth=np.inf)
        print(f"Printing species names: {species_names}")
        print(
            f"Printing reaction stoichiometry:\nReactants:\n{np.array2string(react_stoic.T)}\nProducts:\n{np.array2string(prod_stoic.T)}"
        )
        print(f"Printing Initial state: {init_state}")
        print(f"Printing rate constants: {ks}")
        print(f"Printing shape of reactant stoichiometry: {react_stoic.shape}")

        return PKineticSim(
            species_names=species_names,
            react_stoic=react_stoic,
            prod_stoic=prod_stoic,
            init_state=init_state,
            k_det=ks,
            rxn_names=rxn_names,
            chem_flag=chem_flag,
        )

    @staticmethod
    def p_kinetic_interesterification(
        list_of_stuff: List[Glyceride],
        initial_conc: List[int],
        plucked: List[str],
        arranged: List[str],
        k_calc: str = "permutation",
        chem_flag=False,
    ) -> PKineticSim:
        """
        Will simulate the bath reaction for the given glyceride spieces.

        Parameters:
            list_of_stuff (List[Glyceride]): List of the MAGs, DAGs, TAGs present in the reaction
            initial_conc (List[int]): List of initial concentrtions for each MAGs, DAGs, or TAGs
            k_calc (str): Method to calculate rate constants. Options are "permutation" or
                                    "random". Default is "permutation".
            chem_flag (bool): If True, divide by avogadro's number while calculating stochastic rate constants.

        Returns:
            Simulation: runs the graph to see what TAGs will be left
        """
        #
        react_stoic = []
        prod_stoic = []
        rxn_names: List[str] = []
        ks = []

        # First break TAGs and form DAGs and FAs

        if len(initial_conc) != len(list_of_stuff):
            raise ValueError("initial_conc must have the same length as list_of_fa")

            # Build all unique species once, and REUSE them when building reactions
        unique_dags = OrderedSet()
        dag_lookup: Dict[Tuple[str, int], Tuple[str, SymmetricGlyceride]] = (
            {}
        )  # (gly.name, fa.name, index) -> DAG + fatty acid

        # list of fatty acids that can only react to either the ends or the middles
        mid: List[FattyAcid] = []
        end: List[FattyAcid] = []

        for i in range(len(list_of_stuff)):
            tag = list_of_stuff[i]
            fa0, fa1, fa2 = tag.sn
            if plucked[i] == "end":
                dag0, _ = tag.remove_fatty_acid(index=0)  # G_None_FA_FA
                dag2, _ = tag.remove_fatty_acid(index=2)  # G_FA_FA_None
                unique_dags.add(dag0)
                unique_dags.add(dag2)
                dag_lookup[(tag.name, 0)] = (fa0.name, dag0)
                dag_lookup[(tag.name, 2)] = (fa2.name, dag2)
                # if the arrangement goes to the end or the middle (sorting)
                if arranged[i] == "end":
                    end.append(fa0)
                    end.append(fa2)
                else:
                    mid.append(fa0)
                    mid.append(fa2)
            else:  # its mid
                dag1, _ = tag.remove_fatty_acid(index=1)  # G_FA_None_FA
                unique_dags.add(dag1)
                dag_lookup[(tag.name, 1)] = (fa1.name, dag1)
                if arranged[i] == "end":
                    end.append(fa1)
                else:
                    mid.append(fa1)

        unique_tgs = OrderedSet()
        tg_lookup: Dict[Tuple[str, str], SymmetricGlyceride] = (
            {}
        )  # (dag.name, fa.name) -> TG

        for dag in unique_dags:
            if dag.sn[1] is None:
                for fa in mid:
                    # if the middle is empty
                    tg1 = dag.add_fatty_acid(index=1, fatty_acid=fa)
                    unique_tgs.add(tg1)
                    tg_lookup[(dag.name, fa.name)] = tg1
            else:
                # or its not and so the ends must be empty
                for fa in end:
                    if dag.sn[0] is None:
                        tg1 = dag.add_fatty_acid(index=0, fatty_acid=fa)
                        unique_tgs.add(tg1)
                        tg_lookup[(dag.name, fa.name)] = tg1
                    elif dag.sn[2] is None:
                        tg1 = dag.add_fatty_acid(index=2, fatty_acid=fa)
                        unique_tgs.add(tg1)
                        tg_lookup[(dag.name, fa.name)] = tg1
                    else:
                        raise ValueError("Unexpected diglyceride structure")

        for i in range(len(list_of_stuff)):
            if plucked[i] == arranged[i]:
                dag, fa = tag.remove_fatty_acid(index=1)
                tg_lookup[(dag.name, fa.name)] = list_of_stuff[i]
            else:
                dag0, fa0 = tag.remove_fatty_acid(index=0)
                dag2, fa2 = tag.remove_fatty_acid(index=2)
                tg_lookup[(dag0.name, fa0.name)] = list_of_stuff[i]
                tg_lookup[(dag2.name, fa2.name)] = list_of_stuff[i]

        unique_species = list(OrderedSet.union(unique_dags, unique_tgs))
        midend = OrderedSet(mid + end)  # Combine the two lists together
        init_tags = [init_tags.name for init_tags in list_of_stuff]
        fa_names = [fa.name for fa in midend]
        gly_names = [specie.name for specie in unique_species]
        species_names = [*fa_names, *OrderedSet(gly_names + init_tags)]
        species_idx = {nm: i for i, nm in enumerate(species_names)}

        ns = len(species_names)

        # build reaction names
        for i, tag in enumerate(list_of_stuff):
            plucked_value = plucked[i]
            if plucked_value == "end":
                fa0, dag0 = dag_lookup[(tag.name, 0)]
                fa2, dag2 = dag_lookup[(tag.name, 2)]
                dag0 = dag0.name
                dag2 = dag2.name
                ChemReactSim._add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag0, fa0],
                    k=1.0,
                    name=f"{tag.name} => {dag0} + {fa0}",
                )
                ChemReactSim._add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag2, fa2],
                    k=1.0,
                    name=f"{tag.name} => {dag2} + {fa2}",
                )
            else:
                fa1, dag1 = dag_lookup[(tag.name, 1)]
                dag1 = dag1.name
                ChemReactSim._add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag1, fa1],
                    k=1.0,
                    name=f"{tag.name} => {dag1} + {fa1}",
                )

        # Build TG reactions (reuse prebuilt TGs)
        for dag in unique_dags:
            if dag.sn[1] is not None:
                for fa in end:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    ChemReactSim._add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks,
                        species_idx,
                        reactants=[dag.name, fa.name],
                        products=[tg1],
                        k=2.0,
                        name=f"{dag.name} + {fa.name} => {tg1}",
                    )
            else:
                for fa in mid:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    ChemReactSim._add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks,
                        species_idx,
                        reactants=[dag.name, fa.name],
                        products=[tg1],
                        k=1.0,
                        name=f"{dag.name} + {fa.name} => {tg1}",
                    )

        # Initial state vector
        init_state = np.zeros(len(species_names), dtype=float)
        for gly, c0 in zip(list_of_stuff, initial_conc[0:]):
            init_state[species_idx[gly.name]] = float(c0)

        # Convert list of column vectors into full (ns, nr) matrices
        react_stoic = (
            np.hstack(react_stoic)
            if len(react_stoic)
            else np.zeros((ns, 0), dtype=float)
        )
        prod_stoic = (
            np.hstack(prod_stoic) if len(prod_stoic) else np.zeros((ns, 0), dtype=float)
        )

        ks = np.asarray(ks, dtype=float)
        print("Species index mapping:")
        for i, nm in enumerate(species_names):
            print(f"  [{i:2d}] {nm}")
        print()

        return PKineticSim(
            species_names=species_names,
            react_stoic=react_stoic,
            prod_stoic=prod_stoic,
            init_state=init_state,
            k_det=ks,
            rxn_names=rxn_names,
            chem_flag=chem_flag,
        )

    @staticmethod
    def deoderization(mix: GlycerideMix, T: float, P: float) -> GlycerideMix:
        """
        Perform deoderization on a glyceride mix at a given temperature and pressure.

        Parameters:
        -----------
            mix (GlycerideMix): The glyceride mix to be deoderized.
            T (float): Temperature in Kelvin.
            P (float): Pressure in atm.

        Returns:
        -----------
            GlycerideMix: The deoderized glyceride mix.
        """

    @staticmethod
    def random_intersterification(g_composition: GlycerideMix):
        """
        Perform intersterification among a list of glycerides.

        Args:
            glycerides (list of Glyceride): List of glycerides to undergo intersterification.

        Returns:
            list of Glyceride: The resulting glycerides after intersterification.
        """
        # Random Interesterification model
        g_grid = np.zeros((21, 21, 21))

        # Populate the grid with glyceride chain lengths
        for glyceride, qty in g_composition.components.items():
            chain_lengths = glyceride.chain_lengths()
            # Unpack chain lengths and increment the grid
            g_grid[*chain_lengths] = qty

        # Find the unique fatty acid chain lengths present
        unique_indices_list = np.unique(np.concatenate(np.nonzero(g_grid))).tolist()

        # Generate output by calculating number of fatty acid with each length present
        output = []
        for i in range(0, len(unique_indices_list)):
            # Sum over the grid to count occurrences of each fatty acid length
            output.append(
                1
                / 3
                * sum(
                    sum(g_grid[unique_indices_list[i], :, :]),
                    sum(g_grid[:, unique_indices_list[i], :]),
                    sum(g_grid[:, :, unique_indices_list[i]]),
                )
            )
        return output

    @staticmethod
    def _add_rxn(
        react_stoic,
        prod_stoic,
        rxn_names,
        ks,
        species_idx,
        reactants,
        products,
        k,
        name,
    ):
        """Build one reaction column using full-length vectors"""
        ns = len(species_idx)
        # Each reaction is a column vector (species are rows)
        r_col = np.zeros((ns, 1), dtype=float)
        p_col = np.zeros((ns, 1), dtype=float)

        for nm in reactants:
            r_col[species_idx[nm], 0] += 1.0
        for nm in products:
            p_col[species_idx[nm], 0] += 1.0

        # Append as new columns
        if len(react_stoic) == 0:
            react_stoic.append(r_col)
            prod_stoic.append(p_col)
        else:
            react_stoic.append(r_col)
            prod_stoic.append(p_col)

        ks.append(float(k))
        rxn_names.append(name)
