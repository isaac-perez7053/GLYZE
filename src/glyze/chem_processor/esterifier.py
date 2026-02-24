from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from glyze.glyceride import FattyAcid, SymmetricGlyceride, FattyAcid
from ordered_set import OrderedSet
from glyze.utils import add_rxn
from .p_kinetic_sim import PKineticSim


class Esterifier:

    @staticmethod
    def p_kinetic_esterification_rxn_list(
        list_of_fa: List[FattyAcid],
    ) -> List[str]:
        """
        Return a list containing strings that each represent the
        chemical reactions occurring in the esterification reaction
        shown above.

        Parameters:
            list_of_fa (List[FattyAcid]): List of fatty acid objects.

        Returns:
            List[str]: List of chemical reaction equations
        """
        list_of_rxns = []

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

        # Generate monoglycerides and rxn_names (reuse prebuilt MAGs)
        for fa in list_of_fa:
            mg_end = mag_lookup[("end", fa.name)].name
            mg_mid = mag_lookup[("mid", fa.name)].name
            list_of_rxns.append(f"Glyceride + {fa.name} → {mg_end} + H2O")
            list_of_rxns.append(f"Glyceride + {fa.name} → {mg_mid} + H2O")

        # Build DAG reactions (reuse prebuilt DAGs)
        for mag in unique_mags:
            for fa in list_of_fa:
                if mag.sn[0] is not None:
                    dg1 = dag_lookup[(mag.name, fa.name, 1)].name
                    dg2 = dag_lookup[(mag.name, fa.name, 2)].name
                    list_of_rxns.append(f"{mag.name} + {fa.name} → {dg1} + H2O")
                    list_of_rxns.append(f"{mag.name} + {fa.name} → {dg2} + H2O")

                else:
                    dg1 = dag_lookup[(mag.name, fa.name, 0)].name
                    # two equivalent ends on MAG(mid)
                    list_of_rxns.append(f"Glyceride + {fa.name} → {dg1} + H2O")

        # Build TG reactions (reuse prebuilt TGs)
        for dag in unique_dags:
            for fa in list_of_fa:
                tg1 = tg_lookup[(dag.name, fa.name)].name
                list_of_rxns.append(f"{dag.name} + {fa.name} → {tg1} + H2O")

        # Convert list of column vectors into full (ns, nr) matrices
        return list_of_rxns

    @staticmethod
    def p_kinetic_esterification(
        list_of_fa: List[FattyAcid],
        initial_conc: List[int],
        ks: List[float],
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
            PKineticSim: PKineticSim object representing the batch reaction.
        """

        react_stoic = []
        prod_stoic = []
        rxn_names: List[str] = []
        ks_internal = []

        if ks is not None:
            if len(ks) != len(
                Esterifier.p_kinetic_esterification_rxn_list(list_of_fa=list_of_fa)
            ):
                raise ValueError("Make sure that you input the correct number of ks!")
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
        H2O = "H2O"
        fa_names = [fa.name for fa in list_of_fa]
        gly_names = [specie.name for specie in unique_species]
        species_names = [base_gly, H2O, *fa_names, *gly_names]
        species_idx = {nm: i for i, nm in enumerate(species_names)}
        ns = len(species_names)

        # Generate monoglycerides and rxn_names (reuse prebuilt MAGs)
        for fa in list_of_fa:
            mg_end = mag_lookup[("end", fa.name)].name
            mg_mid = mag_lookup[("mid", fa.name)].name
            # Double count for position 1 and 3 equivalence
            add_rxn(
                react_stoic,
                prod_stoic,
                rxn_names,
                ks_internal,
                species_idx,
                reactants=[base_gly, fa.name],
                products=[mg_end, H2O],
                k=(ks.pop(0) if ks is not None else 2.0),
                name=f"Glyceride + {fa.name} => {mg_end} + H2O",
            )
            add_rxn(
                react_stoic,
                prod_stoic,
                rxn_names,
                ks_internal,
                species_idx,
                reactants=[base_gly, fa.name],
                products=[mg_mid, H2O],
                k=(ks.pop(0) if ks is not None else 1.0),
                name=f"Glyceride + {fa.name} => {mg_mid} + H2O",
            )

        # Build DAG reactions (reuse prebuilt DAGs)
        for mag in unique_mags:
            for fa in list_of_fa:
                if mag.sn[0] is not None:
                    dg1 = dag_lookup[(mag.name, fa.name, 1)].name
                    dg2 = dag_lookup[(mag.name, fa.name, 2)].name
                    add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks_internal,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg1, H2O],
                        k=(ks.pop(0) if ks is not None else 1.0),
                        name=f"{mag.name} + {fa.name} => {dg1} + H2O",
                    )
                    add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks_internal,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg2, H2O],
                        k=(ks.pop(0) if ks is not None else 1.0),
                        name=f"{mag.name} + {fa.name} => {dg2} + H2O",
                    )
                else:
                    dg1 = dag_lookup[(mag.name, fa.name, 0)].name
                    # two equivalent ends on MAG(mid)
                    add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks_internal,
                        species_idx,
                        reactants=[mag.name, fa.name],
                        products=[dg1, H2O],
                        k=(ks.pop(0) if ks is not None else 2.0),
                        name=f"Glyceride + {fa.name} => {dg1} + H2O",
                    )

        # Build TG reactions (reuse prebuilt TGs)
        for dag in unique_dags:
            for fa in list_of_fa:
                # Fill the fatty acid in the empty position knowing that the first
                # position must be filled already
                tg1 = tg_lookup[(dag.name, fa.name)].name
                add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks_internal,
                    species_idx,
                    reactants=[dag.name, fa.name],
                    products=[tg1, H2O],
                    k=(ks.pop(0) if ks is not None else 1.0),
                    name=f"{dag.name} + {fa.name} => {tg1} + H2O",
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

        ks_internal = np.asarray(ks_internal, dtype=float)

        # # sanity checks
        # ns = len(species_names)
        # nr = len(rxn_names)
        # assert react_stoic.shape == (
        #     ns,
        #     nr,
        # ), f"react_stoic shape {react_stoic.shape} != (ns, nr)=({ns}, {nr})"
        # assert prod_stoic.shape == (
        #     ns,
        #     nr,
        # ), f"prod_stoic shape {prod_stoic.shape} != (ns, nr)=({ns}, {nr})"
        # assert ks.shape == (nr,), f"k_det shape {ks.shape} != (nr,)={nr}"
        # assert init_state.shape == (
        #     ns,
        # ), f"init_state shape {init_state.shape} != (ns,)={ns}"

        # print("Species index mapping:")
        # for i, nm in enumerate(species_names):
        #     print(f"  [{i:2d}] {nm}")
        # print()
        # print("First few reactions and stoichiometry rows:")
        # for i in range(min(5, len(rxn_names))):
        #     print(f"{i:3d}: {rxn_names[i]}")
        #     print("    Reactants:", np.where(react_stoic.T[i] != 0)[0])
        #     print("    Products: ", np.where(prod_stoic.T[i] != 0)[0])
        # print()
        # np.set_printoptions(linewidth=np.inf)
        # print(f"Printing species names: {species_names}")
        # print(
        #     f"Printing reaction stoichiometry:\nReactants:\n{np.array2string(react_stoic.T)}\nProducts:\n{np.array2string(prod_stoic.T)}"
        # )
        # print(f"Printing Initial state: {init_state}")
        # print(f"Printing rate constants: {ks}")
        # print(f"Printing shape of reactant stoichiometry: {react_stoic.shape}")

        return PKineticSim(
            species_names=species_names,
            react_stoic=react_stoic,
            prod_stoic=prod_stoic,
            init_state=init_state,
            k_det=ks_internal,
            rxn_names=rxn_names,
            chem_flag=chem_flag,
        )
