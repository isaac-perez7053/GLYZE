from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from glyze.glyceride import Glyceride, FattyAcid, SymmetricGlyceride, FattyAcid
from ordered_set import OrderedSet
from .p_kinetic_sim import PKineticSim
from glyze.utils import add_rxn

AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1e-24


class Interesterifier:

    @staticmethod
    def p_kinetic_intersterification_rxn_list(
        list_of_glycerides: List[Glyceride],
        plucked: List[str],
        arranged: List[str],
    ) -> List[str]:
        """
        Generate a list of strings, representing the chemical equations
        active in the simulation

        Parameters:
            list_of_glycerides (List[Glyceride]): List of the MAGs, DAGs, TAGs present in the reaction
            plucked (List[str]):
            arranged (List[str]):

        Returns:
            List[str]: A list of chemical reaction equations present in the simulation
        """
        # Build all unique species once, and REUSE them when building reactions
        unique_dags = OrderedSet()
        dag_lookup: Dict[Tuple[str, int], Tuple[str, SymmetricGlyceride]] = (
            {}
        )  # (gly.name, fa.name, index) -> DAG + fatty acid
        list_of_rxns = []

        # list of fatty acids that can only react to either the ends or the middles
        mid: List[FattyAcid] = []
        end: List[FattyAcid] = []

        for i in range(len(list_of_glycerides)):
            tag = list_of_glycerides[i]
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

        for i in range(len(list_of_glycerides)):
            if plucked[i] == arranged[i]:
                dag, fa = tag.remove_fatty_acid(index=1)
                tg_lookup[(dag.name, fa.name)] = list_of_glycerides[i]
            else:
                dag0, fa0 = tag.remove_fatty_acid(index=0)
                dag2, fa2 = tag.remove_fatty_acid(index=2)
                tg_lookup[(dag0.name, fa0.name)] = list_of_glycerides[i]
                tg_lookup[(dag2.name, fa2.name)] = list_of_glycerides[i]

        # build reaction names
        for i, tag in enumerate(list_of_glycerides):
            plucked_value = plucked[i]
            if plucked_value == "end":
                fa0, dag0 = dag_lookup[(tag.name, 0)]
                fa2, dag2 = dag_lookup[(tag.name, 2)]
                list_of_rxns.append(f"{tag.name} => {dag0.name} + {fa0}")
                list_of_rxns.append(f"{tag.name} => {dag2.name} + {fa2}")
            else:
                fa1, dag1 = dag_lookup[(tag.name, 1)]
                list_of_rxns.append(f"{tag.name} => {dag1.name} + {fa1}")

        for dag in unique_dags:
            if dag.sn[1] is not None:
                for fa in end:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    list_of_rxns.append(f"{dag.name} + {fa.name} => {tg1}")
            else:
                for fa in mid:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    list_of_rxns.append(f"{dag.name} + {fa.name} => {tg1}")

        return list_of_rxns

    @staticmethod
    def p_kinetic_interesterification(
        list_of_glycerides: List[Glyceride],
        initial_conc: List[int],
        plucked: List[str],
        arranged: List[str],
        ks: List[float],
        chem_flag=False,
    ) -> PKineticSim:
        """
        Will simulate the bath reaction for the given glyceride spieces.

        Parameters:
            list_of_glycerides (List[Glyceride]): List of the MAGs, DAGs, TAGs present in the reaction
            initial_conc (List[int]): List of initial concentrtions for each MAGs, DAGs, or TAGs
            chem_flag (bool): If True, divide by avogadro's number while calculating stochastic rate constants.

        Returns:
            PKineticSimulation: runs the graph to see what TAGs will be left
        """
        #
        react_stoic = []
        prod_stoic = []
        rxn_names: List[str] = []
        ks_internal = []

        # First break TAGs and form DAGs and FAs

        if ks is not None:
            if len(ks) != len(
                Interesterifier.p_kinetic_intersterification_rxn_list(
                    list_of_glycerides, arranged, plucked
                )
            ):
                raise ValueError("Make sure that you input the correct number of ks!")

        if len(initial_conc) != len(list_of_glycerides):
            raise ValueError("initial_conc must have the same length as list_of_fa")

            # Build all unique species once, and REUSE them when building reactions
        unique_dags = OrderedSet()
        dag_lookup: Dict[Tuple[str, int], Tuple[str, SymmetricGlyceride]] = (
            {}
        )  # (gly.name, fa.name, index) -> DAG + fatty acid

        # list of fatty acids that can only react to either the ends or the middles
        mid: List[FattyAcid] = []
        end: List[FattyAcid] = []

        for i in range(len(list_of_glycerides)):
            tag = list_of_glycerides[i]
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

        for i in range(len(list_of_glycerides)):
            if plucked[i] == arranged[i]:
                dag, fa = tag.remove_fatty_acid(index=1)
                tg_lookup[(dag.name, fa.name)] = list_of_glycerides[i]
            else:
                dag0, fa0 = tag.remove_fatty_acid(index=0)
                dag2, fa2 = tag.remove_fatty_acid(index=2)
                tg_lookup[(dag0.name, fa0.name)] = list_of_glycerides[i]
                tg_lookup[(dag2.name, fa2.name)] = list_of_glycerides[i]

        unique_species = list(OrderedSet.union(unique_dags, unique_tgs))
        midend = OrderedSet(mid + end)  # Combine the two lists together
        init_tags = [init_tags.name for init_tags in list_of_glycerides]
        fa_names = [fa.name for fa in midend]
        gly_names = [specie.name for specie in unique_species]
        species_names = [*fa_names, *OrderedSet(gly_names + init_tags)]
        species_idx = {nm: i for i, nm in enumerate(species_names)}

        ns = len(species_names)

        # build reaction names
        for i, tag in enumerate(list_of_glycerides):
            plucked_value = plucked[i]
            if plucked_value == "end":
                fa0, dag0 = dag_lookup[(tag.name, 0)]
                fa2, dag2 = dag_lookup[(tag.name, 2)]
                dag0 = dag0.name
                dag2 = dag2.name
                add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks_internal,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag0, fa0],
                    k=(ks.pop(0) if ks is not None else 1.0),
                    name=f"{tag.name} => {dag0} + {fa0}",
                )
                add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks_internal,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag2, fa2],
                    k=(ks.pop(0) if ks is not None else 1.0),
                    name=f"{tag.name} => {dag2} + {fa2}",
                )
            else:
                fa1, dag1 = dag_lookup[(tag.name, 1)]
                dag1 = dag1.name
                add_rxn(
                    react_stoic,
                    prod_stoic,
                    rxn_names,
                    ks_internal,
                    species_idx,
                    reactants=[tag.name],
                    products=[dag1, fa1],
                    k=(ks.pop(0) if ks is not None else 1.0),
                    name=f"{tag.name} => {dag1} + {fa1}",
                )

        # Build TG reactions (reuse prebuilt TGs)
        for dag in unique_dags:
            if dag.sn[1] is not None:
                for fa in end:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks_internal,
                        species_idx,
                        reactants=[dag.name, fa.name],
                        products=[tg1],
                        k=(ks.pop(0) if ks is not None else 2.0),
                        name=f"{dag.name} + {fa.name} => {tg1}",
                    )
            else:
                for fa in mid:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    add_rxn(
                        react_stoic,
                        prod_stoic,
                        rxn_names,
                        ks_internal,
                        species_idx,
                        reactants=[dag.name, fa.name],
                        products=[tg1],
                        k=(ks.pop(0) if ks is not None else 1.0),
                        name=f"{dag.name} + {fa.name} => {tg1}",
                    )

        # Initial state vector
        init_state = np.zeros(len(species_names), dtype=float)
        for gly, c0 in zip(list_of_glycerides, initial_conc[0:]):
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

        return PKineticSim(
            species_names=species_names,
            react_stoic=react_stoic,
            prod_stoic=prod_stoic,
            init_state=init_state,
            k_det=ks_internal,
            rxn_names=rxn_names,
            chem_flag=chem_flag,
        )
