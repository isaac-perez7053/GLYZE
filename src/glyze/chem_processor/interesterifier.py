from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from glyze.glyceride import Glyceride, FattyAcid, SymmetricGlyceride
from ordered_set import OrderedSet
from .p_kinetic_sim import PKineticSim
from glyze.utils import add_rxn

AVOGADRO = 6.02214076e23
CM3_PER_A3 = 1e-24


class Interesterifier:

    @staticmethod
    def _normalized_choice(value: str) -> str:
        value = str(value).strip().lower()
        if value not in {"end", "mid"}:
            raise ValueError(f"Expected 'end' or 'mid', got {value!r}")
        return value

    @staticmethod
    def _pluck_records(
        list_of_glycerides: List[Glyceride],
        plucked: List[str],
        arranged: List[str],
    ):
        if len(plucked) != len(list_of_glycerides) or len(arranged) != len(
            list_of_glycerides
        ):
            raise ValueError(
                "plucked and arranged must have the same length as list_of_glycerides"
            )

        unique_dags = OrderedSet()
        dag_lookup: Dict[Tuple[str, int], Tuple[str, SymmetricGlyceride]] = {}
        mid: List[FattyAcid] = []
        end: List[FattyAcid] = []
        removal_records = []

        for i, tag in enumerate(list_of_glycerides):
            plucked_i = Interesterifier._normalized_choice(plucked[i])
            arranged_i = Interesterifier._normalized_choice(arranged[i])

            if not isinstance(tag, Glyceride):
                continue

            occupied = [idx for idx, fa in enumerate(tag.sn) if fa is not None]
            if not occupied:
                continue

            if plucked_i == "mid":
                candidate_indices = [1] if tag.sn[1] is not None else []
            else:
                candidate_indices = [idx for idx in (0, 2) if tag.sn[idx] is not None]

            for idx in candidate_indices:
                dag, fa = tag.remove_fatty_acid(index=idx)
                unique_dags.add(dag)
                dag_lookup[(tag.name, idx)] = (fa.name, dag)
                removal_records.append(
                    {
                        "tag": tag,
                        "idx": idx,
                        "fa": fa,
                        "dag": dag,
                        "plucked": plucked_i,
                        "arranged": arranged_i,
                    }
                )
                if arranged_i == "end":
                    end.append(fa)
                else:
                    mid.append(fa)

        return unique_dags, dag_lookup, removal_records, mid, end

    @staticmethod
    def _build_tg_lookup(unique_dags, removal_records, mid, end):
        unique_tgs = OrderedSet()
        tg_lookup: Dict[Tuple[str, str], SymmetricGlyceride] = {}

        for dag in unique_dags:
            if dag.sn[1] is None:
                for fa in mid:
                    tg1 = dag.add_fatty_acid(index=1, fatty_acid=fa)
                    unique_tgs.add(tg1)
                    tg_lookup[(dag.name, fa.name)] = tg1
            else:
                for fa in end:
                    empty_ends = [idx for idx in (0, 2) if dag.sn[idx] is None]
                    if not empty_ends:
                        raise ValueError(
                            "Unexpected glyceride structure: no empty end position available."
                        )
                    tg1 = dag.add_fatty_acid(index=empty_ends[0], fatty_acid=fa)
                    unique_tgs.add(tg1)
                    tg_lookup[(dag.name, fa.name)] = tg1

        for rec in removal_records:
            tg_lookup[(rec["dag"].name, rec["fa"].name)] = rec["tag"]

        return unique_tgs, tg_lookup

    @staticmethod
    def interesterification_rxn_list(
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
        list_of_rxns = []
        init_ks = []

        unique_dags, dag_lookup, removal_records, mid, end = (
            Interesterifier._pluck_records(list_of_glycerides, plucked, arranged)
        )
        _, tg_lookup = Interesterifier._build_tg_lookup(
            unique_dags, removal_records, mid, end
        )

        for rec in removal_records:
            fa_name, dag = dag_lookup[(rec["tag"].name, rec["idx"])]
            list_of_rxns.append(f"{rec['tag'].name} => {dag.name} + {fa_name}")
            init_ks.append(1.0)

        for dag in unique_dags:
            if dag.sn[1] is not None:
                empty_end_count = sum(1 for idx in (0, 2) if dag.sn[idx] is None)
                k_default = float(max(empty_end_count, 1))
                for fa in end:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    list_of_rxns.append(f"{dag.name} + {fa.name} => {tg1}")
                    init_ks.append(k_default)
            else:
                for fa in mid:
                    tg1 = tg_lookup[(dag.name, fa.name)].name
                    list_of_rxns.append(f"{dag.name} + {fa.name} => {tg1}")
                    init_ks.append(1.0)

        return list_of_rxns, init_ks

    @staticmethod
    def interesterification(
        list_of_glycerides: List[Glyceride],
        initial_conc: List[int],
        plucked: List[str],
        arranged: List[str],
        ks: List[float],
        chem_flag=False,
        units: str = "moles",
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
        react_stoic = []
        prod_stoic = []
        rxn_names: List[str] = []
        ks_internal = []

        if len(initial_conc) != len(list_of_glycerides):
            raise ValueError(
                "initial_conc must have the same length as list_of_glycerides"
            )

        unique_dags, dag_lookup, removal_records, mid, end = (
            Interesterifier._pluck_records(list_of_glycerides, plucked, arranged)
        )
        unique_tgs, tg_lookup = Interesterifier._build_tg_lookup(
            unique_dags, removal_records, mid, end
        )

        unique_species = list(OrderedSet.union(unique_dags, unique_tgs))
        midend = OrderedSet(mid + end)
        init_tags = [
            init_tag.name
            for init_tag in list_of_glycerides
            if isinstance(init_tag, Glyceride)
        ]
        fa_names = [fa.name for fa in midend]
        gly_names = [specie.name for specie in unique_species]
        species_names = [*fa_names, *OrderedSet(gly_names + init_tags)]
        species_idx = {nm: i for i, nm in enumerate(species_names)}
        ns = len(species_names)

        ks_working = list(ks) if ks is not None else None

        for rec in removal_records:
            fa_name, dag = dag_lookup[(rec["tag"].name, rec["idx"])]
            add_rxn(
                react_stoic,
                prod_stoic,
                rxn_names,
                ks_internal,
                species_idx,
                reactants=[rec["tag"].name],
                products=[dag.name, fa_name],
                k=(ks_working.pop(0) if ks_working is not None else 1.0),
                name=f"{rec['tag'].name} => {dag.name} + {fa_name}",
            )

        for dag in unique_dags:
            if dag.sn[1] is not None:
                empty_end_count = sum(1 for idx in (0, 2) if dag.sn[idx] is None)
                k_default = float(max(empty_end_count, 1))
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
                        k=(ks_working.pop(0) if ks_working is not None else k_default),
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
                        k=(ks_working.pop(0) if ks_working is not None else 1.0),
                        name=f"{dag.name} + {fa.name} => {tg1}",
                    )

        init_state = np.zeros(len(species_names), dtype=float)
        for gly, c0 in zip(list_of_glycerides, initial_conc[0:]):
            if isinstance(gly, Glyceride):
                init_state[species_idx[gly.name]] = float(c0)

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
            units=units,
        )
