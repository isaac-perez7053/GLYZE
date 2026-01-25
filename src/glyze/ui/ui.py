from __future__ import annotations

import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List
from glyze import FattyAcid, Glyceride, GlycerideMix, ChemReactSim
from rdkit.Chem import Mol


# TODO: Store premade fatty acid objects in a json file
def make_palmitic() -> FattyAcid:
    return FattyAcid(length=16)


def make_stearic() -> FattyAcid:
    return FattyAcid(length=18)


def make_oleic() -> FattyAcid:
    return FattyAcid(length=18, db_positions=(9,), db_stereo=("Z",))


def make_linoleic() -> FattyAcid:
    return FattyAcid(length=18, db_positions=(9, 12), db_stereo=("Z", "Z"))


def make_linolenic() -> FattyAcid:
    return FattyAcid(length=18, db_positions=(9, 12, 15), db_stereo=("Z", "Z", "Z"))


def path_preset(path: str) -> List[List[FattyAcid], List[Glyceride]]:
    """Grab FattyAcids and Glycerides already built in a subdirectory"""
    pass


FA_PRESETS = {
    "Palmitic (16:0)": make_palmitic,
    "Stearic (18:0)": make_stearic,
    "Oleic (18:1 d9 cis)": make_oleic,
    "Linoleic (18:2 d9,12 cis,cis)": make_linoleic,
    "Linolenic (18:3 d9,12,15 cis,cis,cis)": make_linolenic,
}


def describe_fatty_acid(fa: FattyAcid) -> str:
    """Nice label for a FattyAcid."""
    name = getattr(fa, "name", "")
    if name:
        return name

    length = getattr(fa, "length", "?")
    db_positions = getattr(fa, "db_positions", ())
    if db_positions:
        db_str = ",".join(str(p) for p in db_positions)
        return f"{length}:{len(db_positions)} Δ{db_str}"
    return f"{length}:0"


def describe_glyceride(gly: Glyceride) -> str:
    """Nice label for a Glyceride."""
    name = getattr(gly, "name", "")
    if name:
        return name

    sn = getattr(gly, "sn", ())
    if sn:
        fa_labels = [describe_fatty_acid(fa) for fa in sn]
        return " / ".join(fa_labels)

    return repr(gly)


def rdkit_mol_to_plotly_3d(mol: Mol) -> go.Figure:
    """
    Convert an RDKit Mol with 3D coordinates to a simple 3D Plotly figure.
    Bonds are a bit thicker and atoms a bit bigger for clarity.
    """
    conf = mol.GetConformer()

    xs, ys, zs, symbols = [], [], [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
        symbols.append(atom.GetSymbol())

    element_colors = {
        "C": "black",
        "H": "lightgray",
        "O": "red",
        "N": "blue",
        "P": "orange",
        "S": "yellow",
    }
    colors = [element_colors.get(sym, "purple") for sym in symbols]

    # Bonds as line segments
    bond_x, bond_y, bond_z = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        xi, yi, zi = xs[i], ys[i], zs[i]
        xj, yj, zj = xs[j], ys[j], zs[j]

        bond_x += [xi, xj, None]
        bond_y += [yi, yj, None]
        bond_z += [zi, zj, None]

    fig = go.Figure()

    # Atoms (You can make them larger here if you want)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=7, color=colors),
            text=symbols,
            hoverinfo="text",
        )
    )

    # Bonds (thicker sticks, may make them thicker not sure)
    fig.add_trace(
        go.Scatter3d(
            x=bond_x,
            y=bond_y,
            z=bond_z,
            mode="lines",
            line=dict(width=5, color="dimgray"),
            hoverinfo="none",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def build_fatty_acid_mix_from_state():
    """
    Return a list of (FattyAcid, quantity) from session_state.
    Quantities are arbitrary units; you can normalize inside your package.
    """
    mix = []
    for fa, qty in zip(st.session_state.fatty_acids, st.session_state.fa_quantities):
        if qty and qty > 0:
            mix.append((fa, float(qty)))
    return mix


def build_glyceride_mix_from_state(units):
    """
    Build a GlycerideMix from current glycerides + quantities.
    """
    comps = []
    for gly, qty in zip(st.session_state.glycerides, st.session_state.gly_quantities):
        if qty and qty > 0:
            comps.append((gly, float(qty)))

    if not comps:
        return None

    mix = GlycerideMix(comps, units=units)
    return mix


def main():
    st.set_page_config(page_title="GLYZE - Glyceride Builder", layout="wide")

    # Initialize session state
    if "fatty_acids" not in st.session_state:
        st.session_state.fatty_acids = []
    if "fa_quantities" not in st.session_state:
        st.session_state.fa_quantities = []
    if "glycerides" not in st.session_state:
        st.session_state.glycerides = []
    if "gly_quantities" not in st.session_state:
        st.session_state.gly_quantities = []
    if "glyceride_mix" not in st.session_state:
        st.session_state.glyceride_mix = None
    if "fatty_acid_mix" not in st.session_state:
        st.session_state.glyceride_mix = None

    st.title("GLYZE - Glyceride and Lipid sYnthetiZation Engine")

    tab_builder, tab_chem, tab_visc = st.tabs(["Builder", "ChemProcessor / simulations", "Viscosity Empirical Model"])

    with tab_builder:
        st.sidebar.header("Fatty Acid Builder")

        st.sidebar.subheader("Preset fatty acids")
        preset_choice = st.sidebar.selectbox(
            "Choose a preset", options=list(FA_PRESETS.keys())
        )
        if st.sidebar.button("Add preset fatty acid"):
            try:
                fa = FA_PRESETS[preset_choice]()
                st.session_state.fatty_acids.append(fa)
                st.session_state.fa_quantities.append(1.0)
                st.sidebar.success(f"Added: {describe_fatty_acid(fa)}")
            except Exception as e:
                st.sidebar.error(f"Error adding preset fatty acid: {e}")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Custom fatty acid")

        # Input fatty acid parameters for custom fatty acids
        length = st.sidebar.number_input(
            "Length (number of carbons)",
            min_value=2,
            max_value=30,
            value=16,
        )
        db_positions_str = st.sidebar.text_input(
            "Double bond positions (comma-separated, e.g. 9,12)", value=""
        )
        db_stereo_str = st.sidebar.text_input(
            "Double bond stereochemistry (comma-separated, e.g. Z,Z)", value=""
        )
        # TODO: Test branches works correctly
        branches_str = st.sidebar.text_input(
            "Branch positions (comma-separated tuples, e.g. (pos1, type2); (pos2, type2))",
            value="",
        )

        # Button to add custom fatty acid with input processing
        if st.sidebar.button("Add custom fatty acid"):
            try:
                # Double bond positions and stereochemistry parsing
                if db_positions_str.strip():
                    db_positions = tuple(
                        int(p.strip()) for p in db_positions_str.split(",")
                    )
                else:
                    db_positions = ()

                # Double bond stereochemistry parsing
                if db_stereo_str.strip():
                    db_stereo = tuple(s.strip() for s in db_stereo_str.split(","))
                else:
                    db_stereo = ()

                # Branches parsing
                if branches_str.strip():
                    branches = []
                    for item in branches_str.split(";"):
                        item = item.strip().strip("()")
                        pos_type = item.split(",")
                        if len(pos_type) == 2:
                            pos = int(pos_type[0].strip())
                            btype = pos_type[1].strip()
                            branches.append((pos, btype))
                        else:
                            raise st.sidebar.error(f"Invalid branch format: {item}")
                    branches = tuple(branches)
                else:
                    branches = ()

                fa = FattyAcid(
                    length=int(length),
                    db_positions=db_positions,
                    db_stereo=db_stereo,
                    branches=branches,
                )

                st.session_state.fatty_acids.append(fa)

                # Begin with zero quantity unless specified otherwise
                st.session_state.fa_quantities.append(0.0)
                st.sidebar.success(f"Added: {describe_fatty_acid(fa)}")
            except Exception as e:
                st.sidebar.error(f"Error creating custom fatty acid: {e}")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Delete fatty acids")

        # Delete fatty acid selection and button
        fa_list = st.session_state.fatty_acids
        if fa_list:
            fa_labels = [
                f"{i}: {describe_fatty_acid(fa)}" for i, fa in enumerate(fa_list)
            ]
            fa_to_remove_idx = st.sidebar.selectbox(
                "Select fatty acid to delete",
                options=list(range(len(fa_list))),
                format_func=lambda i: fa_labels[i],
            )
            if st.sidebar.button("Delete selected fatty acid"):
                removed_fa = st.session_state.fatty_acids.pop(fa_to_remove_idx)
                st.session_state.fa_quantities.pop(fa_to_remove_idx)
                st.sidebar.success(f"Removed: {describe_fatty_acid(removed_fa)}")
        else:
            st.sidebar.info("No fatty acids yet. Add some above!")

        # Establish columns on main page builder tab
        col_left, col_right = st.columns([1, 2])

        # LEFT: lists, quantities, glyceride builder
        with col_left:
            st.subheader("Fatty acids & quantities")
            if st.session_state.fatty_acids:
                # Ensure units are set by the user
                st.caption("")
                units = st.selectbox(
                    "Select units for fatty acid quantities",
                    options=["mass normalized", "mole", "gram"],
                )
                for i, fa in enumerate(st.session_state.fatty_acids):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}** - {describe_fatty_acid(fa)}")
                    with col2:
                        qty_key = f"fa_qty_{i}"
                        current = float(st.session_state.fa_quantities[i])
                        st.session_state.fa_quantities[i] = st.number_input(
                            "Amt",
                            min_value=0.0,
                            value=current,
                            step=0.1,
                            key=qty_key,
                        )
            else:
                st.info("No fatty acids defined yet.")

            # Glyceride builder section
            st.markdown("---")
            st.subheader("Glyceride builder (sn-1 / sn-2 / sn-3)")

            if not st.session_state.fatty_acids:
                st.warning("Add at least one fatty acid to build glycerides.")
            else:
                # Store the fatty acid labels for selection
                fa_labels = [
                    f"{i}: {describe_fatty_acid(fa)}"
                    for i, fa in enumerate(st.session_state.fatty_acids)
                ]
                indices = list(range(len(st.session_state.fatty_acids)))

                # Select fatty acids for each sn position
                sn1_idx = st.selectbox(
                    "sn-1 fatty acid",
                    options=indices,
                    format_func=lambda i: fa_labels[i],
                )

                sn2_idx = st.selectbox(
                    "sn-2 fatty acid",
                    options=indices,
                    format_func=lambda i: fa_labels[i],
                )

                sn3_idx = st.selectbox(
                    "sn-3 fatty acid",
                    options=indices,
                    format_func=lambda i: fa_labels[i],
                )

                default_qty = 0.0
                gly_qty = st.number_input(
                    "Quantity to assign to this glyceride (on creation)",
                    min_value=0.0,
                    value=default_qty,
                    step=0.1,
                )

                if st.button("Create glyceride"):
                    # Ensure units are selected
                    if units is None:
                        st.error("Must select units for fatty acid quantities first.")
                    try:
                        fa1 = (
                            st.session_state.fatty_acids[sn1_idx]
                            if sn1_idx is not None
                            else None
                        )
                        fa2 = (
                            st.session_state.fatty_acids[sn2_idx]
                            if sn2_idx is not None
                            else None
                        )
                        fa3 = (
                            st.session_state.fatty_acids[sn3_idx]
                            if sn3_idx is not None
                            else None
                        )

                        gly = Glyceride(sn=(fa1, fa2, fa3))
                        st.session_state.glycerides.append(gly)
                        st.session_state.gly_quantities.append(float(gly_qty))
                        st.success(f"Added glyceride: {describe_glyceride(gly)}")
                    except Exception as e:
                        st.error(f"Error creating glyceride: {e}")

            st.markdown("---")
            st.subheader("Glycerides & quantities")
            if st.session_state.glycerides:
                for i, g in enumerate(st.session_state.glycerides):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}** - {describe_glyceride(g)}")
                    with col2:
                        qty_key = f"gly_qty_{i}"
                        current = float(st.session_state.gly_quantities[i])
                        st.session_state.gly_quantities[i] = st.number_input(
                            "Amt",
                            min_value=0.0,
                            value=current,
                            step=0.1,
                            key=qty_key,
                        )

                # Delete glyceride from list section
                st.markdown("**Delete a glyceride**")
                gly_labels = [
                    f"{i}: {describe_glyceride(g)}"
                    for i, g in enumerate(st.session_state.glycerides)
                ]

                gly_to_remove_idx = st.selectbox(
                    "Select glyceride to delete",
                    options=list(range(len(st.session_state.glycerides))),
                    format_func=lambda i: gly_labels[i],
                    key="delete_gly_select",
                )
                if st.button("Delete selected glyceride"):
                    removed_gly = st.session_state.glycerides.pop(gly_to_remove_idx)
                    st.session_state.gly_quantities.pop(gly_to_remove_idx)
                    st.success(f"Removed glyceride: {describe_glyceride(removed_gly)}")
            else:
                st.info("No glycerides created yet.")

        # RIGHT: viewer for a selected glyceride
        with col_right:
            st.subheader("Visualize a glyceride")

            if not st.session_state.glycerides:
                st.info("Create a glyceride on the left to visualize it here.")
            else:
                gly_labels = [
                    f"{i}: {describe_glyceride(g)}"
                    for i, g in enumerate(st.session_state.glycerides)
                ]
                selected_gly_idx = st.selectbox(
                    "Choose glyceride to plot",
                    options=list(range(len(st.session_state.glycerides))),
                    format_func=lambda i: gly_labels[i],
                    key="view_gly_select",
                )
                selected_gly = st.session_state.glycerides[selected_gly_idx]

                st.markdown(f"**Selected:** {describe_glyceride(selected_gly)}")

                name = getattr(selected_gly, "name", None)
                molar_mass = getattr(selected_gly, "molar_mass", None)
                if name is not None:
                    st.markdown(f"**Name:** {name}")
                if molar_mass is not None:
                    st.markdown(f"**Molar mass:** {molar_mass:.3f} g/mol")

                with st.expander("Show Python repr"):
                    st.code(repr(selected_gly))

                st.markdown("### 3D structure")
                try:
                    mol = selected_gly.glyceride_to_rdkit(optimize=True)
                    fig = rdkit_mol_to_plotly_3d(mol)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating 3D structure: {e}")

    with tab_chem:
        st.subheader("Batch Reaction Processor")
        input_type = st.radio(
            "Choose Process",
            ("Interesterification", "Esterification", "Deodorization"),
        )

        # Glyceride Mix Builder
        # TODO: Must add plucked and arranged, test plotly graph output and make sure fatty acid mixture works
        plucked = []
        arranged = []
        if input_type.startswith("Interesterification"):
            st.markdown("### GlycerideMix from current glycerides")
            if not st.session_state.glycerides:
                st.info("No glycerides defined - go to the Builder tab first.")
            else:
                if st.button("Create / update GlycerideMix from glycerides"):
                    mix = build_glyceride_mix_from_state(units=units)
                    if mix is None:
                        st.warning("All glyceride quantities are zero.")
                    else:
                        st.session_state.glyceride_mix = mix
                        st.success(
                            "Updated `glyceride_mix` in session_state "
                            "(st.session_state.glyceride_mix)."
                        )

                if st.session_state.glyceride_mix is not None:
                    st.markdown("**Current GlycerideMix (repr):**")
                    st.code(repr(st.session_state.glyceride_mix))
                else:
                    st.info("No GlycerideMix has been built yet.")

        elif input_type.startswith("Esterification"):
            st.markdown("### FattyAcid mixture from current fatty acids")
            # Define Glycerol quantity first
            glycerol_qty = st.number_input(
                label=f"Glycerol Quantity in {units}", min_value=0.0, step=0.1
            )
            # Define fatty acid quantities
            if not st.session_state.fatty_acids:
                st.info("No fatty acids defined - go to the Builder tab first")
            else:
                if st.button("Create / update FattyAcid mixture from fatty acids"):
                    fa_mix = build_fatty_acid_mix_from_state()
                    if not fa_mix:
                        st.warning("All fatty-acid quantities are zero.")
                    else:
                        st.markdown("**Preview of fatty-acid mix:**")
                        # Esablish fatty acid mix with Glycerol
                        st.session_state.fa_mix = [("Glycerol", glycerol_qty), fa_mix]
                        for fa, q in fa_mix:
                            st.markdown(
                                f"- {describe_fatty_acid(fa)} -- amount = {q} ({units})"
                            )

        else:
            st.markdown(
                "### GlycerideMix and fatty acids from current glycerides and fatty acids"
            )

        st.markdown("---")
        st.subheader("Interesterification simulation (example)")

        temperature = st.number_input("Temperature (K)", value=373.15)
        pressure = st.number_input("Pressure (bar)", value=1.0)
        time = st.number_input("Time (sec)", value=60.0)
        n_cycles = st.number_input(
            "Number of cycles / steps", min_value=1, max_value=100000, value=100
        )
        random_seed = st.number_input("Random seed", value=42, step=1)

        if st.button("Run"):
            try:
                # Interesterification run option
                if input_type.startswith("Interesterification"):
                    if st.session_state.glyceride_mix is None:
                        st.error("You need to build a GlycerideMix first.")
                        return
                    else:
                        # Build quantity list from Glyercide Mixture
                        qty_list = []
                        for _, qty in st.session_state.glyceride_mix:
                            qty_list.append(qty)
                        sim = ChemReactSim.p_kinetic_interesterification(
                            list_of_stuff=st.session_state.glyceride_mix.glyceride_list,
                            initial_conc=qty_list,
                        )
                        # Solve and plot results
                        try:
                            sim.solve(t_span=time)
                            plot = sim.plot_interactive()
                            st.plotly_chart(plot, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating 3D structure: {e}")

                elif input_type.startswith("Esterification"):
                    if st.session_state.fatty_acid_mix is None:
                        st.error("You need a fatty acid mixture first")
                        return

                st.success("Interesterification simulation completed.")
            except Exception as e:
                st.error(f"Error running ChemProcessor: {e}")
    
    
    with tab_visc:
        #st.subheader("Empirical Viscosity Predictions")
        # put viscosity UI here
            #original data/ coding logic from MATLAB
            COEFS = {
        "C2":  (0.07, 105.2911, 0.0112, -885.4359),
        "C3":  (-4.056, 967.9563, 134.5872, 31.4778),
        "C4":  (-0.384050317, 122.5512048, -6.489359596, -2543.72728),
        "C6":  (-0.4725, 156.0213, -7.1172, -3537.5806),
        "C7":  (0.0084, 127.5008, -6.3182, -2561.4509),
        "C8":  (0.1859, 129.532, -6.0275, -2510.8992),
        "C9":  (0.0493, 149.653, -6.5789, -3139.137),
        "C10": (0.3359649655, 143.6640376, -6.245550956, -2870.995961),
        "C11": (-0.6474906112, 250.18623, -9.343606644, -7309.246012),
        "C12": (-0.7105139749, 273.087563, -9.038660766, -7891.211795),
        "C13": (-1.05060727, 336.5282457, -11.49439642, -12026.74479),
        "C14": (-4.654706547, 977.63118, -18.92319412, -59170.32269),
        "C15": (-0.9835811776, 341.2594921, 0.06795266312, -6745.162413),
        "C16": (-2.049637695, 525.3225883, 0.1003348094, -13801.58031),
        "C17": (14.86066425, -2879.173742, -26.60583529, 266728.0745),
        "C18": (-7.892604764, 1639.942578, -20.20174222, -109208.2254),
        "C19": (-27.63059908, 5210.787366, -22.05698668, -382904.7109),
    }

    MOLAR_MASS = {
        "C2": 218.18, "C3": 260.28, "C4": 302.367, "C5": 344.448, "C6": 386.529,
        "C7": 428.61, "C8": 470.691, "C9": 512.772, "C10": 554.853, "C11": 596.934,
        "C12": 639.015, "C13": 681.096, "C14": 723.177, "C15": 765.258, "C16": 807.339,
        "C17": 849.42, "C18": 891.501, "C19": 933.582
    }

    def tags_from(s: str):
        return [t for t in re.split(r"[,\s]+", (s or "").strip().upper()) if t]

    def floats_from(s: str):
        vals = [x for x in re.split(r"[,\s]+", (s or "").strip()) if x]
        return np.array([float(x) for x in vals], dtype=float) if vals else np.array([], dtype=float)

    def range_for(tag: str):
        n = int(tag[1:])
        if 2 <= n <= 10: return 20.0, 80.0
        if 11 <= n <= 13: return 30.0, 80.0
        if 14 <= n <= 16: return 50.0, 80.0
        if 17 <= n <= 19: return 70.0, 80.0
        raise ValueError(tag)

    def mu(T, A, B, C, E):
        T = np.asarray(T, float)
        return np.exp(A + B / (T + C) + E / (T**2))

    def mass_to_mole_frac(tags, mass_fracs):
        w = np.asarray(mass_fracs, float)
        w = w / w.sum()
        M = np.array([MOLAR_MASS[t] for t in tags], float)
        n = w / M
        return n / n.sum()

    st.set_page_config(page_title="Viscosity Mixer")
    st.title("Viscosity vs Temperature - Mixture Widget")

    with st.expander("Valid temperature ranges by tag", expanded=False):
        st.write(
            "- C2-C10: 20-80 °C\n"
            "- C11-C13: 30-80 °C\n"
            "- C14-C16: 50-80 °C\n"
            "- C17-C19: 70-80 °C"
        )

    # Defaults (example inputs)
    default_tags = "C3 C4 C6"
    default_mf = "0.30 0.40 0.30"
    default_tr = "60 70 1"

    tags_str = st.text_input("TAGS to mix (e.g., C3 C4 C6)", value=default_tags)
    mf_str = st.text_input("Mass fractions (same order as TAGS)", value=default_mf)
    tr_str = st.text_input("Temperature range °C: start end [step] (e.g., 60 70 1)", value=default_tr)

    auto_run = st.toggle("Auto-run when inputs are valid (press Enter after typing)", value=True)

    run_clicked = st.button("Run / Recompute", type="primary", disabled=auto_run)

    def validate_and_compute(tags_str, mf_str, tr_str):
        tags = tags_from(tags_str)
        if not tags:
            return None, "Alert: no TAGS provided."

        bad = [t for t in tags if (t not in COEFS) or (t not in MOLAR_MASS)]
        if bad:
            return None, f"Alert: unknown TAGS: {', '.join(bad)}"

        mf = floats_from(mf_str)
        if mf.size != len(tags):
            return None, f"Alert: expected {len(tags)} mass-fraction values."

        if np.any(mf < 0) or np.isclose(mf.sum(), 0.0):
            return None, "Alert: mass fractions must be nonnegative and not all zero."

        tr = floats_from(tr_str)
        if tr.size not in (2, 3):
            return None, "Alert: enter start end [step]."

        t0, t1 = float(tr[0]), float(tr[1])
        step = float(tr[2]) if tr.size == 3 else 1.0

        if step <= 0 or t1 < t0:
            return None, "Alert: invalid range/step."

        invalid = []
        for t in tags:
            lo, hi = range_for(t)
            if not (lo <= t0 and t1 <= hi):
                invalid.append((t, lo, hi))
        if invalid:
            msg = "Alert: out of range for: " + "; ".join([f"{t} ({lo:g}–{hi:g}°C)" for t, lo, hi in invalid])
            return None, msg

        # Compute
        x = mass_to_mole_frac(tags, mf)
        T = np.arange(t0, t1 + 0.5 * step, step)
        U = np.vstack([mu(T, *COEFS[t]) for t in tags])
        u_mix = np.exp(x @ np.log(U))

        return (tags, mf, x, T, U, u_mix), None

    should_run = auto_run or run_clicked
    if should_run:
        result, err = validate_and_compute(tags_str, mf_str, tr_str)
        if err:
            st.error(err)
        else:
            tags, mf, x, T, U, u_mix = result

            st.subheader("Parsed inputs")
            st.write(f"**Tags:** {', '.join(tags)}")
            st.write(f"**Mass fractions (w):** {mf / mf.sum()}")
            st.write(f"**Mole fractions (x):** {x}")

            fig = plt.figure()
            plt.grid(True)
            for i, t in enumerate(tags):
                plt.plot(T, U[i], linewidth=2, label=t)
            plt.plot(T, u_mix, "k--", linewidth=1.5, label="Mixture")
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Viscosity (cP)")
            plt.title("Viscosity vs Temperature")
            plt.legend(loc="best")
            plt.tight_layout()

            left, mid, right = st.columns([1, 2, 1])  # <-- change middle width by adjusting the 2
            with mid:
                 st.pyplot(fig, use_container_width=True, clear_figure=True)

            st.download_button(
                "Download computed data (CSV)",
                data=("T_C," + ",".join(tags) + ",Mixture\n" + "\n".join(
                    [",".join([f"{T[i]:g}"] + [f"{U[j,i]:.8g}" for j in range(len(tags))] + [f"{u_mix[i]:.8g}"])
                    for i in range(len(T))]
                )),
                file_name="viscosity_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Edit inputs and press Enter (or toggle Auto-run), then run the calculation.")

if __name__ == "__main__":
    main()
