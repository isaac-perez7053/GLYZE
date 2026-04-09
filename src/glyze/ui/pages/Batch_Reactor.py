import streamlit as st
from glyze import Interesterifier, PKineticSim, Esterifier
import numpy as np
from glyze.glyceride import Glyceride, FattyAcid


from glyze.ui.pages.Glyceride_Mix import PAGE_CSS, MixRow, rows_to_display_df


st.set_page_config(page_title="GLYZE — Batch Reactor", page_icon="🧈", layout="wide")

# Reuse the exact style from home.py
st.markdown(
    """
    <style>
    :root{
        --accentA:#2563EB;
        --accentB:#7C3AED;
        --text:#0F172A;
        --muted:#475569;
        --border:rgba(15,23,42,0.12);
    }

    .stApp{
        background:
            radial-gradient(1100px circle at 15% 0%, rgba(255,237,213,0.60) 0%, rgba(255,237,213,0) 55%),
            radial-gradient(900px circle at 90% 10%, rgba(219,234,254,0.70) 0%, rgba(219,234,254,0) 55%),
            #ffffff;
        color:var(--text);
    }

    section.main > div{
        max-width: 1320px;
        padding-top: 1.2rem;
        padding-bottom: 2.8rem;
    }

    .glyze-card{
        padding: 1.4rem 1.6rem;
    }

    .center{ text-align:center; }

    .glyze-hero{ margin-top:0.2rem; margin-bottom:0.6rem; }

    .glyze-logo{
        display:inline-flex;
        align-items:center;
        gap:12px;
        padding:10px 14px;
        border-radius:999px;
        border:1px solid rgba(15,23,42,0.08);
        background:rgba(255,255,255,0.65);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        box-shadow:0 10px 24px rgba(15,23,42,0.06);
    }

    .glyze-word{
        font-weight:900;
        letter-spacing:0.02em;
        font-size:2.2rem;
        line-height:1.0;
        background:linear-gradient(90deg, #0F172A 0%, #1D4ED8 55%, #7C3AED 100%);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        margin:0;
    }

    .butter-badge{
        width:42px; height:42px;
        border-radius:14px;
        display:grid; place-items:center;
        background:linear-gradient(145deg, rgba(255,237,213,0.95), rgba(219,234,254,0.85));
        border:1px solid rgba(15,23,42,0.08);
        box-shadow:0 10px 22px rgba(15,23,42,0.08);
        font-size:20px;
    }

    .glyze-tagline{
        margin:0.65rem auto 0.2rem auto;
        max-width:64ch;
        color:var(--muted);
        font-size:1.0rem;
    }

    .glyze-divider{
        width: min(640px, 92%);
        height: 1px;
        margin: 1.1rem auto 1.0rem auto;
        background: linear-gradient(to right,
            rgba(15,23,42,0),
            rgba(15,23,42,0.20),
            rgba(15,23,42,0)
        );
        border-radius:999px;
        position:relative;
    }
    .glyze-divider:after{
        content:"";
        position:absolute;
        left:50%;
        top:50%;
        transform:translate(-50%,-50%);
        width:10px;
        height:10px;
        border-radius:999px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(15,23,42,0.14);
        box-shadow: 0 6px 16px rgba(15,23,42,0.08);
    }

    .glyze-card{
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(15,23,42,0.10);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
        backdrop-filter: blur(7px);
        -webkit-backdrop-filter: blur(7px);
        margin-top: 0.9rem;
    }

    .start-title{
        text-align:center;
        font-size:1.65rem;
        font-weight:800;
        margin:0.2rem 0 0.25rem 0;
        color:var(--text);
    }
    .start-subtitle{
        text-align:center;
        color:var(--muted);
        margin-bottom:0.85rem;
    }

    div.stButton > button{
        width: 100%;
        border-radius: 16px !important;
        padding: 0.95rem 1.05rem !important;
        font-weight: 750 !important;
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06);
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, filter 120ms ease;
    }

    div.stButton > button:hover{
        transform: translateY(-1px);
        background: linear-gradient(90deg, var(--accentA), var(--accentB)) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.20) !important;
        box-shadow: 0 18px 35px rgba(37, 99, 235, 0.22);
        filter: brightness(1.02);
    }

    div.stButton > button:active{
        transform: translateY(0px);
        box-shadow: 0 10px 18px rgba(15,23,42,0.10);
    }

    button[kind="primary"]{
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06) !important;
    }

    h3{
        margin-top: 0.2rem !important;
        margin-bottom: 0.6rem !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06) !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06) !important;
    }

    div[data-testid="stSelectbox"] * ,
    div[data-testid="stMultiSelect"] * {
        color: var(--text) !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
        border-color: rgba(37,99,235,0.55) !important;
        box-shadow:
            0 10px 20px rgba(15,23,42,0.06),
            0 0 0 4px rgba(37,99,235,0.18) !important;
    }

    div[data-baseweb="select"] span {
        background: transparent !important;
    }
    :root{
    --field-bg: rgba(255,255,255,0.88);
    --field-border: rgba(15,23,42,0.12);
    --field-shadow: 0 10px 20px rgba(15,23,42,0.06);
    --focus-ring: 0 0 0 4px rgba(37,99,235,0.18);
    }

    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] summary{
    background: var(--field-bg) !important;
    border: 1px solid var(--field-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--field-shadow) !important;
    }

    div[data-testid="stExpander"] summary{
    padding: 0.85rem 1.0rem !important;
    }

    div[data-testid="stExpander"] summary *{
    color: var(--text) !important;
    }

    div[data-testid="stExpander"] summary:hover{
    border-color: rgba(37,99,235,0.22) !important;
    }

    div[data-testid="stExpander"] summary svg{
    fill: var(--text) !important;
    color: var(--text) !important;
    }

    div[data-testid="stExpander"] div[role="region"]{
    background: transparent !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input{
    background: var(--field-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--field-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--field-shadow) !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div{
    background: var(--field-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--field-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--field-shadow) !important;
    }

    ul[role="listbox"],
    div[data-baseweb="popover"]{
    background: rgba(255,255,255,0.96) !important;
    color: var(--text) !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 40px rgba(15,23,42,0.10) !important;
    }

    li[role="option"]{
    color: var(--text) !important;
    }
    li[role="option"][aria-selected="true"]{
    background: rgba(37,99,235,0.10) !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within{
    border-color: rgba(37,99,235,0.55) !important;
    box-shadow: var(--field-shadow), var(--focus-ring) !important;
    }

    div[data-testid="stNumberInput"] button{
    background: var(--field-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--field-border) !important;
    box-shadow: none !important;
    }
    div[data-testid="stExpander"] summary{
  filter: none !important;
  -webkit-text-fill-color: var(--text) !important;
    }
    div[data-testid="stNumberInput"] button:hover{
    border-color: rgba(37,99,235,0.35) !important;
    }

    div[data-testid="stDataEditor"]{
    border-radius: 16px !important;
    }

    div[data-testid="stDataEditor"] [role="grid"],
    div[data-testid="stDataEditor"] .stDataFrame,
    div[data-testid="stDataEditor"] [data-testid="stTable"]{
    background: rgba(255,255,255,0.92) !important;
    color: var(--text) !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 16px !important;
    box-shadow: var(--field-shadow) !important;
    }

    div[data-testid="stDataEditor"] [role="columnheader"]{
    background: rgba(248,250,252,0.95) !important;
    color: var(--text) !important;
    border-bottom: 1px solid rgba(15,23,42,0.08) !important;
    }

    div[data-testid="stDataEditor"] [role="gridcell"]{
    background: rgba(255,255,255,0.90) !important;
    color: var(--text) !important;
    border-bottom: 1px solid rgba(15,23,42,0.06) !important;
    }

    div[data-testid="stDataEditor"] [data-testid="stDataEditorEmpty"]{
    color: rgba(71,85,105,0.85) !important;
    }

    div[data-testid="stDataEditor"] [role="gridcell"][aria-selected="true"],
    div[data-testid="stDataEditor"] [role="row"][aria-selected="true"] [role="gridcell"]{
    background: rgba(37,99,235,0.10) !important;
    outline: none !important;
    }

    div[data-testid="stDataEditor"] input{
    background: rgba(255,255,255,0.98) !important;
    color: var(--text) !important;
    border: 1px solid rgba(37,99,235,0.35) !important;
    box-shadow: var(--focus-ring) !important;
    }

    div[data-testid="stAlert"]{
    border-radius: 16px !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    box-shadow: 0 14px 28px rgba(15,23,42,0.07) !important;
    }
    div[data-testid="stAlert"] *{
    color: var(--text) !important;
    }

    div[data-testid="stSlider"] [role="slider"],
    div[data-testid="stSlider"] [data-baseweb="slider"]{
    color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


HERO_HTML = """
<div class="center glyze-hero">
    <div class="glyze-logo">
        <div class="glyze-word">GLYZE</div>
        <div class="butter-badge">🧈</div>
    </div>
    <div class="glyze-tagline">
        Build and export a glyceride mixture — add by name or by structure, edit concentrations, switch units.
    </div>
</div>
<div class="glyze-divider"></div>
"""


st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)


def get_mix_rows():
    return st.session_state.setdefault("mix_rows", [])


def get_processor_rows():
    if "processor_mix_rows" in st.session_state:
        return st.session_state["processor_mix_rows"]
    return get_mix_rows()


def unwrap_component(comp):
    """Return the underlying chemical object if wrapped in MixtureComponent."""
    if hasattr(comp, "component"):
        return comp.component
    return comp


def comp_display_name(comp) -> str:
    comp = unwrap_component(comp)
    if comp in {"H2O", "Glycerol"}:
        return str(comp)
    if hasattr(comp, "name"):
        return str(comp.name)
    return str(comp)


def is_fatty_acid_component(comp) -> bool:
    return isinstance(unwrap_component(comp), FattyAcid)


def is_glyceride_component(comp) -> bool:
    return isinstance(unwrap_component(comp), Glyceride)


def split_interesterification_rows(rows):
    """Separate reactive glycerides from spectator species for interesterification."""
    reactive_rows = []
    inert_rows = []

    for row in rows:
        if is_glyceride_component(row.comp):
            reactive_rows.append(MixRow(unwrap_component(row.comp), float(row.moles)))
        else:
            inert_rows.append(MixRow(unwrap_component(row.comp), float(row.moles)))

    return reactive_rows, inert_rows


def merge_mix_rows(rows_a, rows_b):
    """Merge rows by display name while preserving a representative component object."""
    merged = {}

    for row in [*rows_a, *rows_b]:
        comp = unwrap_component(row.comp)
        name = comp_display_name(comp)
        if name not in merged:
            merged[name] = {"comp": comp, "Moles": 0.0}
        merged[name]["Moles"] += float(row.moles)

    return [
        MixRow(v["comp"], v["Moles"])
        for _, v in sorted(merged.items(), key=lambda kv: kv[0])
    ]


def initialize_reactor_state(prefix: str):
    st.session_state[f"{prefix}_initialized"] = False
    st.session_state[f"{prefix}_reactions_initialized"] = False
    st.session_state[f"{prefix}_sim_initialized"] = False
    st.session_state[f"{prefix}_sim_ran"] = False
    st.session_state[f"{prefix}_species"] = []
    st.session_state[f"{prefix}_init_concentrations"] = []
    st.session_state[f"{prefix}_species_names"] = []
    st.session_state[f"{prefix}_mixture_eqns"] = []
    st.session_state[f"{prefix}_init_consts"] = []
    st.session_state[f"{prefix}_ks"] = []
    st.session_state[f"{prefix}_results_df"] = None
    st.session_state[f"{prefix}_time_points"] = None
    st.session_state[f"{prefix}_sim_object"] = None
    st.session_state[f"{prefix}_solution"] = None


def build_results_dataframe(sim: PKineticSim):
    import pandas as pd

    sol = sim.sol
    results_dict = {"Time": sol.t}
    for i, species_name in enumerate(sim.species_names):
        results_dict[str(species_name)] = sol.y[i, :]
    return pd.DataFrame(results_dict)


def build_cv_from_sim_to_csv(sim: PKineticSim):
    import pandas as pd
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)

    try:
        sim.to_csv(str(tmp_path))
        cv_df = pd.read_csv(tmp_path)
        cv_bytes = tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return cv_df, cv_bytes


def build_mix_preview_dataframe(rows):
    import pandas as pd

    return pd.DataFrame(
        {
            "Species": [comp_display_name(r.comp) for r in rows],
            f" {st.session_state.get('display_unit', 'Moles')}": [
                r.moles for r in rows
            ],
        }
    )


def build_carbon_number_distribution(sim: PKineticSim):
    import pandas as pd
    from glyze.glyceride import Glyceride, FattyAcid

    if not hasattr(sim, "sol"):
        raise ValueError(
            "Simulation results not found. Please run the simulation first."
        )

    carbon_rows = []

    for i, name in enumerate(sim.species_names):
        species_name = str(name)
        final_conc = float(sim.sol.y[i, -1])

        if species_name in ("H2O", "Water"):
            continue

        try:
            if species_name == "Glycerol":
                molar_mass = 92.094
                carbon_number = 3
            elif species_name.count("_") == 3:
                gly_obj = Glyceride.from_name(species_name)
                molar_mass = gly_obj.molar_mass
                carbon_number = gly_obj.num_carbons
            else:
                fa_obj = FattyAcid.from_name(species_name)
                molar_mass = fa_obj.molar_mass
                carbon_number = fa_obj.num_carbons

            final_mass = final_conc * molar_mass

            carbon_rows.append(
                {
                    "Species": species_name,
                    "Carbon Number": carbon_number,
                    "Final Concentration": final_conc,
                    "Final Mass (g)": final_mass,
                }
            )
        except Exception:
            continue

    if len(carbon_rows) == 0:
        return pd.DataFrame(
            columns=["Carbon Number", "Final Mass (g)", "Weight by Carbon Number"]
        )

    carbon_df = pd.DataFrame(carbon_rows)

    grouped_df = (
        carbon_df.groupby("Carbon Number", as_index=False)["Final Mass (g)"]
        .sum()
        .sort_values("Carbon Number")
    )

    total_mass = grouped_df["Final Mass (g)"].sum()
    if total_mass > 0:
        grouped_df["Weight by Carbon Number"] = (
            grouped_df["Final Mass (g)"] / total_mass
        )
    else:
        grouped_df["Weight by Carbon Number"] = 0.0

    return grouped_df


def upload_resulting_mixture(prefix: str, success_message: str):
    result_mix_obj = st.session_state.get(f"{prefix}_result_mix_object", None)
    result_mix_rows = st.session_state.get(f"{prefix}_result_mix_rows", None)

    if result_mix_obj is None or result_mix_rows is None:
        raise ValueError("No resulting mixture found. Please run the simulation first.")

    st.session_state["glyze_mix_object"] = result_mix_obj
    st.session_state["glyze_mix_rows"] = result_mix_rows
    st.session_state["mix_rows"] = result_mix_rows
    st.session_state["glyze_mix_initialized"] = True
    st.session_state["processor_mix_rows"] = result_mix_rows
    st.session_state["esterifier_mix_rows"] = result_mix_rows
    st.session_state["esterifier_mix_object"] = result_mix_obj
    st.session_state["esterifier_initialized"] = True
    st.success(success_message)


def show_simulation_results(prefix: str):
    sim = st.session_state[f"{prefix}_sim_object"]
    results_df = st.session_state[f"{prefix}_results_df"]

    st.subheader("Simulation Results")
    st.divider()
    st.subheader("Send resulting mixture")

    if st.button(
        "Upload Resulting Mixture",
        use_container_width=True,
        key=f"{prefix}_send_mix_to_gmix",
    ):
        try:
            upload_resulting_mixture(
                prefix,
                "Resulting mixture sent to Glyceride Mix page and made available to the other reactor tab.",
            )
        except Exception as e:
            st.error(str(e))

    results_mode = st.radio(
        "Display results as:",
        ["Plot", "CV DataFrame", "Carbon Number Bar Chart"],
        horizontal=True,
        key=f"{prefix}_results_mode",
    )

    if results_mode == "Plot":
        try:
            plot_fig = sim.plot_interactive(
                st.session_state[f"{prefix}_solution"],
                return_fig=True,
            )
            st.plotly_chart(plot_fig, use_container_width=True)
        except Exception:
            plot_df = results_df.set_index("Time")
            st.line_chart(plot_df)

        csv_name = st.text_input(
            "Enter filename for CSV (without extension)",
            value=f"{prefix}_simulation_results",
            key=f"{prefix}_plot_csv_filename",
        )
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Simulation Results",
            data=csv_bytes,
            file_name=csv_name + ".csv" if not csv_name.endswith(".csv") else csv_name,
            mime="text/csv",
            use_container_width=True,
            key=f"{prefix}_plot_csv_download",
        )

    elif results_mode == "CV DataFrame":
        try:
            cv_df, cv_bytes = build_cv_from_sim_to_csv(sim)
            st.dataframe(cv_df, use_container_width=True)

            st.download_button(
                "Download CV CSV",
                data=cv_bytes,
                file_name=f"{prefix}_cv_results.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{prefix}_cv_download",
            )
        except Exception as e:
            st.error(f"Could not build CV from sim.to_csv: {str(e)}")

    else:
        carbon_df = build_carbon_number_distribution(sim)

        if carbon_df.empty:
            st.warning(
                "No carbon-number data could be constructed from the simulation output."
            )
        else:
            import plotly.express as px

            carbon_fig = px.bar(
                carbon_df,
                x="Carbon Number",
                y="Weight by Carbon Number",
                hover_data=["Final Mass (g)"],
            )
            carbon_fig.update_layout(
                xaxis_title="Carbon Number",
                yaxis_title="Weight by Carbon Number",
            )
            st.plotly_chart(carbon_fig, use_container_width=True)
            st.dataframe(carbon_df, use_container_width=True)

            csv_name = st.text_input(
                "Enter filename for CSV (without extension)",
                value=f"{prefix}_simulation_results",
                key=f"{prefix}_carbon_csv_filename",
            )

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Simulation Results",
                data=csv_bytes,
                file_name=(
                    csv_name + ".csv" if not csv_name.endswith(".csv") else csv_name
                ),
                mime="text/csv",
                use_container_width=True,
                key=f"{prefix}_carbon_csv_download",
            )


def split_esterification_rows(rows):
    """Separate fatty-acid feed rows from glycerol rows.

    The uploaded class definitions show that mixture entries may be stored as
    MixtureComponent wrappers around FattyAcid/Glyceride/string payloads, so we
    unwrap before classification.
    """
    fatty_acid_rows = []
    glycerol_moles = 0.0

    for row in rows:
        comp = unwrap_component(row.comp)
        name = comp_display_name(comp)
        if isinstance(comp, FattyAcid):
            fatty_acid_rows.append(MixRow(comp, float(row.moles)))
        elif name == "Glycerol":
            glycerol_moles += float(row.moles)

    return fatty_acid_rows, glycerol_moles


def render_interesterification_tab():
    prefix = "inter"
    rows = get_processor_rows()

    if not rows:
        st.info("Please build or import a mixture on the Glyceride Mix page first.")
        return

    st.write(f"Loaded {len(rows)} species from the current mixture.")
    st.subheader("Current Glyceride Mix")
    # Display rows in dataframe in the middle of the screen
    df = rows_to_display_df(rows, st.session_state.get("display_unit", "Moles"))
    st.dataframe(df, use_container_width=True)

    # Start first 3 columns with TAG mixture
    if st.button("Initialize with Mixture", key=f"{prefix}_initialize_mixture_button"):
        reactive_rows, inert_rows = split_interesterification_rows(rows)

        if not reactive_rows:
            st.warning(
                "The current mixture does not contain any glycerides to interesterify. "
                "Only MAGs, DAGs, and TAGs use plucked/arranged settings."
            )
        else:
            st.session_state[f"{prefix}_species"] = [
                unwrap_component(r.comp) for r in reactive_rows
            ]
            st.session_state[f"{prefix}_init_concentrations"] = [
                r.moles for r in reactive_rows
            ]
            st.session_state[f"{prefix}_species_names"] = [
                comp_display_name(r.comp) for r in reactive_rows
            ]
            st.session_state[f"{prefix}_inert_rows"] = inert_rows
            st.session_state[f"{prefix}_inert_species_names"] = [
                comp_display_name(r.comp) for r in inert_rows
            ]
            st.session_state[f"{prefix}_initialized"] = True
            st.session_state[f"{prefix}_reactions_initialized"] = False
            st.session_state[f"{prefix}_sim_initialized"] = False
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Mixture initialized for the interesterifier.")

    if st.session_state[f"{prefix}_initialized"]:
        species = st.session_state[f"{prefix}_species"]
        names = st.session_state[f"{prefix}_species_names"]
        inert_names = st.session_state.get(f"{prefix}_inert_species_names", [])

        if inert_names:
            st.info(
                "These species are carried through unchanged during interesterification: "
                + ", ".join(inert_names)
            )

        st.subheader("Choose Plucked / Arranged")
        header1, header2, header3 = st.columns([0.22, 0.22, 0.56], gap="medium")
        with header1:
            st.markdown("**Plucked**")
        with header2:
            st.markdown("**Arranged**")
        with header3:
            st.markdown("**Species**")

        for i in range(len(species)):
            col1, col2, col3 = st.columns([0.22, 0.22, 0.56], gap="medium")
            with col1:
                st.selectbox(
                    "Plucked",
                    options=["end", "mid"],
                    format_func=lambda x: x.title(),
                    key=f"{prefix}_plucked_{i}",
                    label_visibility="collapsed",
                )
            with col2:
                st.selectbox(
                    "Arranged",
                    options=["end", "mid"],
                    format_func=lambda x: x.title(),
                    key=f"{prefix}_arranged_{i}",
                    label_visibility="collapsed",
                )
            with col3:
                st.markdown(
                    f"<div style='padding-top:0.45rem; padding-bottom:0.35rem;'>{names[i]}</div>",
                    unsafe_allow_html=True,
                )

        # Get ready button for rate constants
        if st.button("Add Reactions", key=f"{prefix}_add_reactions_button"):
            plucked = [
                st.session_state[f"{prefix}_plucked_{i}"] for i in range(len(species))
            ]
            arranged = [
                st.session_state[f"{prefix}_arranged_{i}"] for i in range(len(species))
            ]

            mixture_eqns, init_consts = Interesterifier.interesterification_rxn_list(
                species,
                arranged=arranged,
                plucked=plucked,
            )

            st.session_state[f"{prefix}_plucked"] = plucked
            st.session_state[f"{prefix}_arranged"] = arranged
            st.session_state[f"{prefix}_mixture_eqns"] = mixture_eqns
            st.session_state[f"{prefix}_init_consts"] = init_consts
            st.session_state[f"{prefix}_reactions_initialized"] = True
            st.session_state[f"{prefix}_sim_initialized"] = False
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Reaction list initialized.")

    if st.session_state[f"{prefix}_reactions_initialized"]:
        mixture_eqns = st.session_state[f"{prefix}_mixture_eqns"]
        init_consts = st.session_state[f"{prefix}_init_consts"]

        st.subheader("Reaction Equations and Rate Constants")
        select_all_button = st.number_input(
            "Scale all initial rate constants by:",
            value=1.0,
            step=0.1,
            key=f"{prefix}_select_all_plucked",
        )

        header1, header2 = st.columns([0.22, 0.78], gap="medium")
        with header1:
            st.markdown("**Rate Constant**")
        with header2:
            st.markdown("**Reaction Equation**")

        for i in range(len(mixture_eqns)):
            col1, col2 = st.columns([0.22, 0.78], gap="medium")
            with col1:
                st.number_input(
                    "Rate Constant",
                    value=float(init_consts[i]) * float(select_all_button),
                    key=f"{prefix}_k_{i}",
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown(
                    f"<div style='padding-top:0.45rem; padding-bottom:0.35rem; white-space:normal;'>{mixture_eqns[i]}</div>",
                    unsafe_allow_html=True,
                )

        if st.button(
            "Initialize Simulation", key=f"{prefix}_initialize_simulation_button"
        ):
            ks = [st.session_state[f"{prefix}_k_{i}"] for i in range(len(mixture_eqns))]
            st.session_state[f"{prefix}_ks"] = ks
            st.session_state[f"{prefix}_sim_initialized"] = True
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Simulation parameters initialized.")

    if st.session_state[f"{prefix}_sim_initialized"]:
        st.subheader("Simulation Parameters")

        beginning = st.number_input(
            "Simulation Start Time",
            value=0.0,
            key=f"{prefix}_sim_start_time",
        )
        end = st.number_input(
            "Simulation End Time",
            value=10.0,
            key=f"{prefix}_sim_end_time",
        )
        num_points = st.number_input(
            "Number of Evaluation Points",
            min_value=10,
            value=100,
            step=10,
            key=f"{prefix}_sim_num_points",
        )

        if st.button("Run Simulation", key=f"{prefix}_run_simulation_button"):
            species = st.session_state[f"{prefix}_species"]
            init_concentrations = st.session_state[f"{prefix}_init_concentrations"]
            ks = st.session_state[f"{prefix}_ks"]
            arranged = st.session_state[f"{prefix}_arranged"]
            plucked = st.session_state[f"{prefix}_plucked"]
            inert_rows = st.session_state.get(f"{prefix}_inert_rows", [])

            sim: PKineticSim = Interesterifier.interesterification(
                species,
                initial_conc=init_concentrations,
                ks=ks,
                arranged=arranged,
                plucked=plucked,
                units=st.session_state.get("display_unit", "Moles"),
            )

            t_eval = np.linspace(beginning, end, num=int(num_points))
            sol = sim.solve(t_span=(beginning, end), t_eval=t_eval)
            st.session_state[f"{prefix}_sim_object"] = sim
            st.session_state[f"{prefix}_solution"] = sol
            st.session_state[f"{prefix}_results_df"] = build_results_dataframe(sim)
            st.session_state[f"{prefix}_time_points"] = t_eval
            st.session_state[f"{prefix}_sim_ran"] = True
            try:
                reactive_result_rows = [
                    MixRow(comp, Moles) for comp, Moles in sim.glyceride_mix.mix.items()
                ]
                st.session_state[f"{prefix}_result_mix_object"] = sim.glyceride_mix
                st.session_state[f"{prefix}_result_mix_rows"] = merge_mix_rows(
                    reactive_result_rows,
                    inert_rows,
                )
            except Exception as e:
                st.warning(f"Could not construct resulting GlycerideMix: {str(e)}")
            st.success("Simulation completed.")

    if st.session_state[f"{prefix}_sim_ran"]:
        show_simulation_results(prefix)


def render_esterification_tab():
    prefix = "ester"
    rows = get_processor_rows()

    if not rows:
        st.info("Please build or import a mixture on the Glyceride Mix page first.")
        return

    st.write(f"Loaded {len(rows)} species from the current mixture.")
    st.subheader("Current Glyceride Mix")
    # Display rows in dataframe in the middle of the screen
    df = rows_to_display_df(rows, st.session_state.get("display_unit", "Moles"))
    st.dataframe(df, use_container_width=True)

    fatty_acid_rows, detected_glycerol = split_esterification_rows(rows)

    if st.button("Initialize with Mixture", key=f"{prefix}_initialize_mixture_button"):
        if not fatty_acid_rows:
            st.error("Esterification requires fatty acids in the current mixture.")
        else:
            st.session_state[f"{prefix}_species"] = [r.comp for r in fatty_acid_rows]
            st.session_state[f"{prefix}_init_concentrations"] = [
                float(r.moles) for r in fatty_acid_rows
            ]
            st.session_state[f"{prefix}_species_names"] = [
                comp_display_name(r.comp) for r in fatty_acid_rows
            ]
            st.session_state[f"{prefix}_glycerol_init"] = float(detected_glycerol)
            st.session_state[f"{prefix}_initialized"] = True
            st.session_state[f"{prefix}_reactions_initialized"] = False
            st.session_state[f"{prefix}_sim_initialized"] = False
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Mixture initialized for the esterifier.")

    if st.session_state[f"{prefix}_initialized"]:
        species = st.session_state[f"{prefix}_species"]
        init_concentrations = st.session_state[f"{prefix}_init_concentrations"]
        names = st.session_state[f"{prefix}_species_names"]

        st.subheader("Feed Composition")
        st.caption(
            "The esterifier uses glycerol plus the fatty acids present in the current mixture."
        )

        glycerol_initial = st.number_input(
            "Initial Glycerol Amount",
            min_value=0.0,
            value=float(st.session_state.get(f"{prefix}_glycerol_init", 0.0)),
            step=0.1,
            key=f"{prefix}_glycerol_input",
        )

        feed_header1, feed_header2 = st.columns([0.30, 0.70], gap="medium")
        with feed_header1:
            st.markdown("**Initial Moles**")
        with feed_header2:
            st.markdown("**Fatty Acid**")

        for i in range(len(species)):
            col1, col2 = st.columns([0.30, 0.70], gap="medium")
            with col1:
                st.number_input(
                    "Initial Moles",
                    min_value=0.0,
                    value=float(init_concentrations[i]),
                    step=0.1,
                    key=f"{prefix}_fa_init_{i}",
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown(
                    f"<div style='padding-top:0.45rem; padding-bottom:0.35rem;'>{names[i]}</div>",
                    unsafe_allow_html=True,
                )

        # Get ready button for rate constants
        if st.button("Add Reactions", key=f"{prefix}_add_reactions_button"):
            st.session_state[f"{prefix}_glycerol_init"] = float(glycerol_initial)
            st.session_state[f"{prefix}_init_concentrations"] = [
                st.session_state[f"{prefix}_fa_init_{i}"] for i in range(len(species))
            ]

            mixture_eqns = Esterifier.esterification_rxn_list(species)
            init_consts = [1.0] * len(mixture_eqns)

            st.session_state[f"{prefix}_mixture_eqns"] = mixture_eqns
            st.session_state[f"{prefix}_init_consts"] = init_consts
            st.session_state[f"{prefix}_reactions_initialized"] = True
            st.session_state[f"{prefix}_sim_initialized"] = False
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Reaction list initialized.")

    if st.session_state[f"{prefix}_reactions_initialized"]:
        mixture_eqns = st.session_state[f"{prefix}_mixture_eqns"]
        init_consts = st.session_state[f"{prefix}_init_consts"]

        st.subheader("Reaction Equations and Rate Constants")
        select_all_button = st.number_input(
            "Scale all initial rate constants by:",
            value=1.0,
            step=0.1,
            key=f"{prefix}_select_all_plucked",
        )

        header1, header2 = st.columns([0.22, 0.78], gap="medium")
        with header1:
            st.markdown("**Rate Constant**")
        with header2:
            st.markdown("**Reaction Equation**")

        for i in range(len(mixture_eqns)):
            col1, col2 = st.columns([0.22, 0.78], gap="medium")
            with col1:
                st.number_input(
                    "Rate Constant",
                    value=float(init_consts[i]) * float(select_all_button),
                    key=f"{prefix}_k_{i}",
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown(
                    f"<div style='padding-top:0.45rem; padding-bottom:0.35rem; white-space:normal;'>{mixture_eqns[i]}</div>",
                    unsafe_allow_html=True,
                )

        if st.button(
            "Initialize Simulation", key=f"{prefix}_initialize_simulation_button"
        ):
            ks = [st.session_state[f"{prefix}_k_{i}"] for i in range(len(mixture_eqns))]
            st.session_state[f"{prefix}_ks"] = ks
            st.session_state[f"{prefix}_sim_initialized"] = True
            st.session_state[f"{prefix}_sim_ran"] = False
            st.success("Simulation parameters initialized.")

    if st.session_state[f"{prefix}_sim_initialized"]:
        st.subheader("Simulation Parameters")

        beginning = st.number_input(
            "Simulation Start Time",
            value=0.0,
            key=f"{prefix}_sim_start_time",
        )
        end = st.number_input(
            "Simulation End Time",
            value=10.0,
            key=f"{prefix}_sim_end_time",
        )
        num_points = st.number_input(
            "Number of Evaluation Points",
            min_value=10,
            value=100,
            step=10,
            key=f"{prefix}_sim_num_points",
        )

        if st.button("Run Simulation", key=f"{prefix}_run_simulation_button"):
            species = st.session_state[f"{prefix}_species"]
            fatty_acid_initial_conc = st.session_state[f"{prefix}_init_concentrations"]
            glycerol_initial = float(st.session_state[f"{prefix}_glycerol_init"])
            ks = list(st.session_state[f"{prefix}_ks"])

            sim: PKineticSim = Esterifier.esterification(
                list_of_fa=species,
                initial_conc=[glycerol_initial, *fatty_acid_initial_conc],
                ks=ks,
                units=st.session_state.get("display_unit", "Moles"),
            )

            t_eval = np.linspace(beginning, end, num=int(num_points))
            sol = sim.solve(t_span=(beginning, end), t_eval=t_eval)
            st.session_state[f"{prefix}_sim_object"] = sim
            st.session_state[f"{prefix}_solution"] = sol
            st.session_state[f"{prefix}_results_df"] = build_results_dataframe(sim)
            st.session_state[f"{prefix}_time_points"] = t_eval
            st.session_state[f"{prefix}_sim_ran"] = True
            try:
                st.session_state[f"{prefix}_result_mix_object"] = sim.glyceride_mix
                st.session_state[f"{prefix}_result_mix_rows"] = [
                    MixRow(comp, Moles) for comp, Moles in sim.glyceride_mix.mix.items()
                ]
            except Exception as e:
                st.warning(f"Could not construct resulting GlycerideMix: {str(e)}")
            st.success("Simulation completed.")

    if st.session_state[f"{prefix}_sim_ran"]:
        show_simulation_results(prefix)


# Initialize state once per tab
if "inter_initialized" not in st.session_state:
    initialize_reactor_state("inter")
if "ester_initialized" not in st.session_state:
    initialize_reactor_state("ester")


# Card wrapper
st.markdown('<div class="glyze-card">', unsafe_allow_html=True)
st.markdown('<div class="start-title">Batch Reactor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="start-subtitle">Initialize with a mixture, build reactions, then run the simulation.</div>',
    unsafe_allow_html=True,
)

# Put esterification first, then interesterification
ester_tab, inter_tab = st.tabs(["Esterification", "Interesterification"])

with ester_tab:
    render_esterification_tab()

with inter_tab:
    render_interesterification_tab()

# close card
st.markdown("</div>", unsafe_allow_html=True)
