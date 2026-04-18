import streamlit as st
import pandas as pd
import numpy as np

from glyze import Deodorizer as DeodorizerEngine
from glyze.glyceride import Glyceride, FattyAcid
from glyze.glyceride_mix import GlycerideMix

from glyze.ui.pages.Glyceride_Mix import (
    PAGE_CSS,
    MixRow,
    comp_display_name,
    get_mix_rows,
    rows_to_display_df,
)


st.set_page_config(page_title="GLYZE — Deodorizer", page_icon="🧈", layout="wide")

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

    /* ---- Hero ---- */
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

    /* ---- Divider ---- */
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

    /* ---- Card ---- */
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

    /* ---- Buttons ---- */
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

    /* Subheader spacing inside card */
    h3{
        margin-top: 0.2rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* Text / number inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06) !important;
    }

    /* Selectbox + multiselect (BaseWeb) */
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

    /* Focus highlight */
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

    /* ---------- Expanders ---------- */
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
    div[data-testid="stExpander"] summary{
        filter: none !important;
        -webkit-text-fill-color: var(--text) !important;
    }

    /* ---------- Number input buttons ---------- */
    div[data-testid="stNumberInput"] button{
        background: var(--field-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--field-border) !important;
        box-shadow: none !important;
    }
    div[data-testid="stNumberInput"] button:hover{
        border-color: rgba(37,99,235,0.35) !important;
    }

    /* ---------- Dropdown menus ---------- */
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

    /* ---------- Data editor / table ---------- */
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

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"]{
        border-radius: 16px !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
        box-shadow: 0 14px 28px rgba(15,23,42,0.07) !important;
    }
    div[data-testid="stAlert"] *{
        color: var(--text) !important;
    }

    /* ---------- Sliders ---------- */
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
        Deodorize a glyceride mixture! 
    </div>
</div>
<div class="glyze-divider"></div>
"""


st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)


def get_processor_rows():
    """Retrieve the current mixture rows from session state."""
    if "processor_mix_rows" in st.session_state:
        return st.session_state["processor_mix_rows"]
    return get_mix_rows()


def unwrap_component(comp):
    """Return the underlying chemical object if wrapped in MixtureComponent."""
    if hasattr(comp, "component"):
        return comp.component
    return comp


def classify_component(comp) -> str:
    """Return a human-readable type label for a mixture component."""
    base = unwrap_component(comp)
    if isinstance(base, Glyceride):
        n_fa = sum(1 for x in base.sn if x is not None)
        return {1: "MAG", 2: "DAG", 3: "TAG"}.get(n_fa, "Glyceride")
    if isinstance(base, FattyAcid):
        return "FFA"
    return "Other"


def build_mix_object(rows):
    """Build a GlycerideMix from the current MixRow list."""
    mix_dict = {r.comp: r.moles for r in rows}
    return GlycerideMix(mix_dict, units="Moles", sort=True)


def build_feed_dataframe(rows):
    """Build a preview DataFrame of the current feed mixture."""
    return pd.DataFrame(
        {
            "Species": [comp_display_name(r.comp) for r in rows],
            "Type": [classify_component(r.comp) for r in rows],
            "Moles": [r.moles for r in rows],
        }
    )


def build_results_dataframe(initial, result_mix, T_K, P_Pa):
    """
    Build a comparison DataFrame showing initial vs. final quantities,
    removal amounts, removal percentages, vapor pressures, and A = P/VP.
    """
    rows = []
    for comp, qi in initial.items():
        qf = result_mix.mix.get(comp, 0.0)
        removed = qi - qf
        pct = (removed / qi * 100) if qi > 0 else 0.0
        vp = comp.vapor_pressure(T_K)
        A = P_Pa / vp if vp > 0 else float("inf")
        rows.append(
            {
                "Species": comp.name,
                "Type": classify_component(comp),
                "Initial": qi,
                "Final": qf,
                "Removed": removed,
                "% Removed": pct,
                "VP (Pa)": vp,
                "A = P/VP": A,
            }
        )
    return pd.DataFrame(rows)


def upload_resulting_mixture(result_rows, result_mix_obj):
    """Push the deodorizer result back to the Glyceride Mix page."""
    st.session_state["glyze_mix_object"] = result_mix_obj
    st.session_state["glyze_mix_rows"] = result_rows
    st.session_state["mix_rows"] = result_rows
    st.session_state["glyze_mix_initialized"] = True
    st.session_state["processor_mix_rows"] = result_rows
    st.success("Resulting mixture sent to the Glyceride Mix page.")


def initialize_deodorizer_state():
    """Set default session state keys for the deodorizer page."""
    st.session_state.setdefault("deod_initialized", False)
    st.session_state.setdefault("deod_ran", False)
    st.session_state.setdefault("deod_T_C", 250.0)
    st.session_state.setdefault("deod_P_mbar", 3.0)
    st.session_state.setdefault("deod_S", 1.0)
    st.session_state.setdefault(
        "deod_entrainment", 0.05
    )  # 5% yield loss due to mechanical carryover, indepdendent of vapor pressure
    st.session_state.setdefault("deod_result_mix", None)
    st.session_state.setdefault("deod_result_df", None)
    st.session_state.setdefault("deod_initial_snapshot", None)


initialize_deodorizer_state()
rows = get_processor_rows()

if not rows:
    st.info("Please build or import a mixture on the Glyceride Mix page first.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

st.write(f"Loaded {len(rows)} species from the current mixture.")
# Display rows in dataframe in the middle of the screen
df = rows_to_display_df(rows, st.session_state.get("display_unit", "Moles"))
st.dataframe(df, use_container_width=True)


# Display various input parameters for the deodorizer
st.subheader("Operating Conditions")

left_col, right_col = st.columns(2, gap="medium")

# Temperature and pressure input on the left
with left_col:
    T_C = st.number_input(
        "Temperature (°C)",
        min_value=100.0,
        max_value=600.0,
        value=float(st.session_state["deod_T_C"]),
        step=5.0,
        key="deod_T_C_input",
    )

    P_mbar = st.number_input(
        "System pressure (mbar)",
        min_value=0.1,
        max_value=100.0,
        value=float(st.session_state["deod_P_mbar"]),
        step=0.5,
        key="deod_P_mbar_input",
    )

# Mechanical entrainment and steam factor on the right
with right_col:
    entrainment = st.number_input(
        "Entrainment fraction (0-0.20)",
        min_value=0.0,
        max_value=0.20,
        value=float(st.session_state["deod_entrainment"]),
        step=0.005,
        format="%.4f",
        key="deod_entrainment_input",
        help="Fraction of each component lost to mechanical carryover, independent of vapor pressure.",
    )

    S_fixed = st.number_input(
        "Steam factor S (mol steam / mol oil)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["deod_S"]),
        step=0.1,
        key="deod_S_input",
    )


#
if st.button("Run Deodorizer", use_container_width=True, key="deod_run_button"):
    # Convert units
    T_K = T_C + 273.15
    P_Pa = P_mbar * 100.0  # mbar -> Pa

    # Build the GlycerideMix object from current rows
    try:
        mix_obj = build_mix_object(rows)
    except Exception as e:
        st.error(f"Could not build mixture: {e}")
        st.stop()

    # Snapshot initial state before running for plotting and comparison
    initial_snapshot = {comp: qty for comp, qty in mix_obj.mix.items()}

    # Single-pass stripping at fixed S
    result_mix = DeodorizerEngine.deodorizer(
        mix=mix_obj, S=S_fixed, T=T_K, P=P_Pa, entrainment=entrainment
    )
    S_used = S_fixed

    # Build results DataFrame
    results_df = build_results_dataframe(initial_snapshot, result_mix, T_K, P_Pa)

    # Persist to session state
    st.session_state["deod_T_C"] = T_C
    st.session_state["deod_P_mbar"] = P_mbar
    st.session_state["deod_S"] = S_used
    st.session_state["deod_entrainment"] = entrainment
    st.session_state["deod_ran"] = True
    st.session_state["deod_result_mix"] = result_mix
    st.session_state["deod_result_df"] = results_df
    st.session_state["deod_initial_snapshot"] = initial_snapshot
    st.session_state["deod_S_used"] = S_used

    st.success("Deodorization completed.")


# Diplay results if available
if st.session_state["deod_ran"]:
    results_df = st.session_state["deod_result_df"]
    result_mix = st.session_state["deod_result_mix"]
    initial_snapshot = st.session_state["deod_initial_snapshot"]
    S_used = st.session_state.get("deod_S_used", 0.0)

    st.subheader("Results")

    # Summary metrics
    initial_total = sum(initial_snapshot.values())
    final_total = sum(result_mix.quantities)
    total_removed = initial_total - final_total

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steam factor S", f"{S_used:.4f}")
    m2.metric("Initial moles", f"{initial_total:.4f}")
    m3.metric("Final moles", f"{final_total:.4f}")
    m4.metric("Total removed", f"{total_removed:.4f}")

    # Results view mode
    results_mode = st.radio(
        "Display results as:",
        ["Comparison Table", "Bar Chart"],
        horizontal=True,
        key="deod_results_mode",
    )

    if results_mode == "Comparison Table":
        st.dataframe(
            results_df.style.format(
                {
                    "Initial": "{:.6f}",
                    "Final": "{:.6f}",
                    "Removed": "{:.6f}",
                    "% Removed": "{:.2f}",
                    "VP (Pa)": "{:.4e}",
                    "A = P/VP": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    else:
        import plotly.graph_objects as go

        # Filter to glycerides and FFAs only. In other words, only return
        # the columns that are TAGS, DAGS, MAGS, or FFA
        chart_df = results_df[results_df["Type"].isin(["TAG", "DAG", "MAG", "FFA"])]

        # Sort by fatty acids, then glycerides sorted by carbon number,
        # then alphabetically, all in descending order of initial moles
        def type_key(x):
            # Prioritize FFA first, then MAG, DAG, TAG
            if x[1].startswith("G_"):
                num_emptys = sum(
                    [empty for empty in x[1].split("_") if empty == "EMPTY"]
                )
                if num_emptys == 2:
                    return 1
                elif num_emptys == 1:
                    return 2
                else:
                    return 3
            else:
                return 0

        def species_key(x):
            import re

            matches = re.findall("[0-9]+", x[1])
            return sum([int(s) for s in matches if s.isdigit()]) if matches else 0

        from natsort import natsort_keygen
        import random

        chart_df = chart_df.sort_values(
            by=["Species"],
            # key=natsort_keygen()
            key=lambda x: x.str.count("EMPTY") + x.str.count("_"),
        )

        fig = go.Figure(
            [
                go.Bar(name="Initial", x=chart_df["Species"], y=chart_df["Initial"]),
                go.Bar(name="Final", x=chart_df["Species"], y=chart_df["Final"]),
            ]
        )
        fig.update_layout(
            barmode="group",
            title=f"Deodorizer: Initial vs Final (S={S_used:.4f})",
            xaxis_title="Species",
            yaxis_title="Moles",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Export")

    # Send result back to the Glyceride Mix page
    if st.button(
        "Upload Resulting Mixture",
        use_container_width=True,
        key="deod_send_mix_to_gmix",
    ):
        try:
            result_rows = [MixRow(comp, qty) for comp, qty in result_mix.mix.items()]
            upload_resulting_mixture(result_rows, result_mix)
        except Exception as e:
            st.error(str(e))

    # Download results CSV
    csv_name = st.text_input(
        "Enter filename for CSV (without extension)",
        value="deodorizer_results",
        key="deod_csv_filename",
    )
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_bytes,
        file_name=csv_name + ".csv" if not csv_name.endswith(".csv") else csv_name,
        mime="text/csv",
        use_container_width=True,
        key="deod_csv_download",
    )


# close card
st.markdown("</div>", unsafe_allow_html=True)
