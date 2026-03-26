import streamlit as st
import pandas as pd


from glyze.chem_processor.viscosity_calculator import ViscosityCalculator
from glyze.glyceride_mix import GlycerideMix



st.set_page_config(page_title="GLYZE — Viscosity Prediction", page_icon="🧪", layout="wide")

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


def get_mix_rows():
    return st.session_state.setdefault("mix_rows", [])



def get_processor_rows():
    if "processor_mix_rows" in st.session_state:
        return st.session_state["processor_mix_rows"]
    return get_mix_rows()



def unwrap_component(comp):
    """
    Return the underlying chemical object/string when rows store MixtureComponent-like wrappers.
    """
    return getattr(comp, "component", comp)



def comp_display_name(comp) -> str:
    comp = unwrap_component(comp)
    if comp in {"H2O", "Glycerol"}:
        return str(comp)
    if hasattr(comp, "name"):
        return str(comp.name)
    return str(comp)



def build_mix_preview_dataframe(rows):
    return pd.DataFrame(
        {
            "Species": [comp_display_name(r.comp) for r in rows],
            "Moles": [float(r.moles) for r in rows],
        }
    )



def build_viscosity_mix(rows):
    """
    Convert MixRow entries into a GlycerideMix compatible with ViscosityCalculator.
    """
    if not rows:
        raise ValueError("No mixture rows were found. Please initialize a mixture first.")

    mix_dict = {}
    for row in rows:
        comp = unwrap_component(row.comp)
        qty = float(row.moles)
        if qty < 0:
            raise ValueError(f"Negative moles are not allowed for {comp_display_name(comp)}.")
        mix_dict[comp] = mix_dict.get(comp, 0.0) + qty

    return GlycerideMix(mix_dict)



def build_summary_dataframe(result):
    return pd.DataFrame(
        {
            "Component": result["tags"],
            "Mass Fraction": result["mass_fractions"],
            "Mole Fraction": result["mole_fractions"],
        }
    )



def build_curve_dataframe(result):
    df = pd.DataFrame({"Temperature (C)": result["temperature_C"]})
    for i, tag in enumerate(result["tags"]):
        df[f"{tag} Viscosity (cP)"] = result["pure_viscosities_cP"][i]
    df["Mixture Viscosity (cP)"] = result["mixture_viscosity_cP"]
    return df



def initialize_viscosity_state():
    st.session_state["visc_calculated"] = False
    st.session_state["visc_mix_preview_df"] = None
    st.session_state["visc_summary_df"] = None
    st.session_state["visc_curve_df"] = None
    st.session_state["visc_plot"] = None
    st.session_state["visc_csv_bytes"] = None
    st.session_state["visc_result"] = None


if "visc_calculated" not in st.session_state:
    initialize_viscosity_state()


st.markdown('<div class="glyze-card">', unsafe_allow_html=True)
st.markdown('<div class="start-title">Viscosity Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="start-subtitle">Use the current mixture to estimate component and mixture viscosity as a function of temperature.</div>',
    unsafe_allow_html=True,
)

rows = get_processor_rows()

with st.expander("Current mixture", expanded=True):
    if rows:
        st.dataframe(build_mix_preview_dataframe(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No mixture is currently loaded. Create or upload a mixture first on the mixture page.")

col1, col2, col3 = st.columns(3)
with col1:
    init_temp = st.number_input("Initial temperature (C)", value=40.0, step=1.0)
with col2:
    final_temp = st.number_input("Final temperature (C)", value=80.0, step=1.0)
with col3:
    step_size = st.number_input("Step size (C)", min_value=0.1, value=1.0, step=0.5)

calc_col, clear_col = st.columns([3, 1])
with calc_col:
    run_prediction = st.button("Calculate viscosity")
with clear_col:
    clear_results = st.button("Clear")

if clear_results:
    initialize_viscosity_state()
    st.rerun()

if run_prediction:
    try:
        mix = build_viscosity_mix(rows)
        import inspect
        result, fig = ViscosityCalculator.calculate_and_plot(
            mix=mix,
            init_temp=float(init_temp),
            final_temp=float(final_temp),
            step_size=float(step_size),
            title="Viscosity vs Temperature",
        )

        mix_preview_df = build_mix_preview_dataframe(rows)
        summary_df = build_summary_dataframe(result)
        curve_df = build_curve_dataframe(result)
        csv_bytes = ViscosityCalculator.to_csv_string(result).encode("utf-8")

        st.session_state["visc_calculated"] = True
        st.session_state["visc_mix_preview_df"] = mix_preview_df
        st.session_state["visc_summary_df"] = summary_df
        st.session_state["visc_curve_df"] = curve_df
        st.session_state["visc_plot"] = fig
        st.session_state["visc_csv_bytes"] = csv_bytes
        st.session_state["visc_result"] = result

        st.success("Viscosity calculation completed.")

    except Exception as exc:
        st.session_state["visc_calculated"] = False
        st.error(str(exc))

if st.session_state.get("visc_calculated", False):
    st.markdown("### Results")

    with st.expander("Parsed mixture summary", expanded=True):
        st.dataframe(
            st.session_state["visc_summary_df"],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Viscosity plot", expanded=True):
        st.plotly_chart(st.session_state["visc_plot"], use_container_width=True)

    with st.expander("Viscosity table", expanded=True):
        st.dataframe(
            st.session_state["visc_curve_df"],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        label="Download CSV",
        data=st.session_state["visc_csv_bytes"],
        file_name="viscosity_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
