import io
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from glyze.glyceride import Glyceride, FattyAcid
from glyze.glyceride_mix import GlycerideMix


st.set_page_config(page_title="GLYZE — Glyceride Mix", page_icon="🧈", layout="wide")

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
        max-width: 1320px;   /* was 920px on home */
        padding-top: 1.2rem;
        padding-bottom: 2.8rem;
    }

    .glyze-card{
        padding: 1.4rem 1.6rem; /* slightly roomier */
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

    /* Make the dropdown text + icon dark (some themes force gray) */
    div[data-testid="stSelectbox"] * ,
    div[data-testid="stMultiSelect"] * {
        color: var(--text) !important;
    }

    /* Focus highlight: same accent as home.py */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
        border-color: rgba(37,99,235,0.55) !important;   /* accentA */
        box-shadow:
            0 10px 20px rgba(15,23,42,0.06),
            0 0 0 4px rgba(37,99,235,0.18) !important;   /* soft glow */
    }

    /* Optional: remove weird dark “pill” around select value on some builds */
    div[data-baseweb="select"] span {
        background: transparent !important;
    }
    :root{
    --field-bg: rgba(255,255,255,0.88);
    --field-border: rgba(15,23,42,0.12);
    --field-shadow: 0 10px 20px rgba(15,23,42,0.06);
    --focus-ring: 0 0 0 4px rgba(37,99,235,0.18);
    }

    /* ---------- Expanders (black header bar) ---------- */
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

    /* Expand/collapse caret */
    div[data-testid="stExpander"] summary svg{
    fill: var(--text) !important;
    color: var(--text) !important;
    }

    /* Expander content area */
    div[data-testid="stExpander"] div[role="region"]{
    background: transparent !important;
    }


    /* ---------- Inputs + Selects (BaseWeb) ---------- */
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

    /* Dropdown menu panel (often black) */
    ul[role="listbox"],
    div[data-baseweb="popover"]{
    background: rgba(255,255,255,0.96) !important;
    color: var(--text) !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 40px rgba(15,23,42,0.10) !important;
    }

    /* Dropdown options */
    li[role="option"]{
    color: var(--text) !important;
    }
    li[role="option"][aria-selected="true"]{
    background: rgba(37,99,235,0.10) !important;
    }

    /* Focus highlight */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within{
    border-color: rgba(37,99,235,0.55) !important;
    box-shadow: var(--field-shadow), var(--focus-ring) !important;
    }

    /* Number input +/- buttons (they can appear dark) */
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
    /* ---------- Data editor / table (the black middle list) ---------- */
    /* Outer container */
    div[data-testid="stDataEditor"]{
    border-radius: 16px !important;
    }

    /* The “grid” itself (Streamlit uses a grid component under the hood) */
    div[data-testid="stDataEditor"] [role="grid"],
    div[data-testid="stDataEditor"] .stDataFrame,
    div[data-testid="stDataEditor"] [data-testid="stTable"]{
    background: rgba(255,255,255,0.92) !important;
    color: var(--text) !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 16px !important;
    box-shadow: var(--field-shadow) !important;
    }

    /* Header row */
    div[data-testid="stDataEditor"] [role="columnheader"]{
    background: rgba(248,250,252,0.95) !important;
    color: var(--text) !important;
    border-bottom: 1px solid rgba(15,23,42,0.08) !important;
    }

    /* Cells */
    div[data-testid="stDataEditor"] [role="gridcell"]{
    background: rgba(255,255,255,0.90) !important;
    color: var(--text) !important;
    border-bottom: 1px solid rgba(15,23,42,0.06) !important;
    }

    /* The “empty” / placeholder row text */
    div[data-testid="stDataEditor"] [data-testid="stDataEditorEmpty"]{
    color: rgba(71,85,105,0.85) !important;
    }

    /* Selected cell / row highlight */
    div[data-testid="stDataEditor"] [role="gridcell"][aria-selected="true"],
    div[data-testid="stDataEditor"] [role="row"][aria-selected="true"] [role="gridcell"]{
    background: rgba(37,99,235,0.10) !important;
    outline: none !important;
    }

    /* Editor input when editing a cell */
    div[data-testid="stDataEditor"] input{
    background: rgba(255,255,255,0.98) !important;
    color: var(--text) !important;
    border: 1px solid rgba(37,99,235,0.35) !important;
    box-shadow: var(--focus-ring) !important;
    }


    /* ---------- Alerts (info box was blue, but keep consistent) ---------- */
    div[data-testid="stAlert"]{
    border-radius: 16px !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    box-shadow: 0 14px 28px rgba(15,23,42,0.07) !important;
    }
    div[data-testid="stAlert"] *{
    color: var(--text) !important;
    }


    /* ---------- Sliders / misc dark bars (if any appear) ---------- */
    div[data-testid="stSlider"] [role="slider"],
    div[data-testid="stSlider"] [data-baseweb="slider"]{
    color: var(--text) !important;
    }
    </style>

    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# Card wrapper
st.markdown('<div class="glyze-card">', unsafe_allow_html=True)
st.markdown('<div class="start-title">Glyceride Mix Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="start-subtitle">Add species on the left, edit the mixture in the center, export on the right.</div>',
    unsafe_allow_html=True,
)


MM_H2O = 18.01528
MM_GLYCEROL = 92.09382


def parse_component_from_string(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty component name.")

    if s == "H2O":
        return "H2O"
    if s.lower() in {"glycerol", "gly"}:
        return "Glycerol"

    if s.startswith("G_"):
        return Glyceride.from_name(s)
    if s.startswith("N"):
        return FattyAcid.from_name(s)

    raise ValueError("Unrecognized component. Use N..., G_..., H2O, or Glycerol.")


def format_fa_name(
    carbons: int,
    db_count: int,
    db_positions: List[int],
    db_stereo: List[str],
    methyl_positions: List[int],
    hydroxyl_positions: List[int],
) -> str:
    """
    Format the name of the fatty acid as input from the user
    """
    cc = int(carbons)
    dd = int(db_count)
    name = f"N{cc:02d}D{dd:d}"

    if dd > 0:
        if len(db_positions) != dd or len(db_stereo) != dd:
            raise ValueError("DB count must match number of DB positions and stereochem entries.")
        name += "P" + "".join(f"{int(p):02d}{st.upper()}" for p, st in zip(db_positions, db_stereo))

    if methyl_positions:
        name += "M" + "".join(f"{int(p):02d}" for p in methyl_positions)

    if hydroxyl_positions:
        name += "OH" + "".join(f"{int(p):02d}" for p in hydroxyl_positions)

    return name


def molar_mass_of(comp: Any) -> float:
    """
    Grab the molar mass of a component
    """
    if comp == "H2O":
        return MM_H2O
    if comp == "Glycerol":
        return MM_GLYCEROL
    if hasattr(comp, "molar_mass"):
        return float(comp.molar_mass)
    raise ValueError(f"Cannot determine molar mass for component {comp!r}")


def comp_display_name(comp: Any) -> str:
    if comp in {"H2O", "Glycerol"}:
        return str(comp)
    if hasattr(comp, "name"):
        return str(comp.name)
    return str(comp)

# Dataclass containing a row of the list in the middle of the page
@dataclass
class MixRow:
    comp: Any
    moles: float


def get_mix_rows() -> List[MixRow]:
    return st.session_state.setdefault("mix_rows", [])


def set_mix_rows(rows: List[MixRow]) -> None:
    st.session_state["mix_rows"] = rows


def merge_rows(rows: List[MixRow]) -> List[MixRow]:
    merged: Dict[str, MixRow] = {}
    for r in rows:
        key = comp_display_name(r.comp)
        if key in merged:
            merged[key].moles += float(r.moles)
        else:
            merged[key] = MixRow(r.comp, float(r.moles))
    return [merged[k] for k in sorted(merged.keys())]



def add_to_mix(comp: Any, qty: float, qty_unit: str) -> None:
    """
    Adds a component to the mixture
    """
    qty = float(qty)
    if qty <= 0:
        raise ValueError("Quantity must be > 0")

    if qty_unit == "mole":
        moles = qty
    elif qty_unit == "gram":
        moles = qty / molar_mass_of(comp)
    else:
        raise ValueError(f"Unsupported add unit: {qty_unit}")

    rows = get_mix_rows()
    rows.append(MixRow(comp, moles))
    set_mix_rows(merge_rows(rows))


def rows_to_display_df(rows: List[MixRow], display_unit: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"Species": [], "Quantity": []})

    species = [comp_display_name(r.comp) for r in rows]

    if display_unit == "mole":
        qty = [r.moles for r in rows]
    elif display_unit == "gram":
        qty = [r.moles * molar_mass_of(r.comp) for r in rows]
    elif display_unit == "mass_fraction":
        masses = [r.moles * molar_mass_of(r.comp) for r in rows]
        total = sum(masses) or 1.0
        qty = [m / total for m in masses]
    else:
        raise ValueError(f"Unsupported display unit: {display_unit}")

    return pd.DataFrame({"Species": species, "Quantity": qty})


def apply_table_edits_to_rows(edited_df: pd.DataFrame, rows: List[MixRow], display_unit: str) -> List[MixRow]:
    name_to_comp = {comp_display_name(r.comp): r.comp for r in rows}

    new_rows: List[MixRow] = []
    for _, rec in edited_df.iterrows():
        name = str(rec["Species"])
        q = float(rec["Quantity"])
        if q < 0:
            raise ValueError("Quantities cannot be negative.")
        comp = name_to_comp.get(name)
        if comp is None:
            continue

        # Choose to display which units
        if display_unit == "mole":
            moles = q
        elif display_unit == "gram":
            moles = q / molar_mass_of(comp) if q > 0 else 0.0
        elif display_unit == "mass_fraction":
            # Interpret as fractions on a 1 g basis; renormalize after
            moles = (q * 1.0) / molar_mass_of(comp) if q > 0 else 0.0
        else:
            raise ValueError(f"Unsupported unit: {display_unit}")

        new_rows.append(MixRow(comp, moles))

    if display_unit == "mass_fraction" and new_rows:
        fracs = edited_df["Quantity"].astype(float).tolist()
        s = sum(fracs)
        if s > 0:
            fracs = [f / s for f in fracs]
            new_rows = [
                MixRow(r.comp, (fracs[i] * 1.0) / molar_mass_of(r.comp)) for i, r in enumerate(new_rows)
            ]

    return merge_rows(new_rows)

# Turn the data frame to a csv (TODO: Ensure that this dataframe is accessible
#to other pages and can be turned into a GlycerideMix)
def export_csv_bytes(rows: List[MixRow], export_unit: str) -> bytes:
    df = rows_to_display_df(rows, export_unit)
    df.insert(2, "Units", export_unit)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")



display_unit = st.session_state.setdefault("display_unit", "mole")
export_unit = st.session_state.setdefault("export_unit", "mole")

left, mid, right = st.columns([1.2, 1.9, 1.2], gap="large")

# On the left of the screen, add a species using the code from the first iteration of the ai
with left:
    st.subheader("Add species")

    # Quick add by name 
    with st.expander("Quick add by name", expanded=True):
        # How to use instructions
        st.write("Examples: `N18D1P09Z`, `G_N18D1P09Z_N16D0_N16D0`, `H2O`, `Glycerol`")
        
        # Text input for name
        name_str = st.text_input("Component name", key="quick_name")

        # qty input
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1, key="quick_qty")

        # Units
        qty_unit = st.selectbox("Quantity units", ["mole", "gram"], key="quick_qty_unit")

        # Add button
        if st.button("Add", use_container_width=True, key="quick_add_btn"):
            try:
                comp = parse_component_from_string(name_str)
                add_to_mix(comp, qty, qty_unit)
                st.success(f"Added {name_str} ({qty:g} {qty_unit}).")
            except Exception as e:
                st.error(str(e))

    # Build fatty acids from individual inputs
    with st.expander("Build fatty acid by fields"):
        # Carbons and db_count
        carbons = st.number_input("Carbons (CC)", min_value=2, max_value=40, value=18, step=1, key="fa_cc")
        db_count = st.number_input("Double bonds (DD)", min_value=0, max_value=8, value=1, step=1, key="fa_dd")

        #Double bond positions input
        st.caption("Double-bond positions + stereo (Z/E) — one per double bond.")
        db_pos_str = st.text_input("DB positions (comma-separated)", value="9" if db_count else "", key="fa_dbpos")
        db_st_str = st.text_input("DB stereo (comma-separated, Z/E)", value="Z" if db_count else "", key="fa_dbst")

        # Optional branches input
        st.caption("Optional branches")
        methyl_str = st.text_input("Methyl positions (comma-separated)", value="", key="fa_methyl")
        hydroxyl_str = st.text_input("Hydroxyl positions (comma-separated)", value="", key="fa_oh")

        fa_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1, key="fa_qty")
        fa_qty_unit = st.selectbox("Quantity units", ["mole", "gram"], key="fa_qty_unit")

        # Parse the integers (then will be input as strings)
        def parse_int_list(s: str) -> List[int]:
            s = (s or "").strip()
            if not s:
                return []
            return [int(x.strip()) for x in s.split(",") if x.strip()]

        # Parse the lists (they are also input as strings)
        def parse_stereo_list(s: str) -> List[str]:
            s = (s or "").strip()
            if not s:
                return []
            out = [x.strip().upper() for x in s.split(",") if x.strip()]
            for x in out:
                if x not in {"Z", "E"}:
                    raise ValueError("Stereo must be Z or E.")
            return out

        # Build and add the fatty acid to an internal list
        if st.button("Build + Add Fatty Acid", use_container_width=True, key="fa_add_btn"):
            try:
                db_positions = parse_int_list(db_pos_str)
                db_stereo = parse_stereo_list(db_st_str)
                methyl_positions = parse_int_list(methyl_str)
                hydroxyl_positions = parse_int_list(hydroxyl_str)

                # Format the fa name
                fa_name = format_fa_name(
                    carbons=carbons,
                    db_count=db_count,
                    db_positions=db_positions,
                    db_stereo=db_stereo,
                    methyl_positions=methyl_positions,
                    hydroxyl_positions=hydroxyl_positions,
                )
                # Grab the fa from name and add to mix
                fa_obj = FattyAcid.from_name(fa_name)
                add_to_mix(fa_obj, fa_qty, fa_qty_unit)
                st.success(f"Added {fa_name} ({fa_qty:g} {fa_qty_unit}).")
            except Exception as e:
                st.error(str(e))

    # Build glycerides using fatty acids
    with st.expander("Build glyceride from 3 fatty-acid names"):
        st.write("Enter three FA names, then create `G_fa1_fa2_fa3`.")
        fa1 = st.text_input("FA 1", value="N18D1P09Z", key="g_fa1")
        fa2 = st.text_input("FA 2", value="N16D00", key="g_fa2")
        fa3 = st.text_input("FA 3", value="N16D00", key="g_fa3")

        g_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1, key="g_qty")
        g_qty_unit = st.selectbox("Quantity units", ["mole", "gram"], key="g_qty_unit")

        # Add the objects to the mixture 
        if st.button("🧈 Build + Add Glyceride", use_container_width=True, key="g_add_btn"):
            try:
                _ = FattyAcid.from_name(fa1.strip())
                _ = FattyAcid.from_name(fa2.strip())
                _ = FattyAcid.from_name(fa3.strip())

                g_name = f"G_{fa1.strip()}_{fa2.strip()}_{fa3.strip()}"
                g_obj = Glyceride.from_name(g_name)
                add_to_mix(g_obj, g_qty, g_qty_unit)
                st.success(f"Added {g_name} ({g_qty:g} {g_qty_unit}).")
            except Exception as e:
                st.error(str(e))


with mid:
    # Controls the list in the middle of the screen
    st.subheader("Mixture (edit here)")

    rows = get_mix_rows()

    # Establish both columns
    c1, c2 = st.columns([1, 1])
    with c1:
        # Display units in select box
        new_display_unit = st.selectbox(
            "Display/edit units",
            ["mole", "gram", "mass_fraction"],
            index=["mole", "gram", "mass_fraction"].index(display_unit),
            key="display_unit_select",
        )
    # In row 2, have clear mix button
    with c2:
        if st.button("🧹 Clear mix", use_container_width=True):
            set_mix_rows([])
            st.rerun()

    st.session_state["display_unit"] = new_display_unit
    display_unit = new_display_unit

    # Display rows in dataframe in the middle of the screen
    df = rows_to_display_df(rows, display_unit)

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Species": st.column_config.TextColumn(disabled=True),
            "Quantity": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.6g"),
        },
        key="mix_table_editor",
    )

    # Apply the edits made to the table!
    if st.button("Apply table edits", use_container_width=True, key="apply_edits_btn"):
        try:
            new_rows = apply_table_edits_to_rows(edited, rows, display_unit)
            set_mix_rows(new_rows)
            st.success("Updated mixture.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    # Calculate the total moles for internal use and total mass in grams
    if rows:
        total_moles = sum(r.moles for r in rows)
        total_mass = sum(r.moles * molar_mass_of(r.comp) for r in rows)
        s1, s2 = st.columns(2)
        s1.metric("Total moles (internal)", f"{total_moles:.6g}")
        s2.metric("Total mass (g)", f"{total_mass:.6g}")
    else:
        st.info("Add components on the left to start building your mixture.")


with right:
    # Add export functionality
    st.subheader("Export")

    export_unit = st.selectbox(
        "Export units",
        ["mole", "gram", "mass_fraction"],
        index=["mole", "gram", "mass_fraction"].index(export_unit),
        key="export_unit_select",
    )
    st.session_state["export_unit"] = export_unit

    # Get the rows from the data frameto export csv
    rows = get_mix_rows()
    csv_bytes = export_csv_bytes(rows, export_unit)

    # Button to download CSV (TODO: Add path input)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="glyceride_mix.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=(len(rows) == 0),
    )

    st.divider()
    st.subheader("Build object")

    if st.button("Build GlycerideMix", use_container_width=True, disabled=(len(rows) == 0)):
        try:
            mix_dict = {r.comp: r.moles for r in rows}
            mix_obj = GlycerideMix(mix_dict, units="mole", sort=True)
            st.success(f"Built: {mix_obj.name}")
            st.code(repr(mix_obj))
        except Exception as e:
            st.error(str(e))

# close card
st.markdown("</div>", unsafe_allow_html=True)

