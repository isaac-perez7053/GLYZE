import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import tempfile
from glyze.glyceride import FattyAcid, Glyceride
from glyze.glyceride_mix import GlycerideMix


st.set_page_config(page_title="GLYZE — Glyceride Mix", page_icon="🧈", layout="wide")


PAGE_CSS = """
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
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button{
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

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover{
    transform: translateY(-1px);
    background: linear-gradient(90deg, var(--accentA), var(--accentB)) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.20) !important;
    box-shadow: 0 18px 35px rgba(37, 99, 235, 0.22);
    filter: brightness(1.02);
}

div.stButton > button:active,
div[data-testid="stFormSubmitButton"] > button:active{
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
"""

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

# Card wrapper
st.markdown('<div class="glyze-card">', unsafe_allow_html=True)

MM_H2O = 18.01528
MM_GLYCEROL = 92.09382
DISPLAY_UNITS = ["Moles", "Grams", "Mass Fractions"]
ADD_UNITS = ["Moles", "Grams"]


@lru_cache(maxsize=512)
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
            raise ValueError(
                "DB count must match number of DB positions and stereochem entries."
            )
        name += "P" + "".join(
            f"{int(p):02d}{st.upper()}" for p, st in zip(db_positions, db_stereo)
        )

    if methyl_positions:
        name += "M" + "".join(f"{int(p):02d}" for p in methyl_positions)

    if hydroxyl_positions:
        name += "OH" + "".join(f"{int(p):02d}" for p in hydroxyl_positions)

    return name


def comp_display_name(comp: Any) -> str:
    if comp in {"H2O", "Glycerol"}:
        return str(comp)
    if hasattr(comp, "name"):
        return str(comp.name)
    return str(comp)


@lru_cache(maxsize=1024)
def molar_mass_from_name(name: str) -> float:
    """
    Grab the molar mass of a component
    """
    if name == "H2O":
        return MM_H2O
    if name == "Glycerol":
        return MM_GLYCEROL
    comp = parse_component_from_string(name)
    if hasattr(comp, "molar_mass"):
        return float(comp.molar_mass)
    raise ValueError(f"Cannot determine molar mass for component {name!r}")


def molar_mass_of(comp: Any) -> float:
    return molar_mass_from_name(comp_display_name(comp))


def get_file_name(file_object):
    if hasattr(file_object, "name"):
        return file_object.name
    else:
        return "In-memory stream or no name available"


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

    if qty_unit == "Moles":
        moles = qty
    elif qty_unit == "Grams":
        moles = qty / molar_mass_of(comp)
    else:
        raise ValueError(f"Unsupported add unit: {qty_unit}")

    rows = get_mix_rows()
    rows.append(MixRow(comp, moles))
    set_mix_rows(merge_rows(rows))


def rows_signature(rows: List[MixRow]) -> Tuple[Tuple[str, float, float], ...]:
    return tuple(
        (comp_display_name(r.comp), float(r.moles), molar_mass_of(r.comp)) for r in rows
    )


@st.cache_data(show_spinner=False)
def rows_to_display_df_cached(
    sig: Tuple[Tuple[str, float, float], ...], display_unit: str
) -> pd.DataFrame:
    if not sig:
        return pd.DataFrame({"Species": [], "Quantity": []})

    species = [name for name, _, _ in sig]
    moles = [moles for _, moles, _ in sig]
    molar_masses = [mm for _, _, mm in sig]

    if display_unit == "Moles":
        qty = moles
    elif display_unit == "Grams":
        qty = [m * mm for m, mm in zip(moles, molar_masses)]
    elif display_unit == "Mass Fractions":
        masses = [m * mm for m, mm in zip(moles, molar_masses)]
        total = sum(masses) or 1.0
        qty = [m / total for m in masses]
    else:
        raise ValueError(f"Unsupported display unit: {display_unit}")

    return pd.DataFrame({"Species": species, "Quantity": qty})


def rows_to_display_df(rows: List[MixRow], display_unit: str) -> pd.DataFrame:
    return rows_to_display_df_cached(rows_signature(rows), display_unit)


def apply_table_edits_to_rows(
    edited_df: pd.DataFrame, rows: List[MixRow], display_unit: str
) -> List[MixRow]:
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
        if display_unit == "Moles":
            moles = q
        elif display_unit == "Grams":
            moles = q / molar_mass_of(comp) if q > 0 else 0.0
        elif display_unit == "Mass Fractions":
            # Interpret as fractions on a 1 g basis; renormalize after
            moles = (q * 1.0) / molar_mass_of(comp) if q > 0 else 0.0
        else:
            raise ValueError(f"Unsupported unit: {display_unit}")

        new_rows.append(MixRow(comp, moles))

    if display_unit == "Mass Fractions" and new_rows:
        fracs = edited_df["Quantity"].astype(float).tolist()
        s = sum(fracs)
        if s > 0:
            fracs = [f / s for f in fracs]
            new_rows = [
                MixRow(r.comp, (fracs[i] * 1.0) / molar_mass_of(r.comp))
                for i, r in enumerate(new_rows)
            ]

    return merge_rows(new_rows)


# Turn the data frame to a csv (TODO: Ensure that this dataframe is accessible
# to other pages and can be turned into a GlycerideMix)
@st.cache_data(show_spinner=False)
def export_csv_bytes_cached(
    sig: Tuple[Tuple[str, float, float], ...], export_unit: str
) -> bytes:
    df = rows_to_display_df_cached(sig, export_unit).copy()
    df.insert(2, "Units", export_unit)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_csv_bytes(rows: List[MixRow], export_unit: str) -> bytes:
    return export_csv_bytes_cached(rows_signature(rows), export_unit)


def parse_int_list(s: str) -> List[int]:
    # Parse the integers (then will be input as strings)
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_stereo_list(s: str) -> List[str]:
    # Parse the lists (they are also input as strings)
    s = (s or "").strip()
    if not s:
        return []
    out = [x.strip().upper() for x in s.split(",") if x.strip()]
    for x in out:
        if x not in {"Z", "E"}:
            raise ValueError("Stereo must be Z or E.")
    return out


def import_mix_from_uploaded_csv(uploaded_file) -> List[MixRow]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        mix = GlycerideMix.from_csv(str(tmp_path))
        return [MixRow(comp, moles) for comp, moles in mix.mix.items()]
    finally:
        tmp_path.unlink(missing_ok=True)


def build_mix_object(rows: List[MixRow]) -> GlycerideMix:
    mix_dict = {r.comp: r.moles for r in rows}
    return GlycerideMix(mix_dict, units="Moles", sort=True)


def update_glyze_mixture():
    rows = get_mix_rows()
    try:
        mix_obj = build_mix_object(rows)
        st.session_state["glyze_mix_object"] = mix_obj
        st.session_state["glyze_mix_rows"] = rows
        st.session_state["glyze_mix_initialized"] = True
    except Exception as e:
        st.error(str(e))


display_unit = st.session_state.setdefault("display_unit", "Moles")
export_unit = st.session_state.setdefault("export_unit", "Moles")

left, mid, right = st.columns([1.2, 1.9, 1.2], gap="large")

# On the left of the screen, add a species using the code from the first iteration of the ai
with left:
    st.subheader("Add species")

    # Dictionary containing common name and GLYZE name
    built_fas = {
        "None": None,
        "Propionic acid": "N03D00",
        "Butyric acid": "N04D00",
        "Valeric acid": "N05D00",
        "Caproic acid": "N06D00",
        "Heptanoic acid": "N07D00",
        "Caprylic acid": "N08D00",
        "Nonanoic acid": "N09D00",
        "Capric acid": "N10D00",
        "Undecaonic acid": "N11D00",
        "Lauric acid": "N12D00",
        "Tridecanoic acid": "N13D00",
        "Myristic acid": "N14D00",
        "Pentadecanoic acid": "N15D00",
        "Palmitic acid": "N16D00",
        "Heptadecanoic acid": "N17D00",
        "Stearic acid": "N18D00",
        "Oleic acid": "N18D01P09Z",
        "Arachidic acid": "N20D00",
    }

    # Build preset fatty acids
    with st.expander("Built-in fatty acids", expanded=False):

        # Create a dropdown menu using the keys of the dictionary
        fa_name = st.selectbox(
            "Select a fatty acid to add",
            options=list(built_fas.keys()),
            key="built_fa_select",
            placeholder="None",
        )

        qty = st.number_input(
            "Quanitity",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="built_fa_qty_builtinfa",
        )

        # Units
        qty_unit = st.selectbox(
            "Quantity units", ADD_UNITS, key="built_fa_qty_unit_builtinfa"
        )

        # Add Button
        quick_add = st.button(
            "Add selected fatty acid", use_container_width=True, key="built_fa_add"
        )

        if quick_add:
            try:
                comp = parse_component_from_string(built_fas[fa_name])
                add_to_mix(comp, qty, qty_unit)
                st.success(f"Added {fa_name} ({qty:g} {qty_unit}).")
            except Exception as e:
                st.error(str(e))

    # Build preset glycerides
    with st.expander("Built-in glycerides", expanded=False):

        # First allow option to pick from preset Pure Triglycerides
        built_pure_tags = {
            "None": None,
            "Tributrin": "G_N04D00_N04D00_N04D00",
            "Tricaproin": "G_N06D00_N06D00_N06D00",
            "Tricaprylin": "G_N08D00_N08D00_N08D00",
            "Tricaprin": "G_N10D00_N10D00_N10D00",
            "Trilaurin": "G_N12D00_N12D00_N12D00",
            "Trimyristin": "G_N14D00_N14D00_N14D00",
            "Tripalmitin": "G_N16D00_N16D00_N16D00",
            "Tristearin": "G_N18D00_N18D00_N18D00",
            "Triolein": "G_N18D01P09Z_N18D01P09Z_N18D01P09Z",
        }
        tag_name = st.selectbox(
            "Select pure triglyeceride to add",
            options=list(built_pure_tags.keys()),
            key="built_pure_select",
            placeholder="None",
        )

        # Line to demarcate the two options
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Give option to make glyceride from preset fatty acid
        fa1_name = st.selectbox(
            "FA 1 (sn-1)",
            options=list(built_fas.keys()),
            key="built_g_fa1",
            placeholder="None",
        )
        fa2_name = st.selectbox(
            "FA 2 (sn-2)",
            options=list(built_fas.keys()),
            key="built_g_fa2",
            placeholder="None",
        )
        fa3_name = st.selectbox(
            "FA 3 (sn-3)",
            options=list(built_fas.keys()),
            key="built_g_fa3",
            placeholder="None",
        )

        # Input qty for the glyceride to be added
        qty = st.number_input(
            "Quanitity",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="built_fa_qty_builtintg",
        )

        # Units
        qty_unit = st.selectbox(
            "Quantity units", ADD_UNITS, key="built_fa_qty_unit_builtintg"
        )

        quick_add = st.button(
            "Add selected glyceride", use_container_width=True, key="built_g_add"
        )

        if quick_add:
            try:
                # Ensure that only one of the options is selected
                if (
                    built_fas.get(fa1_name)
                    and built_fas.get(fa2_name)
                    and built_fas.get(fa3_name)
                    and built_pure_tags.get(tag_name)
                ):
                    st.error("Please select only one of the above options")

                # Option for biulding triglyceride form 3 preset fatty acids
                elif (
                    built_fas.get(fa1_name)
                    and built_fas.get(fa2_name)
                    and built_fas.get(fa3_name)
                ):
                    comp = parse_component_from_string(
                        f"G_{built_fas[fa1_name]}_{built_fas[fa2_name]}_{built_fas[fa3_name]}"
                    )

                # OPtion for selecting a preset pure trigyceride by name
                elif tag_name:
                    comp = parse_component_from_string(built_pure_tags[tag_name])
                else:
                    st.error("Please select an option to add a glyceride")

                add_to_mix(comp, qty, qty_unit)
                st.success(f"Added {comp_display_name(comp)} ({qty:g} {qty_unit}).")
            except Exception as e:
                st.error(str(e))

    # This section is meant for adding either water or glycerol
    with st.expander("Quick add extra species", expanded=False):
        #
        extra_species = ["H2O", "Glycerol"]
        extra_name = st.selectbox(
            "Select a species to add", options=extra_species, key="extra_species_select"
        )
        qty = st.number_input(
            "Quanitity", min_value=0.0, value=1.0, step=0.1, key="extra_species_qty"
        )
        qty_unit = st.selectbox(
            "Quantity units", ADD_UNITS, key="extra_species_qty_unit"
        )
        quick_add = st.button(
            "Add selected species", use_container_width=True, key="extra_species_add"
        )

        if quick_add:
            try:
                comp = parse_component_from_string(extra_name)
                add_to_mix(comp, qty, qty_unit)
                st.success(f"Added {extra_name} ({qty:g} {qty_unit}).")
            except Exception as e:
                st.error(str(e))

    # Quick add by name
    with st.expander("Quick add by name", expanded=False):
        with st.form("quick_add_form", clear_on_submit=False):
            # How to use instructions
            st.write(
                "Examples: `N18D01P09Z`, `G_N18D1P09Z_N16D00_N16D00`, `H2O`, `Glycerol`"
            )

            # Text input for name
            name_str = st.text_input("Component name", key="quick_name")

            # qty input
            qty = st.number_input(
                "Quantity", min_value=0.0, value=1.0, step=0.1, key="quick_qty"
            )

            # Units
            qty_unit = st.selectbox("Quantity units", ADD_UNITS, key="quick_qty_unit")

            # Add button
            quick_add = st.form_submit_button("Add", use_container_width=True)

        if quick_add:
            try:
                comp = parse_component_from_string(name_str)
                add_to_mix(comp, qty, qty_unit)
                st.success(f"Added {name_str} ({qty:g} {qty_unit}).")
            except Exception as e:
                st.error(str(e))

with mid:
    # Controls the list in the middle of the screen
    st.subheader("Mixture")

    rows = get_mix_rows()

    # Establish both columns
    c1, c2 = st.columns([1, 1])
    with c1:
        # Display units in select box
        new_display_unit = st.selectbox(
            "Display/edit units",
            DISPLAY_UNITS,
            index=DISPLAY_UNITS.index(display_unit),
            key="display_unit",
        )

    # In row 2, have clear mix button
    with c2:
        if st.button("Clear mix", use_container_width=True):
            set_mix_rows([])
            st.rerun()

    display_unit = st.session_state["display_unit"]

    # Display rows in dataframe in the middle of the screen
    df = rows_to_display_df(rows, display_unit)

    with st.form("mix_editor_form", clear_on_submit=False):
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Species": st.column_config.TextColumn(disabled=True),
                "Quantity": st.column_config.NumberColumn(
                    min_value=0.0, step=0.000001, format="%.15g"
                ),
            },
            key="mix_table_editor",
        )

        # Apply the edits made to the table!
        apply_edits = st.form_submit_button(
            "Apply table edits", use_container_width=True
        )

        # Update glyze mixture globally
        update_glyze_mixture()

    if apply_edits:
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
        DISPLAY_UNITS,
        index=DISPLAY_UNITS.index(export_unit),
        key="export_unit_select",
    )
    st.session_state["export_unit"] = export_unit

    # Get the rows from the data frameto export csv
    rows = get_mix_rows()
    csv_bytes = export_csv_bytes(rows, export_unit)

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="glyceride_mix.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=(len(rows) == 0),
    )

    with st.form("import_csv_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Upload your CSV file", type=["csv"], key="import_csv_uploader"
        )
        import_csv = st.form_submit_button(
            "Import CSV", use_container_width=True, disabled=(uploaded_file is None)
        )

    if import_csv and uploaded_file is not None:
        try:
            imported_rows = import_mix_from_uploaded_csv(uploaded_file)
            st.success(f"Imported: {uploaded_file.name}")
            set_mix_rows(imported_rows)
            st.rerun()
        except Exception as e:
            st.error(f"Error importing CSV: {str(e)}")

    st.divider()
    st.subheader("Build object")

    if st.button(
        "Update Chem Processor", use_container_width=True, disabled=(len(rows) == 0)
    ):
        try:
            mix_obj = build_mix_object(rows)
            st.session_state["glyze_mix_object"] = mix_obj
            st.session_state["glyze_mix_rows"] = rows
            st.session_state["processor_mix_rows"] = rows
            st.session_state["glyze_mix_initialized"] = True
            st.success("Mixture sent to the Chem Processor page.")
        except Exception as e:
            st.error(str(e))

# close card
st.markdown("</div>", unsafe_allow_html=True)
