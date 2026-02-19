import streamlit as st

st.set_page_config(page_title="GLYZE — Deodorizer", page_icon="🧈", layout="wide")

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



# Card wrapper
st.markdown('<div class="glyze-card">', unsafe_allow_html=True)
st.markdown('<div class="start-title">Deodorizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="start-subtitle">Add species on the left, edit the mixture in the center, export on the right.</div>',
    unsafe_allow_html=True,
)
