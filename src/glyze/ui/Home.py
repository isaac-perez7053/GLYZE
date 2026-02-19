import streamlit as st

st.set_page_config(page_title="GLYZE", page_icon="🧈", layout="centered")

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
        max-width: 920px;
        padding-top: 1.2rem;
        padding-bottom: 2.8rem;
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
        font-size:2.7rem;
        line-height:1.0;
        background:linear-gradient(90deg, #0F172A 0%, #1D4ED8 55%, #7C3AED 100%);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        margin:0;
    }

    .butter-badge{
        width:44px; height:44px;
        border-radius:14px;
        display:grid; place-items:center;
        background:linear-gradient(145deg, rgba(255,237,213,0.95), rgba(219,234,254,0.85));
        border:1px solid rgba(15,23,42,0.08);
        box-shadow:0 10px 22px rgba(15,23,42,0.08);
        font-size:22px;
    }

    .glyze-tagline{
        margin:0.7rem auto 0.2rem auto;
        max-width:60ch;
        color:var(--muted);
        font-size:1.05rem;
    }

    /* ---- Replace the weird "search bar" divider ---- */
    .glyze-divider{
        width: min(640px, 92%);
        height: 1px;
        margin: 1.2rem auto 1.1rem auto;
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

    /* Center text above buttons */
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

    /* ---- Buttons: neutral by default, highlight ONLY on hover ---- */
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

    /* Hover highlight (applies to ALL buttons, including the "primary" one) */
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

    /* IMPORTANT: remove Streamlit's default "primary" styling so it isn't permanently highlighted */
    button[kind="primary"]{
        background: rgba(255,255,255,0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06) !important;
    }

    .glyze-footer{
        margin-top:1.3rem;
        text-align:center;
        color:#64748B;
        font-size:0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="center glyze-hero">
        <div class="glyze-logo">
            <div class="glyze-word">GLYZE</div>
            <div class="butter-badge">🧈</div>
        </div>
        <div class="glyze-tagline">
            Glyceride and Lipid sYnthetiZation Engine — model, simulate, and calculate lipid workflows in one place.
        </div>
    </div>
    <div class="glyze-divider"></div>
    """,
    unsafe_allow_html=True,
)

# Card + nav
st.markdown('<div class="glyze-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="start-title">Where do you want to start?</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="start-subtitle">Choose a module to launch. You can always return here from the sidebar.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button("⚗️  Batch Reactor", use_container_width=True):
        st.switch_page("pages/batch_reactor_ui.py")

    if st.button("🧪  Viscosity Model", use_container_width=True):
        st.switch_page("pages/viscosity_model_ui.py")

with col2:
    if st.button("🧼  Deoderizor", use_container_width=True):
        st.switch_page("pages/deoderizor_ui.py")

    if st.button("📈  SFC Calculator", use_container_width=True):
        st.switch_page("pages/sfc_calculator_ui.py")

st.markdown("</div>", unsafe_allow_html=True)
