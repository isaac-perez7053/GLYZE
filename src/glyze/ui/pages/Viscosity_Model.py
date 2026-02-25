from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="GLYZE — Viscosity Prediction Tool", page_icon="🧈", layout="wide")


# Empirical coefficients (from MATLAB)
COEFS = {
    "C2": (0.07, 105.2911, 0.0112, -885.4359),
    "C3": (-4.056, 967.9563, 134.5872, 31.4778),
    "C4": (-0.384050317, 122.5512048, -6.489359596, -2543.72728),
    "C6": (-0.4725, 156.0213, -7.1172, -3537.5806),
    "C7": (0.0084, 127.5008, -6.3182, -2561.4509),
    "C8": (0.1859, 129.532, -6.0275, -2510.8992),
    "C9": (0.0493, 149.653, -6.5789, -3139.137),
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
    "C2": 218.18,
    "C3": 260.28,
    "C4": 302.367,
    "C5": 344.448,
    "C6": 386.529,
    "C7": 428.61,
    "C8": 470.691,
    "C9": 512.772,
    "C10": 554.853,
    "C11": 596.934,
    "C12": 639.015,
    "C13": 681.096,
    "C14": 723.177,
    "C15": 765.258,
    "C16": 807.339,
    "C17": 849.42,
    "C18": 891.501,
    "C19": 933.582,
}


def tags_from(s: str):
    return [t for t in re.split(r"[,\s]+", (s or "").strip().upper()) if t]


def floats_from(s: str):
    vals = [x for x in re.split(r"[,\s]+", (s or "").strip()) if x]
    return np.array([float(x) for x in vals], dtype=float) if vals else np.array([], dtype=float)


def range_for(tag: str):
    n = int(tag[1:])
    if 2 <= n <= 10:
        return 20.0, 80.0
    if 11 <= n <= 13:
        return 30.0, 80.0
    if 14 <= n <= 16:
        return 50.0, 80.0
    if 17 <= n <= 19:
        return 70.0, 80.0
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


def validate_and_compute(tags_str_in: str, mf_str_in: str, tr_str_in: str):
    tags = tags_from(tags_str_in)
    if not tags:
        return None, "Alert: no TAGS provided."

    bad = [t for t in tags if (t not in COEFS) or (t not in MOLAR_MASS)]
    if bad:
        return None, f"Alert: unknown TAGS: {', '.join(bad)}"

    mf = floats_from(mf_str_in)
    if mf.size != len(tags):
        return None, f"Alert: expected {len(tags)} mass-fraction values."

    if np.any(mf < 0) or np.isclose(mf.sum(), 0.0):
        return None, "Alert: mass fractions must be nonnegative and not all zero."

    tr = floats_from(tr_str_in)
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

    # Compute mixture viscosity (log-mixing in mole-fraction space)
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

        left, mid, right = st.columns([1, 2, 1])
        with mid:
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        csv = (
            "T_C," + ",".join(tags) + ",Mixture\n"
            + "\n".join(
                [
                    ",".join(
                        [f"{T[i]:g}"]
                        + [f"{U[j, i]:.8g}" for j in range(len(tags))]
                        + [f"{u_mix[i]:.8g}"]
                    )
                    for i in range(len(T))
                ]
            )
        )

        st.download_button(
            "Download computed data (CSV)",
            data=csv,
            file_name="viscosity_results.csv",
            mime="text/csv",
        )
else:
    st.info("Edit inputs and press Enter (or toggle Auto-run), then run the calculation.")


st.divider()
st.subheader("References")

st.markdown(
    """
1. Anand, K., Ranjan, A., & Mehta, P. S. (2010). *Estimating the viscosity of vegetable oil and biodiesel fuels*. **Energy & Fuels, 24**, 664–672. https://doi.org/10.1021/ef900818s

2. Brookfield Engineering Laboratories, Inc. (n.d.). *DV-II+Pro viscometer: Operating instructions* (Manual No. M03-165-F0612). Brookfield Engineering Laboratories, Inc., Middleboro, MA, USA.
"""
)