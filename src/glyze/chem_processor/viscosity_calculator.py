import re
import numpy as np
import plotly.graph_objects as go

from typing import Dict, Iterable, List, Tuple, Any
from collections import defaultdict

from glyze.glyceride_mix import GlycerideMix


# Empirical coefficients (from MATLAB)
COEFS: Dict[str, Tuple[float, float, float, float]] = {
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


MOLAR_MASS: Dict[str, float] = {
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
    "WATER": 18.01528,
    "GLYCEROL": 92.09382,
}


def floats_from(s: str) -> np.ndarray:
    """
    Convert a comma/space separated string into a float array.
    """
    vals = [x for x in re.split(r"[,\s]+", (s or "").strip()) if x]
    return np.array([float(x) for x in vals], dtype=float) if vals else np.array([], dtype=float)


def range_for(tag: str) -> Tuple[float, float]:
    """
    Valid temperature ranges by tag.

    C2-C10:  20-80 C
    C11-C13: 30-80 C
    C14-C16: 50-80 C
    C17-C19: 70-80 C

    WATER / GLYCEROL are treated separately and allowed broadly here.
    """
    if tag == "WATER":
        return 0.0, 100.0
    if tag == "GLYCEROL":
        return 20.0, 100.0

    n = int(tag[1:])
    if 2 <= n <= 10:
        return 20.0, 80.0
    if 11 <= n <= 13:
        return 30.0, 80.0
    if 14 <= n <= 16:
        return 50.0, 80.0
    if 17 <= n <= 19:
        return 70.0, 80.0
    raise ValueError(f"Unsupported tag '{tag}'.")


def mass_to_mole_frac(tags: Iterable[str], mass_fracs: Iterable[float]) -> np.ndarray:
    """
    Convert mass fractions into mole fractions.
    """
    tags = list(tags)
    w = np.asarray(mass_fracs, dtype=float)
    if len(tags) != len(w):
        raise ValueError("tags and mass_fracs must have the same length.")
    if np.any(w < 0):
        raise ValueError("Mass fractions must be non-negative.")
    if np.isclose(w.sum(), 0.0):
        raise ValueError("Mass fractions must sum to a positive value.")

    w = w / w.sum()
    M = np.array([MOLAR_MASS[t] for t in tags], dtype=float)
    n = w / M
    return n / n.sum()


class ViscosityCalculator:
    """
    Utilities for estimating viscosity curves for a GlycerideMix.

    The current model maps the mixture onto an effective distribution of TAG
    families Cn and now also allows WATER and GLYCEROL as explicit weighted
    components in the final mixing rule.
    """

    @staticmethod
    def mu(T: np.ndarray, A: float, B: float, C: float, E: float) -> np.ndarray:
        """
        Empirical viscosity model for a pure TAG family.

        Parameters
        ----------
        T : np.ndarray
            Temperature in C.
        A, B, C, E : float
            Empirical coefficients.

        Returns
        -------
        np.ndarray
            Viscosity in cP.
        """
        T = np.asarray(T, dtype=float)
        return np.exp(A + B / (T + C) + E / (T ** 2))

    @staticmethod
    def mu_water(T: np.ndarray) -> np.ndarray:
        """
        Pure water viscosity in cP.

        T is in C.
        Correlation form gives viscosity in Pa*s, converted to cP.
        """
        T = np.asarray(T, dtype=float)
        mu_pa_s = 2.414e-5 * 10.0 ** (247.8 / (T + 133.15))
        return 1000.0 * mu_pa_s

    @staticmethod
    def mu_glycerol(T: np.ndarray) -> np.ndarray:
        """
        Pure glycerol viscosity in cP.

        A simple Andrade-style fit in ln(mu[cP]) over moderate temperatures.
        This is intended for practical interpolation on your Streamlit page.
        """
        T = np.asarray(T, dtype=float)
        Tk = T + 273.15

        # Simple smooth fit:
        # ln(mu[cP]) = A + B / Tk
        # Chosen to give roughly realistic values in the common liquid range.
        A = -8.20
        B = 3820.0
        return np.exp(A + B / Tk)

    @staticmethod
    def pure_viscosity(tag: str, T: np.ndarray) -> np.ndarray:
        """
        Dispatch pure viscosity by tag.
        """
        if tag == "WATER":
            return ViscosityCalculator.mu_water(T)
        if tag == "GLYCEROL":
            return ViscosityCalculator.mu_glycerol(T)
        return ViscosityCalculator.mu(T, *COEFS[tag])

    @staticmethod
    def _normalize_tag(length: int) -> str:
        """
        Convert an integer carbon length to a supported tag string, e.g. 18 -> 'C18'.
        """
        tag = f"C{int(length)}"
        if tag not in COEFS:
            raise ValueError(
                f"Fatty-acid/TAG family '{tag}' is not available in COEFS."
            )
        return tag

    @staticmethod
    def _unwrap_component(component: Any) -> Any:
        """
        If the object is a MixtureComponent-like wrapper, return the underlying component.
        """
        return getattr(component, "component", component)

    @staticmethod
    def _component_name(component: Any) -> str:
        """
        Best-effort component name.
        """
        component = ViscosityCalculator._unwrap_component(component)
        return str(getattr(component, "name", component)).strip().lower()

    @staticmethod
    def _is_water_like(component: Any) -> bool:
        """
        Try to detect water entries.
        """
        name = ViscosityCalculator._component_name(component)
        return name in {"water", "h2o", "wat", "sol", "tip3p", "tip4p", "hoh"}

    @staticmethod
    def _is_empty_glyceride(component: Any) -> bool:
        """
        Treat a glyceride with no attached fatty acids as glycerol.
        """
        component = ViscosityCalculator._unwrap_component(component)
        if not hasattr(component, "sn"):
            return False

        try:
            sn = getattr(component, "sn")
            return sn is not None and len(sn) == 3 and all(fa is None for fa in sn)
        except Exception:
            return False


    @staticmethod
    def _is_glycerol_like(component: Any) -> bool:
        """
        Try to detect glycerol entries.
        """
        if ViscosityCalculator._is_empty_glyceride(component):
            return True

        name = ViscosityCalculator._component_name(component)
        return name in {"glycerol", "gly", "glycerin", "propane-1,2,3-triol"}

    @staticmethod
    def _fatty_acid_lengths_from_component(component: Any) -> List[int]:
        """
        Extract fatty-acid lengths from a component in a tolerant way.

        Supported patterns:
        - MixtureComponent-like wrapper via .component
        - component.length
        - component.sn -> iterable of FA-like objects with .length
        - component.fatty_acids -> iterable of FA-like objects with .length
        - component.fa1, component.fa2, component.fa3
        - component.acyl_chains -> iterable with .length
        """
        component = ViscosityCalculator._unwrap_component(component)

        # Free fatty acid case: object itself has a length
        if hasattr(component, "length") and not hasattr(component, "sn"):
            return [int(component.length)]

        # Glyceride .sn tuple/list pattern
        if hasattr(component, "sn"):
            sn = getattr(component, "sn")

            if sn is None:
                raise ValueError(f"{component!r} has sn=None")

            lengths = []
            for i, fa in enumerate(sn):
                if fa is None:
                    continue

                if not hasattr(fa, "length"):
                    raise TypeError(
                        f"Component {component!r} has sn[{i}]={fa!r}, "
                        f"which does not have a .length attribute"
                    )

                val = fa.length
                if val is None:
                    raise ValueError(
                        f"Component {component!r} has sn[{i}]={fa!r} with length=None"
                    )

                lengths.append(int(val))

            if lengths:
                return lengths

            raise ValueError(
                f"Component {component!r} has an sn tuple, but no valid fatty-acid lengths were found. "
                f"sn={sn!r}"
            )
        # Common container attribute
        if hasattr(component, "fatty_acids"):
            lengths = []
            for fa in getattr(component, "fatty_acids"):
                if fa is not None and hasattr(fa, "length"):
                    lengths.append(int(fa.length))
            if lengths:
                return lengths

        # Alternate chain container attribute
        if hasattr(component, "acyl_chains"):
            lengths = []
            for fa in getattr(component, "acyl_chains"):
                if fa is not None and hasattr(fa, "length"):
                    lengths.append(int(fa.length))
            if lengths:
                return lengths

        # Explicit fa1/fa2/fa3 pattern
        lengths = []
        for attr in ("fa1", "fa2", "fa3"):
            if hasattr(component, attr):
                fa = getattr(component, attr)
                if fa is not None and hasattr(fa, "length"):
                    lengths.append(int(fa.length))
        if lengths:
            return lengths

        raise ValueError(
            f"Could not determine fatty-acid lengths for component: {component!r}"
        )

    @staticmethod
    def parse_input_mix(mix: GlycerideMix) -> List[Tuple[str, float]]:
        """
        Convert a GlycerideMix into an effective component distribution.

        The weighting rule is:
        - gather all fatty acids in the mixture, whether attached or free
        - weight them by the component quantity in the mix
        - include water explicitly as WATER
        - include glycerol explicitly as GLYCEROL
        - return relative contributions grouped by chain length / special component

        Returns
        -------
        List[Tuple[str, float]]
            Sorted list like [('C18', 0.5), ('C16', 0.2), ('WATER', 0.2), ('GLYCEROL', 0.1)].
        """
        if not mix.mix:
            raise ValueError("GlycerideMix is empty.")

        result_dict: Dict[str, float] = defaultdict(float)

        # Gather list of fatty acid lengths
        for component, quantity in mix.mix.items():
            if quantity < 0:
                raise ValueError("Component quantities must be non-negative.")

            if ViscosityCalculator._is_water_like(component):
                result_dict["WATER"] += float(quantity)
                continue

            if ViscosityCalculator._is_glycerol_like(component):
                result_dict["GLYCEROL"] += float(quantity)
                continue

            lengths = ViscosityCalculator._fatty_acid_lengths_from_component(component)
            for length in lengths:
                tag = ViscosityCalculator._normalize_tag(length)
                result_dict[tag] += float(quantity)

        if not result_dict:
            raise ValueError(
                "No supported fatty-acid, water, or glycerol content was found in the provided mixture."
            )

        total = sum(result_dict.values())
        if np.isclose(total, 0.0):
            raise ValueError("Parsed component weights sum to zero.")

        def sort_key(item: Tuple[str, float]):
            tag = item[0]
            if tag.startswith("C"):
                return (0, -int(tag[1:]))
            if tag == "GLYCEROL":
                return (1, 0)
            if tag == "WATER":
                return (2, 0)
            return (3, 0)

        # Final sorted relative concentrations
        return sorted(
            ((tag, value / total) for tag, value in result_dict.items()),
            key=sort_key
        )

    @staticmethod
    def validate_temperature_range(tags: Iterable[str], init_temp: float, final_temp: float):
        """
        Check whether the requested temperature window is valid for every tag.
        """
        tags = list(tags)
        t_min = float(init_temp)
        t_max = float(final_temp)

        if t_max < t_min:
            raise ValueError("final_temp must be greater than or equal to init_temp.")

        invalid = []
        for tag in tags:
            low, high = range_for(tag)
            if t_min < low or t_max > high:
                invalid.append(f"{tag}: valid range is {low:g}-{high:g} C")

        if invalid:
            raise ValueError(
                "Requested temperature range is outside the empirical validity window:\n"
                + "\n".join(invalid)
            )

    @staticmethod
    def calculate(
        mix: GlycerideMix,
        init_temp: float,
        final_temp: float,
        step_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute pure-component and mixture viscosity curves from a GlycerideMix.

        Parameters
        ----------
        mix : GlycerideMix
            Mixture of glycerides / fatty acids / water / glycerol.
        init_temp : float
            Initial temperature in C.
        final_temp : float
            Final temperature in C.
        step_size : float, optional
            Step size in C.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing tags, weights, mole fractions, temperature grid,
            pure-component viscosities, and the mixture viscosity.
        """
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        parsed = ViscosityCalculator.parse_input_mix(mix)
        tags = [tag for tag, _ in parsed]
        mass_fracs = np.array([weight for _, weight in parsed], dtype=float)

        ViscosityCalculator.validate_temperature_range(tags, init_temp, final_temp)

        # Compute mixture viscosity (log-mixing in mole-fraction space)
        x = mass_to_mole_frac(tags, mass_fracs)
        T = np.arange(float(init_temp), float(final_temp) + 0.5 * step_size, float(step_size))
        U = np.vstack([ViscosityCalculator.pure_viscosity(tag, T) for tag in tags])
        u_mix = np.exp(x @ np.log(U))

        return {
            "tags": tags,
            "mass_fractions": mass_fracs / mass_fracs.sum(),
            "mole_fractions": x,
            "temperature_C": T,
            "pure_viscosities_cP": U,
            "mixture_viscosity_cP": u_mix,
        }

    @staticmethod
    def make_plot(
        result: Dict[str, Any],
        title: str = "Viscosity vs Temperature"
    ) -> go.Figure:
        """
        Build an interactive Plotly figure from calculate(...).

        Returns
        -------
        go.Figure
            Plotly figure object.
        """
        tags = result["tags"]
        T = result["temperature_C"]
        U = result["pure_viscosities_cP"]
        u_mix = result["mixture_viscosity_cP"]

        fig = go.Figure()

        for i, tag in enumerate(tags):
            fig.add_trace(
                go.Scatter(
                    x=T,
                    y=U[i],
                    mode="lines",
                    name=tag,
                    hovertemplate=(
                        "Component: %{fullData.name}<br>"
                        "T: %{x:.2f} C<br>"
                        "μ: %{y:.6g} cP"
                        "<extra></extra>"
                    ),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=T,
                y=u_mix,
                mode="lines",
                name="Mixture",
                line={"dash": "dash", "width": 3},
                hovertemplate=(
                    "Component: Mixture<br>"
                    "T: %{x:.2f} C<br>"
                    "μ: %{y:.6g} cP"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Temperature (C)",
            yaxis_title="Viscosity (cP)",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Curve",
        )

        return fig

    @staticmethod
    def calculate_and_plot(
        mix: GlycerideMix,
        init_temp: float,
        final_temp: float,
        step_size: float = 1.0,
        title: str = "Viscosity vs Temperature"
    ) -> Tuple[Dict[str, Any], go.Figure]:
        """
        Convenience wrapper that computes the result and returns the Plotly figure.
        """
        result = ViscosityCalculator.calculate(
            mix=mix,
            init_temp=init_temp,
            final_temp=final_temp,
            step_size=step_size,
        )
        fig = ViscosityCalculator.make_plot(result=result, title=title)
        return result, fig

    @staticmethod
    def to_csv_string(result: Dict[str, Any]) -> str:
        """
        Convert a result dictionary into a CSV string for download/use in Streamlit.
        """
        tags = result["tags"]
        T = result["temperature_C"]
        U = result["pure_viscosities_cP"]
        u_mix = result["mixture_viscosity_cP"]

        header = ["T_C", *tags, "Mixture"]
        rows = []

        for i in range(len(T)):
            row = [f"{T[i]:g}"]
            row.extend(f"{U[j, i]:.8g}" for j in range(len(tags)))
            row.append(f"{u_mix[i]:.8g}")
            rows.append(",".join(row))

        return ",".join(header) + "\n" + "\n".join(rows)