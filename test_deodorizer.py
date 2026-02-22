import math


# -------------------------
# Bailey inverse solver
# -------------------------

def solve_V0(Va, S, A, tol=1e-12, max_iter=200):
    """
    Solve Bailey's stripping equation for the final moles of a component.

    This function computes the remaining moles V0 of a volatile component
    after steam stripping, using Bailey’s deodorization equation:

        S = A*ln(Va/V0) + (A-1)*(Va-V0)

    where A = Pt/(E*Pi). The equation is solved numerically using a
    hybrid Newton–bisection method with physical bounds enforcement.

    Parameters
    ----------
    Va : float
        Initial moles of the component.
    S : float
        Steam stripping factor (moles steam per mole oil).
    A : float
        Bailey vaporization parameter Pt/(E*Pi).
    tol : float, optional
        Convergence tolerance for the root solver.
    max_iter : int, optional
        Maximum number of solver iterations.

    Returns
    -------
    float
        Final moles V0 remaining after stripping.
        A small positive value is returned when stripping is effectively complete.

    Notes
    -----
    - Enforces physical bounds to prevent log singularities.
    - Automatically handles extreme stripping cases.
    - Guaranteed non-negative output.
    """

    if Va == 0:
        return 0.0

    # physical bounds
    lo = 1e-18
    hi = Va

    # bailey's equation rearranged to f(V) = 0
    def f(v):
        return A * math.log(Va / v) + (A - 1) * (Va - v) - S

    flo = f(lo)
    fhi = f(hi)

    # if stripping is extreme → effectively zero
    # prevents numerical issues with log(0) and ensures we don't return a negative value
    if flo < 0:
        return lo

    # initial guess
    V = Va * 0.5

    for _ in range(max_iter):

        fv = f(V)

        if abs(fv) < tol:
            return max(V, lo)

        # derivative (Newton step)
        df = -A / V - (A - 1)

        if df != 0:
            Vn = V - fv / df
        else:
            Vn = None

        # enforce bounds
        if Vn is None or Vn <= lo or Vn >= hi:
            Vn = 0.5 * (lo + hi)

        fn = f(Vn)

        # update bracket
        if fn > 0:
            lo = Vn
        else:
            hi = Vn

        V = Vn

    return max(V, lo)


# -------------------------
# Composition evaluation
# -------------------------

def evaluate_composition(S, components, Pt, E):
    """
    Evaluate final mixture composition after steam stripping.

    For each component, Bailey’s equation is solved to determine the
    remaining moles after stripping. Final mole fractions are then
    computed by normalization.

    Parameters
    ----------
    S : float
        Steam stripping factor.
    components : dict
        Dictionary describing mixture components:
            {
                name: {
                    "moles": initial moles,
                    "vapor_pressure": Pi
                }
            }
    Pt : float
        Total system pressure.
    E : float
        Vaporization efficiency factor.

    Returns
    -------
    mole_fractions : dict
        Final mole fraction of each component.
    final_moles : dict
        Final moles of each component.
    total_final : float
        Total moles remaining in the mixture.

    Notes
    -----
    Each component is treated independently under Bailey’s stripping model.
    """

    final = {}
    total_final = 0.0

    for name, c in components.items():

        Va = c["moles"]
        Pi = c["vapor_pressure"]

        A = Pt / (E * Pi)

        V0 = solve_V0(Va, S, A)

        final[name] = V0
        total_final += V0

    mole_fractions = {
        k: v / total_final for k, v in final.items()
    }

    return mole_fractions, final, total_final


# -------------------------
# Steam optimizer
# -------------------------

def optimize_stripping(
    components,
    fatty_acid_names,
    target_FA_fraction,
    Pt,
    E,
    S_bounds=(1e-6, 5.0),
    tol=1e-6,
):
    """
    Optimize steam factor to reach a target fatty acid fraction.

    This function determines the steam stripping factor S required to
    reduce the total mole fraction of specified fatty acids to a desired
    threshold. A bisection search is used to ensure monotonic convergence.

    Parameters
    ----------
    components : dict
        Mixture definition with initial moles and vapor pressures.
    fatty_acid_names : list
        Names of components treated as fatty acids.
    target_FA_fraction : float
        Desired final total mole fraction of fatty acids.
    Pt : float
        Total system pressure.
    E : float
        Vaporization efficiency.
    S_bounds : tuple, optional
        Lower and upper bounds for the steam factor search.
    tol : float, optional
        Convergence tolerance for fatty acid fraction.

    Returns
    -------
    dict
        Optimization results including:
            - optimal_S : float
            - steam_percent : float
            - final_mole_fractions : dict
            - final_moles : dict
            - initial_total_moles : float
            - final_total_moles : float
            - total_removed : float
            - final_FA_fraction : float

    Notes
    -----
    Uses repeated Bailey equation evaluations to enforce a
    multicomponent material balance.
    """
    S_low, S_high = S_bounds

    def FA_fraction(S):

        x, _, _ = evaluate_composition(S, components, Pt, E)
        return sum(x[name] for name in fatty_acid_names)

    for _ in range(100):

        S_mid = 0.5 * (S_low + S_high)
        FA_mid = FA_fraction(S_mid)

        if abs(FA_mid - target_FA_fraction) < tol:
            break

        if FA_mid > target_FA_fraction:
            S_low = S_mid
        else:
            S_high = S_mid

    final_x, final_moles, total_final = evaluate_composition(
        S_mid, components, Pt, E
    )

    initial_total = sum(c["moles"] for c in components.values())

    return {
        "optimal_S": S_mid,
        "steam_percent": 2 * S_mid,
        "final_mole_fractions": final_x,
        "final_moles": final_moles,
        "initial_total_moles": initial_total,
        "final_total_moles": total_final,
        "total_removed": initial_total - total_final,
        "final_FA_fraction": sum(final_x[n] for n in fatty_acid_names),
    }


# -------------------------
# Example system
# -------------------------


# code breaks with these values, but they are correct
# the final total moles is essentially zero...
components = {
    "caprylic acid":        {"moles": 0.2, "vapor_pressure": 6.5416e-1},
    "capric acid":     {"moles": 0.2, "vapor_pressure": 7.3305e-2},
    "tricaprylin": {"moles": 0.2, "vapor_pressure": 8.818e-2},
    "tricaprin":  {"moles": 0.2, "vapor_pressure": 6.804e-3},
    "triacetin": {"moles": 0.2, "vapor_pressure": 7.988e2},
}

fatty_acids = ["caprylic acid", "capric acid"]

Pt = 0.000263
E = 0.5
target_FA = 0.001

result = optimize_stripping(
    components,
    fatty_acids,
    target_FA,
    Pt,
    E,
)

# -------------------------
# Reporting
# -------------------------

print("\n=== Steam Optimization Results ===")

print(f"Optimal steam factor S: {result['optimal_S']:.6f}")
print(f"Steam % of oil: {result['steam_percent']:.2f}%")

print(f"\nFinal fatty acid fraction: {result['final_FA_fraction']:.6f}")

print("\nFinal composition:")
for k, v in result["final_mole_fractions"].items():
    print(f"{k:15s} {v:.6f}")

print("\nMaterial balance:")
print("Initial total moles:", result["initial_total_moles"])
print("Final total moles:", result["final_total_moles"])
print("Total removed:", result["total_removed"])