import math


def solve_V0(Va, S, A, tol=1e-12, max_iter=100):

    # If starting concentration is 0, return 0
    if Va == 0:
        return 0.0
    # Begin with a starting guess
    V0 = Va * 0.5

    #
    def f(v):
        return A * math.log(Va / v) + (A - 1) * (Va - v) - S

    def df(v):
        return -A / v - (A - 1)

    for _ in range(max_iter):

        # First solve for the value of Steam for a given species
        fv = f(V0)
        if abs(fv) < tol:
            return max(V0, 0.0)

        step = fv / df(V0)
        Vnew = V0 - step

        if Vnew <= 0 or Vnew >= Va:
            Vnew = 0.5 * V0

        V0 = Vnew

    raise RuntimeError("V0 solver failed")


def evaluate_composition(S, components, Pt, E):
    """
    Solve Bailey's equation
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

    mole_fractions = {k: v / total_final for k, v in final.items()}

    return mole_fractions, final, total_final


def optimize_stripping(
    components,
    fatty_acid_names,
    target_FA_fraction,
    Pt,
    E,
    S_bounds=(1e-6, 5.0),
    tol=1e-6,
):

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

    final_x, final_moles, total_final = evaluate_composition(S_mid, components, Pt, E)

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

components = {
    "oleic": {"moles": 0.05, "vapor_pressure": 0.012},
    "palmitic": {"moles": 0.03, "vapor_pressure": 0.007},
    "stearic": {"moles": 0.02, "vapor_pressure": 0.005},
    "triglyceride": {"moles": 0.90, "vapor_pressure": 1e-6},
}

fatty_acids = ["oleic", "palmitic", "stearic"]

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
