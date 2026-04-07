from glyze.glyceride_mix import GlycerideMix, _canonical_key, _canonical_component
from glyze.glyceride import Glyceride, FattyAcid
from scipy import optimize
import math
import plotly.graph_objects as go

class Deodorizer:

    @staticmethod
    def _solve_V0(Va, S, A, tol=1e-12, maxiter=100):
        """
        Solve for V0 (remaining moles after stripping) given:
            Va  - initial moles of component
            S   - steam stripping factor [mol steam / mol oil]
            A   - relative volatility ratio P / (E * vapor_pressure)

        The equation is:
            A * ln(Va/V0) + (A - 1)*(Va - V0) - S = 0

        When A << 1 (highly volatile component, low system pressure), the component
        can be completely stripped and f(v) > 0 for all v in (0, Va), meaning no
        finite root exists. Physically this means V0 → 0, so we return a near-zero
        residual (tol) in that case.
        """
        if Va == 0:
            return 0.0

        eps = 1e-16

        def f(v):
            return A * math.log(Va / v) + (A - 1) * (Va - v) - S

        def df(v):
            return -A / v - (A - 1)

        # ------------------------------------------------------------------ #
        # Quick bracket check before attempting any root finder.              #
        # f(Va) = -S < 0 always (since log(1)=0 and last term vanishes).     #
        # If f(eps) <= 0 as well, the root is outside (0, Va) — the          #
        # component is completely stripped; return near-zero residual.        #
        # ------------------------------------------------------------------ #
        try:
            f_lo = f(eps)
        except (ValueError, ZeroDivisionError, OverflowError):
            # log of a tiny positive number may overflow; treat as +inf → bracketed
            f_lo = float('inf')

        f_hi = -S  # f(Va) = A*log(1) + (A-1)*0 - S = -S

        if f_lo <= 0:
            # No root in (0, Va): component is effectively completely stripped
            return tol

        # f_lo > 0 and f_hi < 0 → valid bracket exists, proceed with Newton/Brent
        x0 = Va * 0.5

        try:
            result = optimize.newton(f, x0, fprime=df, tol=tol, maxiter=maxiter)
            if result <= 0 or result > Va:
                raise ValueError("Newton result out of valid range")
            return result

        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            a, b = eps, Va - eps
            try:
                return optimize.brentq(f, a, b, xtol=tol, maxiter=maxiter)
            except ValueError:
                # Bracket still failed — fall back to near-zero (fully stripped)
                return tol

    def deodorizer(mix: GlycerideMix, S: float, T: float, P: float,
                entrainment: float = 0.0, plot: bool = False) -> GlycerideMix:
        """
        Parameters
        ----------
        entrainment : float
            Fraction of each component lost to mechanical carryover
            (0.0-0.05 typical). Applied uniformly to all species
            after vapor stripping.
        """
        result_pairs = []

        for component, Va in mix.mix.items():
            A = P / (component.vapor_pressure(T))
            V0 = Deodorizer._solve_V0(Va, S, A)
            # Apply mechanical entrainment to whatever remains
            V0 *= (1.0 - entrainment)
            result_pairs.append((component, V0))

        final_mix = GlycerideMix(result_pairs, units=mix.units, sort=True)
        
        if plot:
            names, qi_vals, qf_vals = [], [], []
            for comp in mix.mix:
                if not isinstance(comp.component, (Glyceride, FattyAcid)):
                    continue
                names.append(comp.name)
                qi_vals.append(mix.mix[comp])
                qf_vals.append(final_mix.mix.get(comp, 0.0))

            fig = go.Figure([
                go.Bar(name="Initial", x=names, y=qi_vals),
                go.Bar(name="Final",   x=names, y=qf_vals),
            ])
            fig.update_layout(
                barmode="group",
                xaxis_title="Species",
                yaxis_title="Moles",
                xaxis_tickangle=-45,
            )
            fig.show()

        return final_mix

    @staticmethod
    def opt_deodorizer(
        mix: GlycerideMix,
        T: float,
        P: float,
        entrainment = 0.6,
        target: float = 0.001,
        sbounds=(1e-6, 5),
        tol=1e-6,
        nsteps=100,
        verbose=False,
        plot=False
    ):
        sbounds = list(sbounds)

        def FA_fraction(S):
            result_mix = Deodorizer.deodorizer(mix, S, T, P, entrainment)
            total = sum(result_mix.quantities)
            fa_total = sum(
                qty for comp, qty in result_mix.mix.items()
                if comp.component in mix.fa_list
            )
            return fa_total / total

        S_mid = 0.5 * (sbounds[0] + sbounds[1])

        for _ in range(nsteps):
            S_mid = 0.5 * (sbounds[0] + sbounds[1])
            FA_mid = FA_fraction(S_mid)

            if abs(FA_mid - target) < tol:
                break

            if FA_mid > target:
                sbounds[0] = S_mid
            else:
                sbounds[1] = S_mid

        initial_x = {comp: qty for comp, qty in mix.mix.items()}
        initial_total = sum(initial_x.values())

        final_mix = Deodorizer.deodorizer(mix, S_mid, T, P, entrainment)
        final_total = sum(final_mix.quantities)

        # Update the original mix in place
        for component, qty in final_mix.mix.items():
            mix.mix[component] = qty
            key = _canonical_key(_canonical_component(component))
            if key in mix.index_by_key:
                mix.quantities[mix.index_by_key[key]] = qty

        if verbose:
            print("\n=== Steam Optimization Results ===")
            print(f"Optimal steam factor S: {S_mid:.6f}")
            print(f"Steam % of oil: {2 * S_mid:.2f}%")
            fa_final = sum(
                qty for comp, qty in final_mix.mix.items()
                if comp.component in mix.fa_list
            )
            print(f"\nFinal fatty acid fraction: {fa_final / final_total:.6f}")
            print(f"\n{'Species':30s} {'Initial':>12s} {'Final':>12s} {'Removed':>12s}")
            print("-" * 68)
            for comp in initial_x:
                qi = initial_x[comp]
                qf = final_mix.mix.get(comp, 0.0)
                print(f"{comp.name:30s} {qi:12.6f} {qf:12.6f} {qi - qf:12.6f}")
            print("\nMaterial balance:")
            print(f"Initial total moles: {initial_total:.6f}")
            print(f"Final total moles:   {final_total:.6f}")
            print(f"Total removed:       {initial_total - final_total:.6f}")

        if plot:
            names, qi_vals, qf_vals = [], [], []
            for comp in initial_x:
                if not isinstance(comp.component, (Glyceride, FattyAcid)):
                    continue
                names.append(comp.name)
                qi_vals.append(initial_x[comp])
                qf_vals.append(final_mix.mix.get(comp, 0.0))

            fig = go.Figure([
                go.Bar(name="Initial", x=names, y=qi_vals),
                go.Bar(name="Final",   x=names, y=qf_vals),
            ])
            fig.update_layout(
                barmode="group",
                title=f"Deodorizer: Initial vs Final Mole Quantities (S={S_mid:.4f})",
                xaxis_title="Species",
                yaxis_title="Moles",
                xaxis_tickangle=-45,
            )
            fig.show()

        return S_mid