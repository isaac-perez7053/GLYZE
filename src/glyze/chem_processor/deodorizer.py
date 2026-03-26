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

    @staticmethod
    def deodorizer(mix: GlycerideMix, S: float, T: float, P: float):
        """
        Pass the GlycerideMixture through the deodorizer a single time.
        Works on a snapshot of current quantities so self.mix is never mutated.
        """
        total_final = 0.0
        result_quantities = {}

        for component, Va in mix.mix.items():
            A = P / (component.vapor_pressure(T))
            V0 = Deodorizer._solve_V0(Va, S, A)
            result_quantities[component] = V0
            total_final += V0
        return (result_quantities, total_final)

    def plot(self, mix):
        return

    @staticmethod
    def opt_deodorizer(
        mix: GlycerideMix,
        T: float,               # Temperature, [K]
        P: float,               # System pressure, [Pa]
        target: float = 0.001,  # target fatty acid fraction after deodorization [mol/mol]
        sbounds=(1e-6, 5),      # bounds for steam factor S [mol/mol oil]
        tol=1e-6,               # tolerance for bisection method convergence
        nsteps=100,             # number of steps for bisection method
        verbose=False,          # print results or not
        plot=False              # plot results or not
    ):
        """
        Perform deodorization on a glyceride mix at a given temperature and pressure.

        Parameters:
        -----------
            T (float): Temperature in Kelvin.
            P (float): Pressure in Pa.
            target (float): Target fatty acid mole fraction after deodorization.
            sbounds (tuple): Search bounds for steam factor S [mol steam / mol oil].
            tol (float): Bisection convergence tolerance.
            nsteps (int): Maximum bisection iterations.
            verbose (bool): Print diagnostics.
            plot (bool): Plot results.

        Returns:
        -----------
            float: Optimal steam factor S_mid.
        """
        # sbounds must be mutable for bisection
        sbounds = list(sbounds)

        def FA_fraction(S):
            x, total_final = Deodorizer.deodorizer(mix, S, T, P)
            return sum(qty for comp, qty in x.items() if comp.component in mix.fa_list) / total_final

        S_mid = 0.5 * (sbounds[0] + sbounds[1])

        # Bisection to find the steam factor S that achieves the target FA fraction
        for _ in range(nsteps):
            S_mid = 0.5 * (sbounds[0] + sbounds[1])
            FA_mid = FA_fraction(S_mid)

            if abs(FA_mid - target) < tol:
                break

            if FA_mid > target:
                sbounds[0] = S_mid
            else:
                sbounds[1] = S_mid

        # Snapshot initial quantities before mutation
        initial_x = {comp: qty for comp, qty in mix.mix.items()}
        initial_total = sum(initial_x.values())

        final_x, total_final = Deodorizer.deodorizer(mix, S_mid, T, P)

        # Update mix.mix and mix.quantities with final absolute quantities.
        # change_qty only writes to self.quantities (not self.mix), so we
        # update both data structures here directly.
        for component, qty in final_x.items():
            mix.mix[component] = qty
            key = _canonical_key(_canonical_component(component))
            if key in mix.index_by_key:
                mix.quantities[mix.index_by_key[key]] = qty

        if verbose:
            print("\n=== Steam Optimization Results ===")
            print(f"Optimal steam factor S: {S_mid:.6f}")
            print(f"Steam % of oil: {2 * S_mid:.2f}%")
            print(f"\nFinal fatty acid fraction: {sum(qty for comp, qty in final_x.items() if comp.component in mix.fa_list) / total_final:.6f}")
            print(f"\n{'Species':30s} {'Initial':>12s} {'Final':>12s} {'Removed':>12s}")
            print("-" * 68)
            for comp in initial_x:
                qi = initial_x[comp]
                qf = final_x[comp]
                print(f"{comp.name:30s} {qi:12.6f} {qf:12.6f} {qi - qf:12.6f}")
            print("\nMaterial balance:")
            print(f"Initial total moles: {initial_total:.6f}")
            print(f"Final total moles:   {total_final:.6f}")
            print(f"Total removed:       {initial_total - total_final:.6f}")

        if plot:
            names, qi_vals, qf_vals = [], [], []
            for comp in initial_x:
                if not isinstance(comp.component, (Glyceride, FattyAcid)):
                    continue
                names.append(comp.name)
                qi_vals.append(initial_x[comp])
                qf_vals.append(final_x[comp])

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


if __name__ == "__main__":
    """Example usage of the Deodorizer class"""