from glyze.glyceride_mix import GlycerideMix
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
    def deodorizer(mix: GlycerideMix, S: float, T: float, P: float, E: float):
        """
        Pass the GlycerideMixture through the deodorizer a single time.
        Works on a snapshot of current quantities so self.mix is never mutated.
        """
        total_final = 0.0
        result_quantities = {}

        for component, Va in mix.mix.items():
            A = P / (E * component.vapor_pressure(T))
            V0 = Deodorizer._solve_V0(Va, S, A)
            result_quantities[component] = V0
            total_final += V0

        mole_fractions = {k: v / total_final for k, v in result_quantities.items()}
        return (mole_fractions, total_final)

    def plot(self, mix):
        return

    @staticmethod
    def opt_deodorizer(
        mix: GlycerideMix,
        T: float,               # Temperature, [K]
        P: float,               # System pressure, [Pa]
        E: float = 0.5,         # deodorization efficiency factor
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
            E (float): Deodorization efficiency factor (0–1).
            target (float): Target fatty acid mole fraction after deodorization.
            sbounds (tuple): Search bounds for steam factor S [mol steam / mol oil].
            tol (float): Bisection convergence tolerance.
            nsteps (int): Maximum bisection iterations.
            verbose (bool): Print diagnostics.
            plot (bool): Plot results (not yet implemented).

        Returns:
        -----------
            float: Optimal steam factor S_mid.
        """
        # sbounds must be mutable for bisection
        sbounds = list(sbounds)

        def FA_fraction(S):
            """Return the fatty acid mole fraction after a single pass at steam factor S."""
            x, total_final = Deodorizer.deodorizer(mix,S, T, P, E)
            return sum(x.get(fa, 0.0) for fa in mix.fa_list)

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

        final_x, total_final = Deodorizer.deodorizer(mix, S_mid, T, P, E)

        initial_total = sum(mix.mix.values())

        # Update self.mix with final quantities
        for component, frac in final_x.items():
            mix.change_qty(component, frac * total_final)
        

        if verbose:
            print("\n=== Steam Optimization Results ===")
            print(f"Optimal steam factor S: {S_mid:.6f}")
            print(f"Steam % of oil: {2 * S_mid:.2f}%")
            print(f"\nFinal fatty acid fraction: {sum(final_x[n] for n in mix.fa_list):.6f}")
            print("\nFinal composition:")
            for k, v in final_x.items():
                print(f"{k.name:15s} {v:.6f}")
            print("\nMaterial balance:")
            print("Initial total moles:", initial_total)
            print("Final total moles:", total_final)
            print("Total removed:", initial_total - total_final)

        return S_mid


if __name__ == "__main__":
    """Example usage of the Deodorizer class"""