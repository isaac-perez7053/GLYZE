from glyze.glyceride_mix import GlycerideMix
from scipy import optimize
import math
import plotly.graph_objects as go

class Deodorizer:

    def __init__(self, mix: GlycerideMix):
        self.mix = mix

    @staticmethod
    def _solve_V0(Va, S, A, tol=1e-12, maxiter=100):
        if Va == 0:
            return 0.0

        def f(v):
            return A * math.log(Va / v) + (A - 1) * (Va - v) - S

        def df(v):
            return -A / v - (A - 1)

        x0 = Va * 0.5
        eps = 1e-16

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
                raise ValueError(f"_solve_V0 failed to bracket a root for Va={Va}, S={S}, A={A}")

    def single_pass(self, S: float, T: float, P: float, E: float):
        """
        Pass the GlycerideMixture through the deodorizer a single time.
        Works on a snapshot of current quantities so self.mix is never mutated.
        """
        total_final = 0.0
        result_quantities = {}

        for component, Va in self.mix.mix.items():
            A = P / (E * component.vapor_pressure(T))
            V0 = Deodorizer._solve_V0(Va, S, A)
            result_quantities[component] = V0
            total_final += V0

        mole_fractions = {k: v / total_final for k, v in result_quantities.items()}
        return (mole_fractions, total_final)

    def plot(self, mix):
        return
    def deodorizer(
        self,
        T: float,               # Temperature, [K]
        P: float,               # System pressure, [Pa]
        E: float = 0.5,         # deodorization efficiency factor
        target: float = 0.001,  # target fatty acid fraction after deodorization [mol/mol]
        sbounds=(1e-6, 5),      # bounds for steam factor S [mol/mol oil]
        tol=1e-6,               # tolerance for bisection method convergence
        nsteps=100,             # number of steps for bisection method
        verbose=False,           # print results or not
        plot=False               # plot results or not
    ):
        """
        Perform deodorization on a glyceride mix at a given temperature and pressure.

        Parameters:
        -----------
            mix (GlycerideMix): The glyceride mix to be deodorized.
            T (float): Temperature in Kelvin.
            P (float): Pressure in atm.

        Returns:
        -----------
            GlycerideMix: The deodorized glyceride mix.
        """

        def FA_fraction(S):
            """
            report the sum of fatty acids in the final mixture after a single pass with steam factor S
            """
            x, total_final = self.single_pass(S, T, P, E)
            
            return sum(x[name] for name in self.fa_list)

        # use bisection method to find the steam factor S that achieves the target fatty acid fraction
        for _ in range(nsteps):

            S_mid = 0.5 * (sbounds[0] + sbounds[1])
            FA_mid = FA_fraction(S_mid)

            if abs(FA_mid - target) < tol:
                break

            if FA_mid > target:
                sbounds[0] = S_mid
            else:
                sbounds[1] = S_mid

        final_x,  total_final = self.single_pass(S_mid, T, P, E)
        
        # Update self.mix with final quantities
        for component, frac in final_x.items():
            self.mix.change_qty(component, frac * total_final)

        initial_total = sum(self.mix.values())

        if verbose:
            print("\n=== Steam Optimization Results ===")
            print(f"Optimal steam factor S: {S_mid:.6f}")
            print(f"Steam % of oil: {2 * S_mid:.2f}%")
            print(
                f"\nFinal fatty acid fraction: {sum(final_x[n] for n in self.fa_list):.6f}"
            )
            print("\nFinal composition:")
            for k, v in final_x.items():
                print(f"{k:15s} {v:.6f}")

            print("\nMaterial balance:")
            print("Initial total moles:", initial_total)
            print("Final total moles:", total_final)
            print("Total removed:", initial_total - total_final)

        # Return ideal steam value
        return S_mid


if __name__ == "__main__":
    """Example usage of the Deodorizer class"""
