from glyze.glyceride_mix import GlycerideMix
from scipy import optimize
import math

class Deodorizer(GlycerideMix):

    def __init__(self, mix: GlycerideMix):
        super().__init__(mix.mix)

    @staticmethod
    def _solve_V0(Va, S, A, tol=1e-12, maxiter=100):
        if Va == 0:
            return 0.0

        # Bailey's equation wnere the right hand side is equal to zero. 
        def f(v):
            return A * math.log(Va / v) + (A - 1) * (Va - v) - S

        # It's derivative
        def df(v):
            return -A / v - (A - 1)

        # starting guess in (0, Va)
        x0 = Va * 0.5
        # to ensure v stays in (eps, Va-eps)
        eps = 1e-16

        try:
            return optimize.newton(f, x0, fprime=df, tol=tol, maxiter=maxiter)
        
        # Use other methods if the Newton-Raphson root solver fails
        except (RuntimeError, OverflowError, ZeroDivisionError):
            # find a bracket [a,b] where f(a)*f(b) < 0
            a, b = eps, Va - eps
            # try to shrink bracket by sweeping inward if sign not found
            N = 50
            for i in range(N):
                try:
                    # Extremely useful root solver
                    if f(a) * f(b) < 0:
                        return optimize.brentq(f, a, b, xtol=tol, maxiter=maxiter)
                except ValueError:
                    pass
                #TODO: progressively move a inward, b inward etc.
            return optimize.brentq(f, a, b, xtol=tol, maxiter=maxiter)
        
    
    def single_pass(self, S: float, T:float, P: float, E: float):
        """
        Pass the GlycerideMixture through the deoderizer a single time
        """
        total_final = 0.0

        for component, _ in self.mix.items():

            Va = self.mix[component]
            A = P / (E * component.vapor_pressure(T, P))
            V0 = Deodorizer._solve_V0(Va, S, A)
            # Edit the qty of the current GlycerideMix
            self.change_qty(component, V0)
            total_final += V0

        mole_fractions = {
            k: v / total_final for k, v in self.mix.items()
        }

        return (mole_fractions, total_final)

    def deodorizer(self, T: float, P: float, E: float=0.5, Pt: float=0.000263, target: float=0.001, sbounds=(1e-6, 5), tol=1e-6, nsteps=100, verbose=True):
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

            x, (_, _) = self.single_pass(S, T, Pt, E)
            return sum(x[name] for name in self.fa_list)

        for _ in range(nsteps):

            S_mid = 0.5 * (sbounds[0] + sbounds[1])
            FA_mid = FA_fraction(S_mid)

            if abs(FA_mid - target) < tol:
                break

            if FA_mid > target:
                sbounds[0] = S_mid
            else:
                sbounds[1] = S_mid

        final_x, (_, total_final) = self.single_pass(S_mid, T, Pt, E)
        initial_total = sum(self.mix.values())

        if verbose:
            print("\n=== Steam Optimization Results ===")
            print(f"Optimal steam factor S: {S_mid:.6f}")
            print(f"Steam % of oil: {2 * S_mid:.2f}%")
            print(f"\nFinal fatty acid fraction: {sum(final_x[n] for n in self.fa_list):.6f}")
            print("\nFinal composition:")
            for k, v in final_x.items():
                print(f"{k:15s} {v:.6f}")

            print("\nMaterial balance:")
            print("Initial total moles:", initial_total)
            print("Final total moles:", total_final)
            print("Total removed:", initial_total - total_final)

        # Return ideal steam value
        return S_mid

