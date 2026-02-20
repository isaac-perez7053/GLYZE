# import numpy as np
# from scipy.optimize import brentq
# import pandas as pd


# class DSC:

#     def sfc(mix: GlycerideMix, model: str):
#         """ """

#     # Database of TAG properties (beta-prime)
#     TAG_DATABASE = {
#         "TAG": [
#             "C4",
#             "C5",
#             "C6",
#             "C7",
#             "C8",
#             "C9",
#             "C10",
#             "C11",
#             "C12",
#             "C13",
#             "C14",
#             "C15",
#             "C16",
#             "C17",
#             "C18",
#             "C19",
#             "C20",
#         ],
#         "M_gmol": [
#             302.367,
#             344.448,
#             386.529,
#             428.610,
#             470.691,
#             512.772,
#             554.853,
#             596.934,
#             639.015,
#             681.096,
#             723.177,
#             765.258,
#             807.339,
#             849.420,
#             891.501,
#             933.582,
#             975.663,
#         ],
#         "dHfus_betaPrime_Jg": [
#             130.4,
#             137.06,
#             142.44,
#             126.655,
#             128.45,
#             101.6,
#             161.825,
#             130.75,
#             158.1,
#             134.55,
#             171.35,
#             165.4,
#             191.015,
#             175.1,
#             200.05,
#             172.8,
#             193.575,
#         ],
#         "Tm_betaPrime_C": [
#             -75,
#             -60,
#             -42.5,
#             -7.8,
#             -4.1,
#             8.535,
#             23.83,
#             29.23,
#             40.30,
#             42.435,
#             51.67,
#             54.48,
#             61.37,
#             62.335,
#             68.395,
#             68.09,
#             73.585,
#         ],
#     }

#     # test cases
#     TEST_CASES = [
#         {
#             "id": 86,
#             "TAG_names": ["C8", "C10", "C14", "C16"],
#             "w": [0.15, 0.25, 0.5, 0.1],
#             "exp_SFC": [59, 55, 47, 33, 18],
#         },
#         {
#             "id": 132,
#             "TAG_names": [
#                 "C4",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C16",
#                 "C17",
#                 "C18",
#                 "C19",
#                 "C20",
#             ],
#             "w": [
#                 0.06,
#                 0.09,
#                 0.10,
#                 0.06,
#                 0.06,
#                 0.09,
#                 0.08,
#                 0.03,
#                 0.03,
#                 0.09,
#                 0.08,
#                 0.07,
#                 0.05,
#                 0.04,
#             ],
#             "exp_SFC": [27.295, 18.035, 11.195, 7.27, 4.095],
#         },
#         {
#             "id": 215,
#             "TAG_names": [
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C14",
#                 "C16",
#                 "C17",
#                 "C18",
#             ],
#             "w": [0.105, 0.118, 0.162, 0.158, 0.064, 0.058, 0.007, 0.163, 0.152, 0.013],
#             "exp_SFC": [56, 48, 36, 15, 5],
#         },
#         {
#             "id": 113,
#             "TAG_names": [
#                 "C4",
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C14",
#                 "C15",
#                 "C16",
#                 "C17",
#                 "C18",
#                 "C19",
#                 "C20",
#             ],
#             "w": [
#                 0.044,
#                 0.051,
#                 0.059,
#                 0.066,
#                 0.071,
#                 0.075,
#                 0.075,
#                 0.073,
#                 0.07,
#                 0.068,
#                 0.065,
#                 0.062,
#                 0.059,
#                 0.055,
#                 0.045,
#                 0.035,
#                 0.027,
#             ],
#             "exp_SFC": [35, 29, 19, 10, 4.2],
#         },
#         {
#             "id": 134,
#             "TAG_names": [
#                 "C4",
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C14",
#                 "C15",
#                 "C16",
#                 "C17",
#                 "C18",
#                 "C19",
#                 "C20",
#             ],
#             "w": [
#                 0.06,
#                 0.07,
#                 0.05,
#                 0.06,
#                 0.05,
#                 0.06,
#                 0.1,
#                 0.1,
#                 0.07,
#                 0.07,
#                 0.08,
#                 0.08,
#                 0.06,
#                 0.05,
#                 0.04,
#             ],
#             "exp_SFC": [36, 30, 24, 14.76, 8.5],
#         },
#         {
#             "id": 183,
#             "TAG_names": ["C6", "C8", "C10", "C12", "C14", "C16"],
#             "w": [0.05, 0.15, 0.2, 0.05, 0.45, 0.1],
#             "exp_SFC": [58, 53.8, 44, 27, 15],
#         },
#         {
#             "id": 312,
#             "TAG_names": [
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C15",
#                 "C16",
#                 "C17",
#                 "C18",
#                 "C19",
#                 "C20",
#             ],
#             "w": [
#                 0.019,
#                 0.053,
#                 0.087,
#                 0.018,
#                 0.041,
#                 0.099,
#                 0.111,
#                 0.03,
#                 0.039,
#                 0.004,
#                 0.186,
#                 0.157,
#                 0.11,
#                 0.04,
#                 0.006,
#             ],
#             "exp_SFC": [62.67, 58.69, 49, 34.34, 22.67],
#         },
#         {
#             "id": 369,
#             "TAG_names": ["C6", "C8", "C10", "C14", "C16"],
#             "w": [0.05, 0.1, 0.3, 0.35, 0.20],
#             "exp_SFC": [58.8, 51.9, 42.7, 28.8, 14.7],
#         },
#         {
#             "id": 145,
#             "TAG_names": [
#                 "C4",
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C14",
#                 "C16",
#                 "C17",
#                 "C18",
#             ],
#             "w": [
#                 0.007,
#                 0.043,
#                 0.085,
#                 0.112,
#                 0.068,
#                 0.075,
#                 0.101,
#                 0.098,
#                 0.041,
#                 0.037,
#                 0.005,
#                 0.1,
#                 0.093,
#                 0.135,
#             ],
#             "exp_SFC": [20.39, 18.56, 14.33, 6.395, 3.415],
#         },
#         {
#             "id": 151,
#             "TAG_names": ["C10", "C12", "C18"],
#             "w": [0.5, 0.4, 0.1],
#             "exp_SFC": [83.5, 76.88, 52.125, 0.075, -0.21],
#         },
#         {
#             "id": 137,
#             "TAG_names": [
#                 "C4",
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C14",
#                 "C16",
#                 "C17",
#                 "C18",
#             ],
#             "w": [
#                 0.005,
#                 0.044,
#                 0.091,
#                 0.122,
#                 0.077,
#                 0.087,
#                 0.12,
#                 0.117,
#                 0.048,
#                 0.043,
#                 0.005,
#                 0.12,
#                 0.111,
#                 0.01,
#             ],
#             "exp_SFC": [16.2, 11.705, 3.755, 1.095, 0.205],
#         },
#         {
#             "id": 141,
#             "TAG_names": ["C4", "C6", "C8", "C10", "C12", "C14", "C15", "C16", "C18"],
#             "w": [0.05, 0.03, 0.01, 0.03, 0.04, 0.11, 0.02, 0.26, 0.09],
#             "exp_SFC": [54.965, 53.705, 48.465, 34.34, 24.165],
#         },
#         {
#             "id": 309,
#             "TAG_names": ["C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15"],
#             "w": [0.1, 0.1, 0.05, 0.05, 0.15, 0.15, 0.2, 0.2],
#             "exp_SFC": [73.12, 66.725, 58.26, 41.165, 23.28],
#         },
#         {
#             "id": 359,
#             "TAG_names": ["C6", "C8", "C10", "C12", "C14", "C16"],
#             "w": [0.05, 0.15, 0.3, 0.1, 0.3, 0.1],
#             "exp_SFC": [44.27, 35.86, 20.1, 7.135, 1.235],
#         },
#         {
#             "id": 114,
#             "TAG_names": ["C4", "C5", "C6", "C7", "C18", "C19", "C20"],
#             "w": [0.13, 0.16, 0.18, 0.2, 0.14, 0.11, 0.08],
#             "exp_SFC": [81.225, 79.315, 76.655, 65.695, 45.355],
#         },
#         {
#             "id": 139,
#             "TAG_names": ["C4", "C6", "C8", "C10", "C12", "C14", "C15", "C16", "C18"],
#             "w": [0.07, 0.04, 0.02, 0.05, 0.06, 0.17, 0.03, 0.42, 0.14],
#             "exp_SFC": [86.245, 83.235, 78.93, 70.685, 61.515],
#         },
#         {
#             "id": 117,
#             "TAG_names": [
#                 "C4",
#                 "C5",
#                 "C6",
#                 "C7",
#                 "C8",
#                 "C9",
#                 "C10",
#                 "C11",
#                 "C12",
#                 "C13",
#                 "C14",
#                 "C15",
#                 "C16",
#                 "C17",
#                 "C18",
#             ],
#             "w": [
#                 0.003,
#                 0.027,
#                 0.056,
#                 0.075,
#                 0.086,
#                 0.092,
#                 0.093,
#                 0.091,
#                 0.087,
#                 0.084,
#                 0.08,
#                 0.077,
#                 0.074,
#                 0.069,
#                 0.006,
#             ],
#             "exp_SFC": [31.88, 27.3, 17.51, 8.785, 3.265],
#         },
#         {
#             "id": 358,
#             "TAG_names": ["C8", "C10", "C14", "C16"],
#             "w": [0.2, 0.15, 0.2, 0.45],
#             "exp_SFC": [67.84, 68.85, 62.435, 48.445, 31.59],
#         },
#     ]

#     def tagprops(TAG_names):
#         # get TAG properties
#         db = pd.DataFrame(TAG_DATABASE)

#         # Find matching rows
#         mask = db["TAG"].isin(TAG_names)
#         db_filtered = db[db["TAG"].isin(TAG_names)]

#         # Ensure order matches TAG_names
#         db_row = db_filtered.set_index("TAG").loc[TAG_names].reset_index()

#         # Calculate enthalpy of fusion in J/mol
#         dHf_Jmol = db_row["dHfus_betaPrime_Jg"].values * db_row["M_gmol"].values

#         # Convert melting temp to Kelvin
#         Tf_K = db_row["Tm_betaPrime_C"].values + 273.15

#         return dHf_Jmol, Tf_K, db_row

#     def smoothing(TAG_names, x, dHf_Jmol, Tf_K, Tgrid_C, w_C):
#         # smoothing method
#         x = np.array(x).flatten()
#         dHf_Jmol = np.array(dHf_Jmol).flatten()
#         Tf_K = np.array(Tf_K).flatten()
#         Tgrid_C = np.array(Tgrid_C).flatten()

#         N = len(x)

#         # Validate inputs
#         assert len(TAG_names) == N and len(dHf_Jmol) == N and len(Tf_K) == N
#         assert np.all(np.isfinite(x)) and np.all(x >= 0)
#         assert abs(np.sum(x) - 1) < 1e-6
#         assert np.all(np.isfinite(dHf_Jmol)) and np.all(dHf_Jmol > 0)
#         assert np.all(np.isfinite(Tf_K)) and np.all(Tf_K > 0)
#         assert np.isfinite(w_C) and w_C > 0

#         xSum = np.sum(x)
#         xRest = max(0, 1 - xSum)
#         xTot = 1 - xRest

#         R = 8.314462618

#         # Saturation function
#         def x_sat(dHf, Tf, T):
#             return np.exp(-(dHf / R) * (1 / T - 1 / Tf))

#         # Calculate pair solidus temperatures
#         npairs = N * (N - 1) // 2
#         ii = []
#         jj = []
#         Tij_K = []

#         for i in range(N - 1):
#             for j in range(i + 1, N):
#                 ii.append(i)
#                 jj.append(j)

#                 xpair = x[i] + x[j]
#                 if xpair <= 0:
#                     Tij_K.append(np.nan)
#                     continue

#                 Thigh = min(Tf_K[i], Tf_K[j]) - 1e-6
#                 Tlow = 1.0

#                 def f(T):
#                     return (
#                         x_sat(dHf_Jmol[i], Tf_K[i], T)
#                         + x_sat(dHf_Jmol[j], Tf_K[j], T)
#                         - xpair
#                     )

#                 try:
#                     T_solidus = brentq(f, Tlow, Thigh)
#                     Tij_K.append(T_solidus)
#                 except:
#                     Tij_K.append(np.nan)

#         ii = np.array(ii)
#         jj = np.array(jj)
#         Tij_K = np.array(Tij_K)
#         Tij_C = Tij_K - 273.15

#         # Calculate P(A;B) weighting
#         P = np.zeros((N, N))
#         for A in range(N):
#             denom = 0
#             for k in range(N):
#                 if k == A:
#                     continue
#                 denom += x[A] + x[k]

#             for B in range(N):
#                 if B == A:
#                     continue
#                 P[A, B] = (x[A] + x[B]) / denom if denom > 0 else 0

#         # Calculate phi_ij
#         raw = np.zeros(npairs)
#         for k in range(npairs):
#             A = ii[k]
#             B = jj[k]
#             raw[k] = x[A] * P[A, B] + x[B] * P[B, A]

#         c = xTot / np.sum(raw) if np.sum(raw) > 0 else 0
#         phi_pct = 100 * c * raw

#         # Calculate smoothed SFC(T)
#         SFC_pct = np.zeros(len(Tgrid_C))
#         for t in range(len(Tgrid_C)):
#             z = (Tgrid_C[t] - Tij_C) / w_C
#             s = 1 / (1 + np.exp(z))  # smooth sigmoid
#             SFC_pct[t] = np.sum(phi_pct * s)

#         return {"T_C": Tgrid_C, "SFC_pct": SFC_pct, "Tij_C": Tij_C, "phi_pct": phi_pct}

#     def run_single_case(case, w_C=2.5):
#         """Run a single test case and return RMSE."""
#         TAG_names = case["TAG_names"]
#         w = np.array(case["w"])
#         exp_SFC = np.array(case["exp_SFC"])

#         # Normalize mass fractions
#         w = w / np.sum(w)

#         # Get TAG properties
#         dHf_Jmol, Tf_K, db_row = tagprops(TAG_names)

#         # Convert mass fractions to mole fractions
#         M_gmol = db_row["M_gmol"].values
#         n = w / M_gmol
#         x = n / np.sum(n)

#         # Temperature grid
#         Tmin_C = 0
#         Tmax_C = 50
#         dT_C = 0.5
#         Tgrid_C = np.arange(Tmin_C, Tmax_C + dT_C, dT_C)

#         # Run prediction
#         out = smoothing(TAG_names, x, dHf_Jmol, Tf_K, Tgrid_C, w_C)

#         # interpolate at experimental temperatures
#         exp_T_C = np.array([5, 10, 15, 20, 25])
#         pred_at_exp = np.interp(exp_T_C, out["T_C"], out["SFC_pct"])

#         # Calculate RMSE and MAE
#         err = pred_at_exp - exp_SFC
#         rmse = np.sqrt(np.mean(err**2))
#         mae = np.mean(np.abs(err))

#         return {"id": case["id"], "RMSE": rmse, "MAE": mae, "n_TAGs": len(TAG_names)}

#     def main():
#         """Run all test cases and display RMSE results."""
#         print("=" * 70)
#         print("SFC Analysis - RMSE Results for All TAG Sets")
#         print("=" * 70)
#         print()

#         results = []

#         for case in TEST_CASES:
#             try:
#                 result = run_single_case(case, w_C=2.5)
#                 results.append(result)
#                 print(
#                     f"Case {result['id']:3d} ({result['n_TAGs']:2d} TAGs): RMSE = {result['RMSE']:7.3f} SFC%,  MAE = {result['MAE']:7.3f} SFC%"
#                 )
#             except Exception as e:
#                 print(f"Case {case['id']:3d}: ERROR - {str(e)}")

#         print()
#         print("=" * 70)
#         print("Summary Statistics")
#         print("=" * 70)

#         if results:
#             rmse_values = [r["RMSE"] for r in results]
#             mae_values = [r["MAE"] for r in results]

#             print(f"Number of cases analyzed: {len(results)}")
#             print(f"Average RMSE: {np.mean(rmse_values):.3f} SFC%")
#             print(f"Median RMSE:  {np.median(rmse_values):.3f} SFC%")
#             print(
#                 f"Min RMSE:     {np.min(rmse_values):.3f} SFC% (TAG {results[np.argmin(rmse_values)]['id']})"
#             )
#             print(
#                 f"Max RMSE:     {np.max(rmse_values):.3f} SFC% (TAG {results[np.argmax(rmse_values)]['id']})"
#             )
#             print()
#             print(f"Average MAE:  {np.mean(mae_values):.3f} SFC%")

#         print("=" * 70)
