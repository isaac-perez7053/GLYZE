
# Backend/Computation for Melting Enthalpy and Temperature predictor implementation in Python

# Sebastian Mollejas, smolleja@uoguelph.ca

# Date: 2023-05-19
# --------------------------------------------------------------------------------------
# Arun defined function converted into Python
# --------------------------------------------------------------------------------------

import math 
import pandas as pd
import numpy as np
import set_parameters as sp


def mol_weight_calc(numCarbon, numDoubleBonds):
    mwCarbon = 12.0107
    mwHydrogen = 1.0079
    mwOxygen = 15.9994

    head = 3 * mwCarbon + 5 * mwHydrogen
    ester_link = 6 * mwOxygen + 3 * mwCarbon
    body = (numCarbon - 6) * (mwCarbon + 2 * mwHydrogen) - \
        2 * mwHydrogen * numDoubleBonds
    end = 3 * (mwCarbon + 3 * mwHydrogen)

    molWeight = head + ester_link + body + end
    return molWeight


def fOdd_func(num):
    if num % 2 == 0:
        x = 0
    else:
        x = 1
    return x


def fxy_func(x, xo, kx, y, ky):
    z = 2 - math.exp(-1 * ((x - xo) / kx) ** 2) - math.exp(-(y / ky) ** 2)
    return z


def enthalpy_fusion_unsat(hfsat, hol, nol, hel, nel, hli, nli):
    Hf = hfsat + hol * nol + hel * nel + hli * nli
    return Hf


def enthalpy_fusion_sat(nc, h, ho, hxy, fxy, hodd, fodd):
    Hf = (h * nc) + ho + (hxy * fxy) + hodd * fodd
    return Hf


def entropy_fusion_sat(nc, s, so, sxy, fxy, sodd, fodd, fasym):
    R = 8.314

    Ef = (s * nc) + so + (sxy * fxy) + \
        (sodd * fodd) + (R * math.log(2) * fasym)
    return Ef


def melting_temp(nc, Tinf, A, B):
    Tf = Tinf * (1 + A / nc - (A * B) / (nc ** 2))
    return Tf


def SatA(x, y, Ao, Aodd, fodd, Ax, Ax2, Axy, Ay, Ay2):
    A = Ao + Aodd * fodd + Ax * x + Ax2 * \
        x ** 2 + Axy * x * y + Ay * y + Ay2 * y ** 2
    return A

# ------------------------------------------------------------------------------------
# Additional Sebastian defined functions
# ------------------------------------------------------------------------------------

def set_polymorph_col(polymorph):
    if polymorph == "alpha":
        return 1

    if polymorph == "beta prime":
        return 2

    if polymorph == "beta":
        return 3
    
def fasym_func(y, symbol1, symbol3):
    if y == 0 and symbol1 == symbol3:
        fasym = 0
    else:
        fasym = 1

    return fasym

    
# ------------------------------------------------------------------------------------
# CODE START
# ------------------------------------------------------------------------------------

def compute(sn1, sn2, sn3, polymorph, param_type): # add param for type - if type is ref set vars to that else set to other

    # print("Running computation...\n")

    fattyAcids = pd.read_csv("data/asmDB-FattyAcidsParameters.csv")

    # input values
    # sn1 = "Palmitic"
    # sn2 = "Palmitic"
    # sn3 = "Palmitic"
    # polymorph = "alpha"

    parseCol = set_polymorph_col(polymorph)

    # call all our set functions here

    if param_type == "reference":
        h, ho, hxy, s, so, sxy, xo, k, temp_inf_a = sp.sat_enthalpy_params("reference", parseCol)

        Ao, Aodd,  Ax, Ax2, Axy, Ay, Ay2, Bo, Bodd, Bx, Bx2, Bxy, By, By2, temp_inf_b = sp.sat_temp_params(
            "reference", parseCol)
        
        hol, hel, hli = sp.unsat_enthalpy_params("reference", parseCol)

        aO, aD, aJ, aN, aOO, aDD, aJJ, aNN, aOJ, aON, aJN, bO, bJ, bN = sp.unsat_temp_params(
            "reference", parseCol)
        
    elif param_type == "P1L":
        h, ho, hxy, s, so, sxy, xo, k, temp_inf_a = sp.sat_enthalpy_params(
            "P1L", parseCol)

        Ao, Aodd,  Ax, Ax2, Axy, Ay, Ay2, Bo, Bodd, Bx, Bx2, Bxy, By, By2, temp_inf_b = sp.sat_temp_params(
            "P1L", parseCol)

        hol, hel, hli = sp.unsat_enthalpy_params("P1L", parseCol)

        aO, aD, aJ, aN, aOO, aDD, aJJ, aNN, aOJ, aON, aJN, bO, bJ, bN = sp.unsat_temp_params(
            "P1L", parseCol)

    elif param_type in {"PI_L", "PII_L", "PI_C", "PII_C"}:
        # Seilert et al. (2021) Table 2 parameter sets (thermodynamic route).
        h, ho, hxy, s, so, sxy, xo, k, temp_inf_a = sp.sat_enthalpy_params(param_type, parseCol)

        # Polynomial temperature-route parameters are not provided by Seilert; we
        # fall back to reference for those so the code path remains functional.
        Ao, Aodd,  Ax, Ax2, Axy, Ay, Ay2, Bo, Bodd, Bx, Bx2, Bxy, By, By2, temp_inf_b = sp.sat_temp_params(
            param_type, parseCol)

        hol, hel, hli = sp.unsat_enthalpy_params(param_type, parseCol)
        aO, aD, aJ, aN, aOO, aDD, aJJ, aNN, aOJ, aON, aJN, bO, bJ, bN = sp.unsat_temp_params(
            param_type, parseCol)
        
    if len(sn1) == 1 or len(sn2) == 1 or len(sn3) == 1:
        symbol1 = sn1
        symbol2 = sn2
        symbol3 = sn3
    else:
        symbol1 = fattyAcids.loc[fattyAcids["Name"].values == sn1, "Symbol"].values
        symbol2 = fattyAcids.loc[fattyAcids["Name"].values == sn2, "Symbol"].values
        symbol3 = fattyAcids.loc[fattyAcids["Name"].values == sn3, "Symbol"].values

    numCarbon1 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol1, "nc"].values[0]
    numCarbon2 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol2, "nc"].values[0]
    numCarbon3 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol3, "nc"].values[0]

    numDb1 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol1, "nd"].values[0]
    numDb2 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol2, "nd"].values[0]
    numDb3 = fattyAcids.loc[fattyAcids["Symbol"].values == symbol3, "nd"].values[0]

    totalCarbon = numCarbon1 + numCarbon2 + numCarbon3
    totalDoubleBond = numDb1 + numDb2 + numDb3

    p = min(numCarbon1, numCarbon3)
    q = numCarbon2
    r = max(numCarbon1, numCarbon3)

    x = q - p
    y = r - p

    sat_enthalpy_vars = pd.read_csv("data/asmDB-SatAcidEnthalpyCoefficientEstimates.csv")

    experimental_values = pd.read_csv("data/asmExperimentalThermalPropertyPureTAGDatabase.csv")

    tag = symbol1[0] + symbol2[0] + symbol3[0]
    inverse_tag = symbol3[0] + symbol2[0] + symbol1[0]

    mol_weight_sample = mol_weight_calc(totalCarbon, totalDoubleBond);
    
    check_oddness = fOdd_func(numCarbon1) + fOdd_func(numCarbon2) + fOdd_func(numCarbon3)

    fasym = fasym_func(y, symbol1, symbol3)

    if check_oddness > 0:
        fodd = 1
    else:
        fodd = 0

    fxy = fxy_func(x, xo, k, y, k)

    # set beta value for special case
    if polymorph == "beta":
        isBeta = 1
    else:
        isBeta = 0

    if isBeta == 1:
        hodd = sat_enthalpy_vars.iloc[9, parseCol]
    else:
        hodd = 0

    sat_enthalpy_sample = enthalpy_fusion_sat(totalCarbon, h, ho, hxy, fxy, hodd, fodd)
    sat_entropy_sample = entropy_fusion_sat(totalCarbon, s, so, sxy, fxy, hodd, fodd, fasym * isBeta)

    nO = 0  # number of oleic chains in the - symbol O
    nE = 0  # number of elaidic chains in the tag - symbol - E
    nJ  = 0  # number of linoleic chains in the tag - symbol - J
    nN = 0  # number of linolenic chains in the tag - symbol - N


    if totalDoubleBond >= 1:

        # Fatty Acid 1
        if symbol1 == "O":
            nO += 1
        elif symbol1 == "D":
            nE += 1
        elif symbol1 == "J":
            nJ += 1
        elif symbol1 == "N":
            nN += 1

        # Fatty Acid 2
        if symbol2 == "O":
            nO += 1
        elif symbol2 == "D":
            nE += 1
        elif symbol2 == "J":
            nJ += 1
        elif symbol2 == "N":
            nN += 1

        # Fatty Acid 3
        if symbol3 == "O":
            nO += 1
        elif symbol3 == "D":
            nE += 1
        elif symbol3 == "J":
            nJ += 1
        elif symbol3 == "N":
            nN += 1

        hf = sat_enthalpy_sample
        sat_enthalpy_sample = enthalpy_fusion_unsat(hf, hol, nO, hel, nE, hli, nJ)
        

    nOO = max(0, nO - 1)
    nDD = 0
    nJJ = max(0, nJ - 1)
    nNN = max(0, nN - 1)

    nOJ = nO * nJ
    nON = nO * nN
    nJN = nJ * nN

    A0 = ho + hxy * fxy #ho_prime
    B0 = so + sxy * fxy + 8.314 * math.log(2) * fasym * isBeta #so_prime

    A1a = A0/h - B0/s
    B1a = B0/s

    A1b = SatA(x, y, Ao, Aodd, fodd, Ax, Ax2, Axy, Ay, Ay2) # As_a
    B1b = SatA(x, y, Bo, Bodd, fodd, Bx, Bx2, Bxy, By, By2) #As_b

    Au_modelA = (A1a + aO*nO + aD*nE + aJ*nJ + aN*nN + 
            aOO*nOO + aDD*nDD + aJJ*nJJ + aNN*nNN+
            aOJ*nOJ + aON*nON + aJN*nJN)

    Au_modelB = (A1b + aO*nO + aD*nE + aJ*nJ + aN*nN + 
            aOO*nOO + aDD*nDD + aJJ*nJJ + aNN*nNN+
            aOJ*nOJ + aON*nON + aJN*nJN)

    Bu_modelA = B1a + bO*nO + bN*nN + bJ*nJ
    Bu_modelB = B1b + bO*nO + bN*nN + bJ*nJ

    temp_sample_a = melting_temp(totalCarbon, temp_inf_a, Au_modelA, Bu_modelA) - 273.16
    temp_sample_b = melting_temp(totalCarbon, temp_inf_b, Au_modelB, Bu_modelB) - 273.16

    nCarbons = totalCarbon + 3
    nOxygen = 6
    nHydrogen = 2 * (totalCarbon - 6) + 14 - 2 * totalDoubleBond

    chemical_formula = f"C{nCarbons}-O{nOxygen}-H{nHydrogen}"

    tag_row_num = np.where(experimental_values.iloc[:, 0] == tag)[0]
    inverse_tag_row_num = np.where(experimental_values.iloc[:, 0] == inverse_tag)[0]

    if (polymorph == "alpha"):
        exp_i = 1
    elif (polymorph == "beta prime"):
        exp_i = 3
    else:
        exp_i = 5

    if len(tag_row_num) > 0:
        # exp_temp_FOM = experimental_values.iloc[tag_row_num, exp_i + 7].item()
        # exp_enthalpy_FOM = experimental_values.iloc[tag_row_num, exp_i + 1].item()

        try:
            exp_enthalpy = experimental_values.iloc[tag_row_num, exp_i].item()
            if math.isnan(exp_enthalpy):
                exp_enthalpy = "NA"
        except (ValueError, IndexError):
            exp_enthalpy = "NA"

        try:
            exp_temp = experimental_values.iloc[tag_row_num, exp_i + 6].item()
            if math.isnan(exp_temp):
                exp_temp = "NA"
        except (ValueError, IndexError):
            exp_temp = "NA"

    else:
        # exp_enthalpy_FOM = experimental_values.iloc[inverse_tag_row_num, exp_i + 1].item()
        # exp_temp_FOM = experimental_values.iloc[inverse_tag_row_num, exp_i + 7].item()
        
        try:
            exp_enthalpy = experimental_values.iloc[inverse_tag_row_num, exp_i].item()
            if math.isnan(exp_enthalpy):
                exp_enthalpy = "NA"
        except (ValueError, IndexError):
            exp_enthalpy = "NA"

        try:
            exp_temp = experimental_values.iloc[inverse_tag_row_num, exp_i + 6].item()
            if math.isnan(exp_temp):
                exp_temp = "NA"
        except (ValueError, IndexError):
            exp_temp = "NA"

    # --------------------------------------------------------------------------------
    # GC METHOD
    # not sure if needed but why not...
    # --------------------------------------------------------------------------------

    HIJalpha = pd.read_csv("data/GC-data/HIJ-alpha.csv")
    HIJbprime = pd.read_csv("data/GC-data/HIJ-bprime.csv")
    HIJbeta = pd.read_csv("data/GC-data/HIJ-beta.csv")
    KH = pd.read_csv("data/GC-data/KH.csv")

    TIJalpha = pd.read_csv("data/GC-data/TIJ-alpha.csv")
    TIJbprime = pd.read_csv("data/GC-data/TIJ-bprime.csv")
    TIJbeta = pd.read_csv("data/GC-data/TIJ-beta.csv")
    KT = pd.read_csv("data/GC-data/KT.csv")

    if symbol1 == symbol2 and symbol1 == symbol3 and symbol2 == symbol3:
        type = 0    
    if symbol1 == symbol2 and symbol1 != symbol3 and symbol2 != symbol3:
        type = 1
    if symbol1 != symbol2 and symbol1 != symbol3 and symbol2 == symbol3:
        type = 1
    if symbol1 != symbol2 and symbol1 == symbol3 and symbol2 != symbol3:
        type = 2
    if symbol1 != symbol2 and symbol1 != symbol3 and symbol2 != symbol3:
        type = 3

    acids = HIJbeta.columns.tolist()

    if symbol1 not in acids or symbol2 not in acids or symbol3 not in acids:
        y1 = None
        x1 = None
        y2 = None
    else:
        y1 = acids.index(symbol1)
        x1 = acids.index(symbol2)
        y2 = acids.index(symbol3)

    if y1 is not None and y2 is not None and x1 is not None:

        if polymorph == "beta": 
            contributionMatrix = HIJbeta
            contributionMatrixT = TIJbeta
            Kcol = 3
        elif polymorph == "betaP":
            contributionMatrix = HIJbprime
            contributionMatrixT = TIJbprime
            Kcol = 2
        else:
            contributionMatrix = HIJalpha
            contributionMatrixT = TIJalpha
            Kcol = 1

        cont1 = contributionMatrix.iloc[x1, y1].item()
        cont2 = contributionMatrix.iloc[x1, y2].item()

        hfGCM = KH.iloc[type, Kcol].item() * (cont1 + cont2)

        Tcont1 = contributionMatrixT.iloc[x1, y1].item()
        Tcont2 = contributionMatrixT.iloc[x1, y2].item()

        tfGCM = KT.iloc[type, Kcol] * (Tcont1 + Tcont2) - 273.15

    else:
        hfGCM = None
        tfGCM = None

    mol_weight_sample = round(mol_weight_sample, 1)

    sat_enthalpy_sample = round(sat_enthalpy_sample, 1)

    temp_sample_a = round(temp_sample_a, 1)
    temp_sample_b = round(temp_sample_b, 1)

    if hfGCM is None or math.isnan(hfGCM):
        hfGCM = "NA"
    else:
        hfGCM = round(hfGCM, 1)

    if tfGCM is None or math.isnan(tfGCM):
        tfGCM = "NA"
    else:
        tfGCM = round(tfGCM, 1)

    return polymorph, tag, chemical_formula, mol_weight_sample, exp_enthalpy, exp_temp, sat_enthalpy_sample, temp_sample_a, temp_sample_b, hfGCM, tfGCM


# Hf = sat_enthalpy_sample
# Tf modelA = temp_sample_a / Tf modelB = temp_sample_b

