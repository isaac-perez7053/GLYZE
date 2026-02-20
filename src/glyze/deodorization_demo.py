import scipy


def T_1mm_vapor_pressure(numCarbons):
    Temperature = -400 + 161 * scipy.log(numCarbons)  # K
    return Temperature


def enthalpy_of_vaporization(numCarbons):
    enthalpy = 3.18 * numCarbons + 50.5  # kJ/mol
    return enthalpy


def vapor_pressure(numCarbons, Temperature):
    T_1mm = T_1mm_vapor_pressure(numCarbons)
    delta_H_vap = enthalpy_of_vaporization(numCarbons)
    R = 8.314e-3  # kJ/mol-K
    vapor_pressure = scipy.exp(
        -delta_H_vap / R * (1 / Temperature - 1 / T_1mm) + scipy.log(T_1mm)
    )
    return vapor_pressure


def evaporation_rate(x, molar_mass, numCarbons, Temperature):
    Pvap = vapor_pressure(numCarbons, Temperature)  # mmHg
    N_A = 6.022e23  # molecules/mol
    R = 8.314  # J/mol-K
    gamma = 1.0  # activity coefficient
    EVRT = (
        N_A * gamma * x * Pvap * scipy.sqrt(2 * scipy.pi * R * molar_mass * Temperature)
    )  # molecules/m2-s
    return EVRT


def real_evaporation_rate(x, molar_mass, numCarbons, Temperature, efficiency_factor):
    ideal_rate = evaporation_rate(x, molar_mass, numCarbons, Temperature)
    real_rate = (
        ideal_rate * efficiency_factor
    )  # efficiency factor to account for real-world conditions
    return real_rate
