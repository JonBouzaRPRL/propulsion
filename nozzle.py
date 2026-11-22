"""Propulsion analysis functions for nozzle flow. In this calculator we assume the flow is isentropic and the working fluid is air. 
The functions calculate the temperature, pressure, and density ratios for a given Mach number using the isentropic flow relations. 

that is:

the gas follows the ideal calorically perfect assumption (constant gamma = 1.4)

and 

we assume steady state and 1D flow.

"""

import numpy as np
import matplotlib.pyplot as plt
import math

Gamma = 1.4 # assuming air as the working fluid, the ratio of specific heats (Cp/Cv) is approximately 1.4 for diatomic gases like air at standard conditions.

M_Known = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] # Mach numbers for which we want to calculate the ratios
T_TstagExpecteds= [1.0000, 0.9524, 0.8333, 0.6897, 0.5556, 0.4444, 0.3571, 0.2381, 0.1667] # Temperature ratios for the corresponding Mach numbers
P_PstagExpecteds = [1.0000, 0.8430, 0.5283, 0.2724, 0.1278, 0.05853, 0.02722, 0.006586, 0.001890] # Pressure ratios for the corresponding Mach numbers
rho_rhostagExpecteds = [1.0000, 0.8852, 0.6339, 0.3950, 0.2300, 0.1317, 0.07623, 0.02766, 0.01134]  # Density Ratios of corresponding mach numbers

def TempRatio(Gamma, MachNum):
    """
    Calculate the temperature ratio for the flow utilizing the following equation:

    T/Tstag = 1 / (1 + ((Gamma - 1) / 2) * MachNum^2)

    Parameters:
    Gamma (float): Ratio of specific heats (Cp/Cv) for the gas.
    MachNum (float): Mach number of the flow speed.

    Returns:
    float: Temperature ratio T/Tstag for the flow
    """
    if MachNum < 0:
        raise ValueError("Mach number must be 0 or higher for flow.")
    
    T_Tstag = 1/((1+((Gamma-1)/2)*MachNum**2))
    
    return T_Tstag

# Pressure ratio function

def PressureRatio(Gamma, t_Tstag):
    """
    Calculate the pressure ratio for the flow utilizing the following equation:
    P/Pstag = (T/Tstag)^(Gamma/(Gamma-1))

    Parameters:
    Gamma (float): Ratio of specific heats (Cp/Cv) for the gas.
    t_Tstag (float): Temperature ratio T/Tstag for the flow.

    Returns:
    float: Pressure ratio P/Pstag for the flow
    """
    if t_Tstag <=0 or t_Tstag > 1:
        raise ValueError("Temperature ratio must be between 0 and 1 for flow.")
    
    P_Pstag = t_Tstag**(Gamma/(Gamma-1))
    
    return P_Pstag

def DensityRatio(Gamma, t_Tstag):
    """
    Calculate the density ratio for the flow utilizing the following equation:
    rho/rhostag = (T/Tstag)^(1/(Gamma-1))

    Parameters:
    Gamma (float): Ratio of specific heats (Cp/Cv) for the gas.
    t_Tstag (float): Temperature ratio T/Tstag for the flow.

    Returns:
    float: Density ratio rho/rhostag for the flow
    """
    if t_Tstag <=0 or t_Tstag > 1:
        raise ValueError("Temperature ratio must be between 0 and 1 for flow.")
    
    rho_rhostag = t_Tstag**(1/(Gamma-1))
    
    return rho_rhostag

def check(label, computed, expected, tolerance=1e-4):
    """validation check computations for known values of ratios.
    
    Parameters: 
    Label (str) for the check
    computed: The computed (float) value from the function
    expected: The expected (float) value for the check
    tolerance: The acceptable (float) difference between computed and expected values"""
    if abs(computed - expected) > tolerance:
        print(f"Check failed for {label}: computed {computed}, expected {expected}")
        return False
    return True

validation_checks = 0

if len(M_Known) != len(T_TstagExpecteds) or len(M_Known) != len(P_PstagExpecteds) or len(M_Known) != len(rho_rhostagExpecteds):
    raise ValueError("The lengths of the Mach number list and expected lists must be the same")
for i in range(len(M_Known)):
    t_ratio = TempRatio(Gamma, M_Known[i])
    p_ratio = PressureRatio(Gamma, t_ratio)
    rho_ratio = DensityRatio(Gamma, t_ratio)
 #   print(f"Temperature ratio T/Tstag for Mach {M_Known[i]}: {t_ratio:.4g}, Pressure ratio P/Pstag: {p_ratio:.4g}, Density ratio rho/rhostag: {rho_ratio:.4g}")
    if check(f"Temperature ratio for Mach {M_Known[i]}", t_ratio, T_TstagExpecteds[i]):
        validation_checks += 1
    if check(f"Pressure ratio for Mach {M_Known[i]}", p_ratio, P_PstagExpecteds[i]):
        validation_checks += 1
    if check(f"Density ratio for Mach {M_Known[i]}", rho_ratio, rho_rhostagExpecteds[i]):
        validation_checks += 1
print(f"Total validation checks passed: {validation_checks} out of {3*len(M_Known)}")
