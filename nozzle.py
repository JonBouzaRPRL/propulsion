"""Propulsion analysis functions for nozzle flow. In this calculator we assume the flow is isentropic and the working fluid is air. 
The functions calculate the temperature, pressure, and density ratios for a given Mach number using the isentropic flow relations. 

that is:

the gas follows the ideal calorically perfect assumption (constant gamma = 1.4)

and 

we assume steady state and 1D flow.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq as bq


# step 1-4 inputs: gamma and known values for the T/Tstag, P/Pstag, rho/rhostag checks
Gamma = 1.4 # assuming air as the working fluid, the ratio of specific heats (Cp/Cv) is approximately 1.4 for diatomic gases like air at standard conditions.
M_Known = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] # Mach numbers for which we want to calculate the ratios
T_TstagExpecteds= [1.0000, 0.9524, 0.8333, 0.6897, 0.5556, 0.4444, 0.3571, 0.2381, 0.1667] # Temperature ratios for the corresponding Mach numbers
P_PstagExpecteds = [1.0000, 0.8430, 0.5283, 0.2724, 0.1278, 0.05853, 0.02722, 0.006586, 0.001890] # Pressure ratios for the corresponding Mach numbers
rho_rhostagExpecteds = [1.0000, 0.8852, 0.6339, 0.3950, 0.2300, 0.1317, 0.07623, 0.02766, 0.01134]  # Density Ratios of corresponding mach numbers
# step 6 inputs: known A/A* values and their Mach numbers
area_RatioExpecteds = [1.3398, 1.000, 1.1762, 1.6875, 2.6367, 4.2346, 10.7188] # Area ratios for the corresponding Mach numbers
area_Mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] # Mach numbers for which we want to calculate the area ratios
# plot ranges: mach_Range is step 5, areaMach_Range is step 6
mach_Range = np.linspace(0, 5, 200) # mach number range for plotting ratios
areaMach_Range = np.linspace(0.1, 4, 200) # mach number range for plotting area ratio

# step 7 (subsonic) and step 8 (supersonic) round trip test Mach numbers
subsonic_mach_range = [0.1, 0.3, 0.5, 0.7, 0.9] # subsonic mach number test range 
supersonic_mach_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] # supersonic mach number test range 

#step 9 nozzle construction parameters

r_t = 0.025 # throat radius in meters
CR = 4 # construction ratio Inlet Area / Throat Area
eps = 10 # expansion ratio Exit Area / Throat Area
theta_c = np.radians(30) # converging half angle in degrees converted to radians
theta_d = np.radians(15) # divergence half angle in degrees converted to radians

# step 1: temperature ratio function (docstrings added in step 3, np.any for arrays added in step 5)
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
    if np.any(MachNum < 0):
        raise ValueError("Mach number must be 0 or higher for flow.")
    
    T_Tstag = 1/((1+((Gamma-1)/2)*MachNum**2))
    
    return T_Tstag

# step 2: pressure and density ratio functions

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
    if np.any(t_Tstag <=0) or np.any(t_Tstag > 1):
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
    if np.any(t_Tstag <=0) or np.any(t_Tstag > 1):
        raise ValueError("Temperature ratio must be between 0 and 1 for flow.")
    
    rho_rhostag = t_Tstag**(1/(Gamma-1))
    
    return rho_rhostag

# step 4: PASS/FAIL check helper
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


# step 6: area ratio A/A* vs Mach number
def area_Ratio(Gamma, MachNum):
    """
    Calculate the nozzle area and throat arearatio for the flow utilizing the following equation:
    A/A* = (1/MachNum) * ((2/(Gamma+1)) * (1 + ((Gamma-1)/2) * MachNum^2))^((Gamma+1)/(2*(Gamma-1)))

    Parameters:
    Gamma (float): Ratio of specific heats (Cp/Cv) for the gas.
    MachNum (float): Mach number of the flow speed.
    A = cross sectional area of the nozzle at a given point
    A* = cross sectional area of the nozzle at the throat

    Returns:
    float: Area ratio A/A* for the flow
    """
    if np.any(MachNum <= 0):
        raise ValueError("Mach number must be greater than 0 for area ratio calculation.")
    
    area_ratio = (1/MachNum) * ((2/(Gamma+1)) * (1 + ((Gamma-1)/2) * MachNum**2))**((Gamma+1)/(2*(Gamma-1)))
    
    return area_ratio


# step 7: Mach number from A/A* with brentq (subsonic); step 8 added the supersonic flag
def MRootFinder(Gamma, target_area_ratio, supersonic=False):
    """Finds mach number for given target area ratio using brentq root finding method:
    
    Parameters:
    Gamma (float): Ratio of specific heats (Cp/Cv) for the gas.
    target_area_ratio (float): The target area ratio A/A* for which we want to
    find the corresponding Mach number.
    A/A* (float): The area ratio A/A* for the flow.
        A = Cross Sectional Area of point of interest
        A* = Cross sectional area of throat
    Supersonic (bool) default False: If True, find the supersonic Mach number (M>=1). If False, find the subsonic Mach number (0<M<=1).

    returns:
    Mach Number (float): 
        If Supersonic False: The Subsonic (0<M<=1) Mach number corresponding to the given target area ratio. 
        If Supersonic True: The Supersonic (M>=1) Mach number corresponding to the given target area ratio.
    Raises:
    ValueError if A/A* < 1
    """
    def area_ratio_func(MachNum, target_area_ratio):
        return area_Ratio(Gamma, MachNum) - target_area_ratio
    
    if target_area_ratio < 1:
        raise ValueError("Target area ratio must be at least 1 for subsonic flow.")

    if supersonic:
        mach_number = bq(area_ratio_func, 1, 50, args=(target_area_ratio,))
    else:
        # Find the Mach number using brentq root finding method
        mach_number = bq(area_ratio_func, 1e-6, 1, args=(target_area_ratio,))

    return mach_number

# step 7 and step 8 example usage
print(f"Mach number for area ratio 1.3398: {MRootFinder(Gamma, 1.3398):.6f}")  # Example usage of MRootFinder (function)
print(f"Mach number for area ratio 1.6875: {MRootFinder(Gamma, 1.6875, supersonic=True):.6f}")  # Example usage of MRootFinder (function)


# validation counters: step 4, step 6, step 7, step 8 (in that order)
validation_checks = 0
validation_checks_Area = 0
validation_checks_SubMach = 0
validation_checks_SupersonicMach = 0

# step 4: T, P, rho ratio checks against known values
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

# step 6: area ratio checks against known values
if len(area_Mach) != len(area_RatioExpecteds):
    raise ValueError("The lengths of the area Mach number list and expected area ratio list must be the same")
for i in range(len(area_RatioExpecteds)):
    area_ratio = area_Ratio(Gamma, area_Mach[i])
    if check(f"Area ratio for Mach {area_Mach[i]}", area_ratio, area_RatioExpecteds[i], tolerance=1e-4):
        validation_checks_Area += 1
print(f"Total area ratio validation checks passed: {validation_checks_Area} out of {len(area_RatioExpecteds)}")

# step 7: subsonic round trip checks and known value checks
for i in range(len(subsonic_mach_range)):
    start_mach = subsonic_mach_range[i] # Passes subsonic mach list to variable
    forward_ratio = area_Ratio(Gamma, start_mach) # computes area ratio per pass from subsonic machs above
    back_mach = MRootFinder(Gamma, forward_ratio) # A/A* ratio then fed back in to validate mach number
    if check(f"Round trip Mach {start_mach}", back_mach, start_mach, tolerance=1e-6):
        validation_checks_SubMach += 1
if check(f"Mach for area ratio {area_RatioExpecteds[0]}", MRootFinder(Gamma, area_RatioExpecteds[0]), area_Mach[0], tolerance=1e-3):
    validation_checks_SubMach += 1
if check(f"Mach for area ratio {area_RatioExpecteds[1]}", MRootFinder(Gamma, area_RatioExpecteds[1]), area_Mach[1], tolerance=1e-6):
    validation_checks_SubMach += 1

# step 8: supersonic round trip checks
for i in range(len(supersonic_mach_range)):
    start_mach = supersonic_mach_range[i] # Passes supersonic mach list to variable
    forward_ratio = area_Ratio(Gamma, start_mach) # computes area ratio per pass from supersonic machs above
    back_mach = MRootFinder(Gamma, forward_ratio, supersonic=True) # A/A* ratio then fed back in to validate mach number
    if check(f"Round trip Mach {start_mach}", back_mach, start_mach, tolerance=1e-6):
        validation_checks_SupersonicMach += 1

print(f"Total Subsonic Mach number validation checks passed: {validation_checks_SubMach} out of {2 + len(subsonic_mach_range)}")
print(f"Total Supersonic Mach number validation checks passed: {validation_checks_SupersonicMach} out of {len(supersonic_mach_range)}")

# step 5 (T, P, rho) and step 6 (A/A*): arrays over the Mach ranges for plotting
TempRatios = TempRatio(Gamma, mach_Range)
PressureRatios = PressureRatio(Gamma, TempRatios)
DensityRatios = DensityRatio(Gamma, TempRatios)
area_ratios = area_Ratio(Gamma, areaMach_Range)

# step 9 nozzle construction

r_inlet = r_t * np.sqrt(CR) # radius inlet
Lc = (r_inlet - r_t)/ np.tan(theta_c)

r_exit = r_t * np.sqrt(eps) # radius exit
Ld = (r_exit - r_t) / np.tan(theta_d) # 

check("Lc value", Lc, 0.0433013, tolerance=1e-6)

check("Ld Value", Ld, 0.2017433, tolerance=1e-6)

# step 5 plot: isentropic flow ratios
plt.figure(figsize=(12, 8))
plt.title('Isentropic Flow Ratios vs Mach Number')
plt.xlabel('Mach Number')
plt.ylabel('Isentropic Flow Ratio')
plt.plot(mach_Range, TempRatios, label='Temperature Ratio T/Tstag', color='blue')
plt.plot(mach_Range, PressureRatios, label='Pressure Ratio P/Pstag', color='orange')
plt.plot(mach_Range, DensityRatios, label='Density Ratio rho/rhostag', color='green')
plt.legend()
plt.grid(True)
plt.savefig('isentropic_flow_ratios.png')

# step 6 plot: area ratio
plt.figure(figsize=(12, 8))
plt.title('Area Ratio vs Mach Number')
plt.xlabel('Mach Number')
plt.ylabel('Area Ratio A/A*')
plt.plot(areaMach_Range, area_ratios, label='Area Ratio A/A*', color='purple')
plt.legend()
plt.grid(True)
plt.savefig('area_ratio.png')

# keep plt.show() last: it pauses the script until the plot windows are closed
plt.show()