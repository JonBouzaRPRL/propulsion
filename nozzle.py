import numpy as np
import matplotlib.pyplot as plt
import math

Gamma = 1.4

#Temperature Ratio Function
def TempRatio(Gamma, MachNum):
    """
    Calculate the temperature ratio for the flow

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

t_ratioM2 = TempRatio(Gamma, 2.0)
print(f"Temperature ratio T/Tstag for Mach 2.0: {t_ratioM2:.4f}")
t_ratioM05 = TempRatio(Gamma, 0.5)
print(f"Temperature ratio T/Tstag for Mach 0.5: {t_ratioM05:.4f}")
t_RatioM1 = TempRatio(Gamma, 1.0)
print(f"Temperature ratio T/Tstag for Mach 1.0: {t_RatioM1:.4f}")  
t_RatioM0 = TempRatio(Gamma, 0.0)
print(f"Temperature ratio T/Tstag for Mach 0.0: {t_RatioM0:.4f}")

# Pressure ratio function

def PressureRatio(Gamma, t_Tstag):
    """
    Calculate the pressure ratio for the flow

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

t = TempRatio(Gamma, 2.0)
pressure_ratio = PressureRatio(Gamma, t)
print(f"Pressure ratio P/Pstag for Mach 2.0: {pressure_ratio:.4f}")
t = TempRatio(Gamma, 1.0)
pressure_ratio = PressureRatio(Gamma, t)
print(f"Pressure ratio P/Pstag for Mach 1.0: {pressure_ratio:.4f}")
t = TempRatio(Gamma, 0.0)
pressure_ratio = PressureRatio(Gamma, t)
print(f"Pressure ratio P/Pstag for Mach 0.0: {pressure_ratio:.4f}")