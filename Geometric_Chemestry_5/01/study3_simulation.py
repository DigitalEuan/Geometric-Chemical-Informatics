
import math
import numpy as np

# --- UBP-inspired constants and functions ---
# From Reframing_EMC.txt, reinterpreting E=mc^2 as a computational principle
# E -> Time as substrate
# = -> the result of
# M -> a Constant (e.g., pi, e, phi, sqrt(2))
# x -> the Operator of Amplification (Iteration, Faster iteration, Composition, Nesting, Parallelization)
# C -> maximum rate of iteration
# ^2 -> amplification of convergence

def apply_ubp_operator(base_value, operator_type, C_rate, M_constant=math.e):
    """
    Applies a UBP-inspired 'Operator of Amplification' to a base value.
    In this context, we'll use it to modify the rate constant 'k'.

    Args:
        base_value (float): The value to be modified (e.g., the Arrhenius rate constant).
        operator_type (str): Type of UBP operator ('linear', 'quadratic', 'compositional').
        C_rate (float): Analogous to 'maximum rate of iteration' or a scaling factor.
        M_constant (float): A constant, like 'e' or 'pi', acting as a base for amplification.

    Returns:
        float: The modified value.
    """
    if operator_type == 'linear':
        # A simple linear amplification: base_value * (1 + C_rate / 100)
        return base_value * (1 + C_rate / 100.0) # Small linear boost
    elif operator_type == 'quadratic':
        # Quadratic amplification: base_value * (1 + (C_rate / 100)^2)
        return base_value * (1 + (C_rate / 100.0)**2) # Quadratic boost
    elif operator_type == 'compositional':
        # Reflects a slower, relational form of multiplication, converging robustly at scale.
        # Let's model this as a dampened exponential effect based on M_constant and C_rate.
        # This is a conceptual interpretation, aiming to show a different kind of amplification.
        return base_value * (1 + (M_constant * C_rate / 1000.0)) # Example: M_constant * C_rate with a dampening factor
    else:
        return base_value # No UBP operator applied

# 1. Initialization (Study 3 - with Arrhenius Equation and UBP Operators):
# Numerical Values from Study 1
R_initial = 100.0  # Initial concentration in units
Time_Step_dt = 1.0  # Time step duration in seconds
Num_Steps = 10  # Number of simulation steps

# Variables for Arrhenius Equation (adjusted from Study 2 for illustrative purposes)
Pre_exponential_Factor_A = 1.0e5  # Adjusted from 1.0e10 to 1.0e5
Activation_Energy_Ea = 30000.0  # Adjusted from 50000.0 to 30000.0 Joules/mol
Gas_Constant_R = 8.314  # J/(mol*K)
Temperature_K = 298.15  # Kelvin (25 degrees Celsius)

# UBP Parameters
UBP_OPERATOR_TYPE = 'quadratic' # Can be 'linear', 'quadratic', 'compositional', or None
UBP_C_RATE = 10.0 # Analogous to 'maximum rate of iteration' or scaling factor
UBP_M_CONSTANT = math.pi # Using pi as an example constant

# Calculate the base rate constant k using the Arrhenius equation
# k = A * exp(-Ea / (R * T))
base_Rate_Constant_k = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))

# Apply UBP operator to modify the rate constant
Rate_Constant_k_modified = apply_ubp_operator(base_Rate_Constant_k, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)

Current_Time = 0.0
Results_Table = []  # List to store (Time, Concentration) tuples

# Add initial state (T=0)
Results_Table.append((Current_Time, R_initial))

# 2. Simulation Loop (using analytical solution for first-order kinetics):
for i in range(1, Num_Steps + 1):
    # Calculate current time
    Current_Time = i * Time_Step_dt

    # Calculate concentration using the analytical solution with the modified rate constant
    # R(T) = R_initial * exp(-Rate_Constant_k_modified * Current_Time)
    Concentration_R = R_initial * math.exp(-Rate_Constant_k_modified * Current_Time)

    # Record the result
    Results_Table.append((Current_Time, Concentration_R))

# 3. Output Generation:
print("--- Study 3 Simulation Results (with Arrhenius Equation and UBP Operator) ---")
print(f"Temperature: {Temperature_K} K")
print(f"Base Rate Constant (k_base): {base_Rate_Constant_k:.4e} s^-1")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(f"Modified Rate Constant (k_modified): {Rate_Constant_k_modified:.4e} s^-1")
print("\nTime (s) | Concentration (units)")
print("---------|----------------------")
for Time, Concentration in Results_Table:
    print(f"{Time:<9.2f}| {Concentration:<20.4f}")

