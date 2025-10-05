
import math
import pandas as pd
import numpy as np

# --- UBP-inspired constants and functions ---
def apply_ubp_operator(base_value, operator_type, C_rate, M_constant=math.e):
    if operator_type == 'linear':
        return base_value * (1 + C_rate / 100.0)
    elif operator_type == 'quadratic':
        return base_value * (1 + (C_rate / 100.0)**2)
    elif operator_type == 'compositional':
        return base_value * (1 + (M_constant * C_rate / 1000.0))
    else:
        return base_value

# --- Study 9: Coupled UBP Operators in Multi-step Reversible Mechanism with Dynamic Temperature ---

# 1. Initialization:
# Mechanism: A <=> I <=> P
# Step 1: A <=> I (kf1, kr1)
# Step 2: I <=> P (kf2, kr2)

A_initial = 1.0  # Initial concentration of reactant A in mol/L
I_initial = 0.0  # Initial concentration of intermediate I in mol/L
P_initial = 0.0  # Initial concentration of product P in mol/L

Time_Step_dt = 0.1  # Time step duration in seconds
Num_Steps = 100  # Number of simulation steps

Gas_Constant_R_val = 8.314  # J/(mol*K)

# Arrhenius parameters for Step 1 (A <=> I)
Pre_exponential_Factor_A_f1 = 1.0e10  # s^-1
Activation_Energy_Ea_f1 = 60000.0  # J/mol
Pre_exponential_Factor_A_r1 = 1.0e9   # s^-1
Activation_Energy_Ea_r1 = 70000.0   # J/mol
Enthalpy_of_Reaction_dH1 = -20000.0  # J/mol (A -> I, exothermic)

# Arrhenius parameters for Step 2 (I <=> P)
Pre_exponential_Factor_A_f2 = 5.0e9   # s^-1
Activation_Energy_Ea_f2 = 55000.0   # J/mol
Pre_exponential_Factor_A_r2 = 1.0e8   # s^-1
Activation_Energy_Ea_r2 = 65000.0   # J/mol
Enthalpy_of_Reaction_dH2 = -30000.0  # J/mol (I -> P, exothermic)

# Dynamic Temperature Parameters
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K)
Density_solution = 1000.0  # g/L
Volume_L = 1.0 # Volume of the reactor in Liters
Mass_solution = Volume_L * Density_solution # g

# UBP Parameters for kf1 (e.g., quadratic operator)
UBP_OPERATOR_TYPE_KF1 = 'quadratic'
UBP_C_RATE_KF1 = 10.0
UBP_M_CONSTANT_KF1 = math.pi

# UBP Parameters for kr1 (e.g., linear operator)
UBP_OPERATOR_TYPE_KR1 = 'linear'
UBP_C_RATE_KR1 = 5.0
UBP_M_CONSTANT_KR1 = math.e

# UBP Parameters for kf2 (e.g., compositional operator)
UBP_OPERATOR_TYPE_KF2 = 'compositional'
UBP_C_RATE_KF2 = 15.0
UBP_M_CONSTANT_KF2 = math.e

# UBP Parameters for kr2 (no operator for now, or another type)
UBP_OPERATOR_TYPE_KR2 = None # No UBP operator for kr2
UBP_C_RATE_KR2 = 0.0
UBP_M_CONSTANT_KR2 = math.e

Current_Time = 0.0
Current_Temperature_K = Initial_Temperature_K
Current_Concentration_A = A_initial
Current_Concentration_I = I_initial
Current_Concentration_P = P_initial

Results_Table = []

# Add initial state (T=0)
initial_kf1_base = Pre_exponential_Factor_A_f1 * math.exp(-Activation_Energy_Ea_f1 / (Gas_Constant_R_val * Initial_Temperature_K))
initial_kr1_base = Pre_exponential_Factor_A_r1 * math.exp(-Activation_Energy_Ea_r1 / (Gas_Constant_R_val * Initial_Temperature_K))
initial_kf2_base = Pre_exponential_Factor_A_f2 * math.exp(-Activation_Energy_Ea_f2 / (Gas_Constant_R_val * Initial_Temperature_K))
initial_kr2_base = Pre_exponential_Factor_A_r2 * math.exp(-Activation_Energy_Ea_r2 / (Gas_Constant_R_val * Initial_Temperature_K))

initial_kf1 = apply_ubp_operator(initial_kf1_base, UBP_OPERATOR_TYPE_KF1, UBP_C_RATE_KF1, UBP_M_CONSTANT_KF1)
initial_kr1 = apply_ubp_operator(initial_kr1_base, UBP_OPERATOR_TYPE_KR1, UBP_C_RATE_KR1, UBP_M_CONSTANT_KR1)
initial_kf2 = apply_ubp_operator(initial_kf2_base, UBP_OPERATOR_TYPE_KF2, UBP_C_RATE_KF2, UBP_M_CONSTANT_KF2)
initial_kr2 = apply_ubp_operator(initial_kr2_base, UBP_OPERATOR_TYPE_KR2, UBP_C_RATE_KR2, UBP_M_CONSTANT_KR2)

Results_Table.append((Current_Time, Current_Concentration_A, Current_Concentration_I, Current_Concentration_P, Current_Temperature_K, initial_kf1, initial_kr1, initial_kf2, initial_kr2))

# 2. Simulation Loop (using numerical integration):
for i in range(1, Num_Steps + 1):
    # Calculate current base rate constants using Arrhenius equation with current temperature
    kf1_base = Pre_exponential_Factor_A_f1 * math.exp(-Activation_Energy_Ea_f1 / (Gas_Constant_R_val * Current_Temperature_K))
    kr1_base = Pre_exponential_Factor_A_r1 * math.exp(-Activation_Energy_Ea_r1 / (Gas_Constant_R_val * Current_Temperature_K))
    kf2_base = Pre_exponential_Factor_A_f2 * math.exp(-Activation_Energy_Ea_f2 / (Gas_Constant_R_val * Current_Temperature_K))
    kr2_base = Pre_exponential_Factor_A_r2 * math.exp(-Activation_Energy_Ea_r2 / (Gas_Constant_R_val * Current_Temperature_K))

    # Apply UBP operators to respective rate constants
    kf1 = apply_ubp_operator(kf1_base, UBP_OPERATOR_TYPE_KF1, UBP_C_RATE_KF1, UBP_M_CONSTANT_KF1)
    kr1 = apply_ubp_operator(kr1_base, UBP_OPERATOR_TYPE_KR1, UBP_C_RATE_KR1, UBP_M_CONSTANT_KR1)
    kf2 = apply_ubp_operator(kf2_base, UBP_OPERATOR_TYPE_KF2, UBP_C_RATE_KF2, UBP_M_CONSTANT_KF2)
    kr2 = apply_ubp_operator(kr2_base, UBP_OPERATOR_TYPE_KR2, UBP_C_RATE_KR2, UBP_M_CONSTANT_KR2)

    # Calculate rates for each step
    rate_f1 = kf1 * Current_Concentration_A
    rate_r1 = kr1 * Current_Concentration_I
    rate_f2 = kf2 * Current_Concentration_I
    rate_r2 = kr2 * Current_Concentration_P

    # Calculate net change in concentrations (using Euler method)
    # d[A]/dt = -rate_f1 + rate_r1
    # d[I]/dt = rate_f1 - rate_r1 - rate_f2 + rate_r2
    # d[P]/dt = rate_f2 - rate_r2

    delta_A = (-rate_f1 + rate_r1) * Time_Step_dt
    delta_I = (rate_f1 - rate_r1 - rate_f2 + rate_r2) * Time_Step_dt
    delta_P = (rate_f2 - rate_r2) * Time_Step_dt

    # Update concentrations (pre-check for negative values)
    next_A = Current_Concentration_A + delta_A
    next_I = Current_Concentration_I + delta_I
    next_P = Current_Concentration_P + delta_P

    # Simple check to prevent negative concentrations and conserve total mass
    # This is a simplified approach; more robust methods exist for stiff ODEs
    if next_A < 0: delta_A = -Current_Concentration_A
    if next_I < 0: delta_I = -Current_Concentration_I
    if next_P < 0: delta_P = -Current_Concentration_P

    # Recalculate based on adjusted deltas to maintain mass balance if one goes negative
    # This part can be complex for multi-step, multi-species systems. For simplicity, we'll just cap at 0.
    Current_Concentration_A = max(0.0, Current_Concentration_A + delta_A)
    Current_Concentration_I = max(0.0, Current_Concentration_I + delta_I)
    Current_Concentration_P = max(0.0, Current_Concentration_P + delta_P)

    # Moles converted for each step for energy balance
    moles_converted_step1_net = (rate_f1 - rate_r1) * Volume_L * Time_Step_dt
    moles_converted_step2_net = (rate_f2 - rate_r2) * Volume_L * Time_Step_dt

    # Heat change due to both reactions
    heat_change = (moles_converted_step1_net * Enthalpy_of_Reaction_dH1) + \
                  (moles_converted_step2_net * Enthalpy_of_Reaction_dH2)

    # Temperature change
    delta_T = heat_change / (Mass_solution * Specific_Heat_Capacity_solution)
    Current_Temperature_K += delta_T

    # Record the result
    Current_Time = i * Time_Step_dt
    Results_Table.append((Current_Time, Current_Concentration_A, Current_Concentration_I, Current_Concentration_P, Current_Temperature_K, kf1, kr1, kf2, kr2))

# Convert to DataFrame for better presentation
df_study9 = pd.DataFrame(Results_Table, columns=[
    'Time (s)', 'A Concentration (mol/L)', 'I Concentration (mol/L)', 'P Concentration (mol/L)',
    'Temperature (K)', 'kf1 (s^-1)', 'kr1 (s^-1)', 'kf2 (s^-1)', 'kr2 (s^-1)'
])

# 3. Output Generation:
print("--- Study 9 Simulation Results (Coupled UBP Operators in Multi-step Reversible Mechanism) ---")
print(f"Initial Temperature: {Initial_Temperature_K} K")
print(f"UBP Operator on kf1: {UBP_OPERATOR_TYPE_KF1} (C_rate={UBP_C_RATE_KF1}, M_constant={UBP_M_CONSTANT_KF1})")
print(f"UBP Operator on kr1: {UBP_OPERATOR_TYPE_KR1} (C_rate={UBP_C_RATE_KR1}, M_constant={UBP_M_CONSTANT_KR1})")
print(f"UBP Operator on kf2: {UBP_OPERATOR_TYPE_KF2} (C_rate={UBP_C_RATE_KF2}, M_constant={UBP_M_CONSTANT_KF2})")
print(f"UBP Operator on kr2: {UBP_OPERATOR_TYPE_KR2 if UBP_OPERATOR_TYPE_KR2 else 'None'}")
print(df_study9.to_string(index=False))

# Save results to CSV
output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study9_coupled_ubp_operators_results.csv"
df_study9.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

