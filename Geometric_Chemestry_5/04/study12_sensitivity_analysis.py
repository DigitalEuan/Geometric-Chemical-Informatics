
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

# --- Simulation function for sensitivity analysis ---
def simulate_reaction_sensitivity(C_rate, M_constant, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt):
    R_initial, P_initial = initial_conditions
    Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val = arrhenius_params
    Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L = temp_params

    Mass_solution = Volume_L * Density_solution

    Current_Temperature_K = Initial_Temperature_K
    Current_Concentration_R = R_initial
    Current_Concentration_P = P_initial

    temperatures = [Initial_Temperature_K]
    r_concentrations = [R_initial]

    for i in range(1, num_steps + 1):
        kf_base = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Current_Temperature_K))
        kr_base = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Current_Temperature_K))

        kf = apply_ubp_operator(kf_base, UBP_OPERATOR_TYPE, C_rate, M_constant)
        kr = kr_base # UBP operator only on kf for this analysis

        rate_forward = kf * Current_Concentration_R
        rate_reverse = kr * Current_Concentration_P
        net_rate_R = -rate_forward + rate_reverse

        delta_R_concentration = net_rate_R * time_step_dt

        next_R = Current_Concentration_R + delta_R_concentration
        next_P = Current_Concentration_P - delta_R_concentration

        if next_R < 0:
            delta_R_concentration = -Current_Concentration_R
            next_R = 0.0
            next_P = Current_Concentration_R + Current_Concentration_P
        elif next_P < 0:
            delta_R_concentration = Current_Concentration_P
            next_P = 0.0
            next_R = Current_Concentration_R + Current_Concentration_P

        moles_converted_net = -delta_R_concentration * Volume_L
        heat_change = moles_converted_net * Enthalpy_of_Reaction_dH
        delta_T = heat_change / (Mass_solution * Specific_Heat_Capacity_solution)

        Current_Concentration_R += delta_R_concentration
        Current_Concentration_P -= delta_R_concentration
        Current_Temperature_K += delta_T

        Current_Concentration_R = max(0.0, Current_Concentration_R)
        Current_Concentration_P = max(0.0, Current_Concentration_P)

        temperatures.append(Current_Temperature_K)
        r_concentrations.append(Current_Concentration_R)

    return r_concentrations[-1], np.std(temperatures)

# --- Study 12: Sensitivity Analysis ---

# 1. Initialization:
R_initial = 1.0  # Initial concentration of reactant R in mol/L
P_initial = 0.0  # Initial concentration of product P in mol/L
Time_Step_dt = 0.1  # Time step duration in seconds
Num_Steps = 100  # Number of simulation steps

# Arrhenius parameters
Pre_exponential_Factor_A_f = 1.0e10  # s^-1
Activation_Energy_Ea_f = 60000.0  # J/mol
Pre_exponential_Factor_A_r = 1.0e9   # s^-1
Activation_Energy_Ea_r = 70000.0   # J/mol
Gas_Constant_R_val = 8.314  # J/(mol*K)

# Dynamic Temperature Parameters
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K)
Density_solution = 1000.0  # g/L
Enthalpy_of_Reaction_dH = -50000.0  # J/mol (for R -> P, negative for exothermic)
Volume_L = 1.0 # Volume of the reactor in Liters

initial_conditions = (R_initial, P_initial)
arrhenius_params = (Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val)
temp_params = (Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L)

# Parameter ranges for sensitivity analysis
C_rate_values = np.linspace(0.1, 100.0, 10) # 10 values from 0.1 to 100
M_constant_values = np.linspace(0.1, 10.0, 5) # 5 values from 0.1 to 10

operator_types = ['linear', 'quadratic', 'compositional']

sensitivity_results = []

print("--- Study 12: Sensitivity Analysis of UBP Operator Parameters ---")

for op_type in operator_types:
    print(f"\nAnalyzing {op_type} operator...")
    for c_rate in C_rate_values:
        for m_constant in M_constant_values:
            final_R, temp_std = simulate_reaction_sensitivity(
                c_rate, m_constant, op_type, initial_conditions, arrhenius_params,
                temp_params, Num_Steps, Time_Step_dt
            )
            sensitivity_results.append({
                'Operator_Type': op_type,
                'C_rate': c_rate,
                'M_constant': m_constant,
                'Final_R_Concentration': final_R,
                'Temperature_Std_Dev': temp_std
            })
            # print(f"  C_rate: {c_rate:.2f}, M_constant: {m_constant:.2f} -> Final R: {final_R:.4f}, Temp Std: {temp_std:.4f}")

df_sensitivity = pd.DataFrame(sensitivity_results)

# Save results to CSV
output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study12_sensitivity_analysis_results.csv"
df_sensitivity.to_csv(output_path, index=False)
print(f"\nSensitivity analysis results saved to {output_path}")

# Display some results for quick check
print("\nSample of Sensitivity Results:")
print(df_sensitivity.head())
print(df_sensitivity.tail())

