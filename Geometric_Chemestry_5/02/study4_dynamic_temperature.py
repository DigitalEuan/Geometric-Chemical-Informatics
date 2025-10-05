
import math
import pandas as pd

# --- UBP-inspired constants and functions (from previous studies) ---
def apply_ubp_operator(base_value, operator_type, C_rate, M_constant=math.e):
    if operator_type == 'linear':
        return base_value * (1 + C_rate / 100.0)
    elif operator_type == 'quadratic':
        return base_value * (1 + (C_rate / 100.0)**2)
    elif operator_type == 'compositional':
        return base_value * (1 + (M_constant * C_rate / 1000.0))
    else:
        return base_value

# --- Study 4: Dynamic Temperature Modeling ---

# 1. Initialization:
R_initial = 1.0  # Initial concentration of reactant R in mol/L
Time_Step_dt = 0.1  # Time step duration in seconds (reduced for better resolution)
Num_Steps = 100  # Number of simulation steps (increased for longer simulation time)

# Arrhenius parameters
Pre_exponential_Factor_A = 1.0e10  # s^-1
Activation_Energy_Ea = 60000.0  # J/mol
Gas_Constant_R = 8.314  # J/(mol*K)

# Dynamic Temperature Parameters
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K) - Specific heat capacity of the solution (e.g., water)
Density_solution = 1000.0  # g/L - Density of the solution (e.g., water)
Enthalpy_of_Reaction_dH = -5000.0  # J/mol - Enthalpy change of the reaction (reduced for more realistic temp change)
Volume_L = 1.0 # Volume of the reactor in Liters

Mass_solution = Volume_L * Density_solution # g

# UBP Parameters (not directly used in this script for k calculation, but kept for consistency)
UBP_OPERATOR_TYPE = None
UBP_C_RATE = 0.0
UBP_M_CONSTANT = math.pi

Current_Time = 0.0
Current_Temperature_K = Initial_Temperature_K
Current_Concentration_R = R_initial

Results_Table = []

# Add initial state (T=0)
initial_k = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Initial_Temperature_K))
Results_Table.append((Current_Time, Current_Concentration_R, Current_Temperature_K, initial_k))

# 2. Simulation Loop
for i in range(1, Num_Steps + 1):
    Rate_Constant_k = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Current_Temperature_K))

    if UBP_OPERATOR_TYPE:
        Rate_Constant_k = apply_ubp_operator(Rate_Constant_k, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)

    calculated_delta_R_concentration = -Rate_Constant_k * Current_Concentration_R * Time_Step_dt
    delta_R_concentration = max(-Current_Concentration_R, calculated_delta_R_concentration)

    moles_reacted = -delta_R_concentration * Volume_L
    heat_change = moles_reacted * Enthalpy_of_Reaction_dH
    delta_T = heat_change / (Mass_solution * Specific_Heat_Capacity_solution)

    Current_Concentration_R += delta_R_concentration
    Current_Temperature_K += delta_T

    Current_Concentration_R = max(0.0, Current_Concentration_R)

    Current_Time = i * Time_Step_dt
    Results_Table.append((Current_Time, Current_Concentration_R, Current_Temperature_K, Rate_Constant_k))

df_study4 = pd.DataFrame(Results_Table, columns=[
    'Time (s)', 'Concentration (mol/L)', 'Temperature (K)', 'Rate Constant (k) s^-1'
])

print("--- Study 4 Simulation Results (Dynamic Temperature) ---")
print(f"Initial Temperature: {Initial_Temperature_K} K")
print(f"Enthalpy of Reaction: {Enthalpy_of_Reaction_dH} J/mol")
print(f"Specific Heat Capacity (solution): {Specific_Heat_Capacity_solution} J/(g*K)")
print(f"Density of solution: {Density_solution} g/L")
print(f"Volume of reactor: {Volume_L} L")
print(df_study4.to_string(index=False))

output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study4_dynamic_temperature_results.csv"
df_study4.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

