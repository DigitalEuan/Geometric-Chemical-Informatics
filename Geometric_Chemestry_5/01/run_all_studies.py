
import math
import numpy as np
import pandas as pd

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

def run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k):
    Current_Time = 0.0
    Results_Table = []
    Results_Table.append((Current_Time, R_initial))

    for i in range(1, Num_Steps + 1):
        Current_Time = i * Time_Step_dt
        Concentration_R = R_initial * math.exp(-Rate_Constant_k * Current_Time)
        Results_Table.append((Current_Time, Concentration_R))
    return pd.DataFrame(Results_Table, columns=['Time (s)', 'Concentration (units)'])

# --- Common Simulation Parameters ---
R_initial = 100.0
Time_Step_dt = 1.0
Num_Steps = 10

# --- Study 1: Basic First-Order Kinetics ---
print("\n--- Running Study 1: Basic First-Order Kinetics ---")
Rate_Constant_k_study1 = 0.1 # From original Study 1
df_study1 = run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k_study1)
print(df_study1)

# --- Study 2: Arrhenius Equation ---
print("\n--- Running Study 2: Arrhenius Equation ---")
Pre_exponential_Factor_A = 1.0e5
Activation_Energy_Ea = 30000.0
Gas_Constant_R = 8.314
Temperature_K = 298.15

Rate_Constant_k_study2 = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))
df_study2 = run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k_study2)
print(f"Temperature: {Temperature_K} K")
print(f"Calculated Rate Constant (k): {Rate_Constant_k_study2:.4e} s^-1")
print(df_study2)

# --- Study 3: Arrhenius Equation with UBP Operator ---
print("\n--- Running Study 3: Arrhenius Equation with UBP Operator (Quadratic) ---")
UBP_OPERATOR_TYPE = 'quadratic'
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

Rate_Constant_k_study3_base = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))
Rate_Constant_k_study3_modified = apply_ubp_operator(Rate_Constant_k_study3_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)
df_study3_quadratic = run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k_study3_modified)
print(f"Temperature: {Temperature_K} K")
print(f"Base Rate Constant (k_base): {Rate_Constant_k_study3_base:.4e} s^-1")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(f"Modified Rate Constant (k_modified): {Rate_Constant_k_study3_modified:.4e} s^-1")
print(df_study3_quadratic)

print("\n--- Running Study 3: Arrhenius Equation with UBP Operator (Linear) ---")
UBP_OPERATOR_TYPE = 'linear'
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

Rate_Constant_k_study3_base = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))
Rate_Constant_k_study3_modified = apply_ubp_operator(Rate_Constant_k_study3_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)
df_study3_linear = run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k_study3_modified)
print(f"Temperature: {Temperature_K} K")
print(f"Base Rate Constant (k_base): {Rate_Constant_k_study3_base:.4e} s^-1")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(f"Modified Rate Constant (k_modified): {Rate_Constant_k_study3_modified:.4e} s^-1")
print(df_study3_linear)

print("\n--- Running Study 3: Arrhenius Equation with UBP Operator (Compositional) ---")
UBP_OPERATOR_TYPE = 'compositional'
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

Rate_Constant_k_study3_base = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))
Rate_Constant_k_study3_modified = apply_ubp_operator(Rate_Constant_k_study3_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)
df_study3_compositional = run_simulation(R_initial, Time_Step_dt, Num_Steps, Rate_Constant_k_study3_modified)
print(f"Temperature: {Temperature_K} K")
print(f"Base Rate Constant (k_base): {Rate_Constant_k_study3_base:.4e} s^-1")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(f"UBP M_constant: {UBP_M_CONSTANT}")
print(f"Modified Rate Constant (k_modified): {Rate_Constant_k_study3_modified:.4e} s^-1")
print(df_study3_compositional)

# --- Save all results to CSV for later analysis ---
df_study1.to_csv('/home/ubuntu/ChemicalReactionKinetics/results/study1_results.csv', index=False)
df_study2.to_csv('/home/ubuntu/ChemicalReactionKinetics/results/study2_results.csv', index=False)
df_study3_quadratic.to_csv('/home/ubuntu/ChemicalReactionKinetics/results/study3_quadratic_results.csv', index=False)
df_study3_linear.to_csv('/home/ubuntu/ChemicalReactionKinetics/results/study3_linear_results.csv', index=False)
df_study3_compositional.to_csv('/home/ubuntu/ChemicalReactionKinetics/results/study3_compositional_results.csv', index=False)

print("\nAll study results saved to CSV files in /home/ubuntu/ChemicalReactionKinetics/results/")

