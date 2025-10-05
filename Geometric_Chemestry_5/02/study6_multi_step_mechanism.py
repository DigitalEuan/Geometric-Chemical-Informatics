
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

# --- Study 6: Multi-step Reaction Mechanism with Dynamic Temperature and UBP Operators ---

# 1. Initialization:
A_initial = 1.0  # Initial concentration of reactant A in mol/L
I_initial = 0.0  # Initial concentration of intermediate I in mol/L
P_initial = 0.0  # Initial concentration of product P in mol/L
Time_Step_dt = 0.1  # Time step duration in seconds (reduced for better numerical stability)
Num_Steps = 100  # Number of simulation steps (increased for longer simulation time)

# Arrhenius parameters for elementary step 1 (A -> I)
Pre_exponential_Factor_A1 = 1.0e10  # s^-1
Activation_Energy_Ea1 = 60000.0  # J/mol
Enthalpy_of_Reaction_dH1 = -30000.0  # J/mol (exothermic)

# Arrhenius parameters for elementary step 2 (I -> P)
Pre_exponential_Factor_A2 = 5.0e9  # s^-1
Activation_Energy_Ea2 = 55000.0  # J/mol
Enthalpy_of_Reaction_dH2 = -20000.0  # J/mol (exothermic)

Gas_Constant_R_val = 8.314  # J/(mol*K)

# Dynamic Temperature Parameters (from Study 4/5)
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K)
Density_solution = 1000.0  # g/L
Volume_L = 1.0 # Volume of the reactor in Liters
Mass_solution = Volume_L * Density_solution # g

# UBP Parameters (applied to k1 for demonstration)
UBP_OPERATOR_TYPE = 'quadratic' # Example: apply quadratic UBP operator to k1
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

Current_Time = 0.0
Current_Temperature_K = Initial_Temperature_K
Current_Concentration_A = A_initial
Current_Concentration_I = I_initial
Current_Concentration_P = P_initial

Results_Table = []  # List to store (Time, A, I, P, Temperature, k1, k2) tuples

# Add initial state (T=0)
initial_k1_base = Pre_exponential_Factor_A1 * math.exp(-Activation_Energy_Ea1 / (Gas_Constant_R_val * Initial_Temperature_K))
initial_k1_modified = apply_ubp_operator(initial_k1_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)
initial_k2 = Pre_exponential_Factor_A2 * math.exp(-Activation_Energy_Ea2 / (Gas_Constant_R_val * Initial_Temperature_K))
Results_Table.append((Current_Time, Current_Concentration_A, Current_Concentration_I, Current_Concentration_P, Current_Temperature_K, initial_k1_modified, initial_k2))

# 2. Simulation Loop (using numerical integration for concentrations and temperature):
for i in range(1, Num_Steps + 1):
    # Calculate current rate constants k1 and k2 using Arrhenius equation with current temperature
    k1_base = Pre_exponential_Factor_A1 * math.exp(-Activation_Energy_Ea1 / (Gas_Constant_R_val * Current_Temperature_K))
    k1 = apply_ubp_operator(k1_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)

    k2 = Pre_exponential_Factor_A2 * math.exp(-Activation_Energy_Ea2 / (Gas_Constant_R_val * Current_Temperature_K))

    # Calculate rates of elementary steps
    rate_step1 = k1 * Current_Concentration_A
    rate_step2 = k2 * Current_Concentration_I

    # Calculate changes in concentrations (using Euler method)
    delta_A = -rate_step1 * Time_Step_dt
    delta_I = (rate_step1 - rate_step2) * Time_Step_dt
    delta_P = rate_step2 * Time_Step_dt

    # Update concentrations
    next_A = Current_Concentration_A + delta_A
    next_I = Current_Concentration_I + delta_I
    next_P = Current_Concentration_P + delta_P

    # Ensure concentrations remain non-negative
    Current_Concentration_A = max(0.0, next_A)
    Current_Concentration_I = max(0.0, next_I)
    Current_Concentration_P = max(0.0, next_P)

    # Moles reacted in each step for energy balance
    moles_reacted_step1 = (A_initial - Current_Concentration_A) * Volume_L # Moles of A consumed
    moles_reacted_step2 = (P_initial - next_P) * Volume_L # Moles of P formed

    # Heat change due to reaction (sum of heat from both elementary steps)
    # Note: this is a simplification. A more rigorous approach would track heat generation per step
    # based on actual reaction extent for each step within the dt.
    # For now, let's use the change in A and I to estimate heat generation.
    heat_change = (-delta_A * Enthalpy_of_Reaction_dH1 + (delta_I + delta_A) * Enthalpy_of_Reaction_dH2) * Volume_L # J
    # A more accurate approach for heat_change:
    # heat_change = (rate_step1 * Enthalpy_of_Reaction_dH1 + rate_step2 * Enthalpy_of_Reaction_dH2) * Time_Step_dt * Volume_L
    # Let's use the more accurate one:
    heat_change = (rate_step1 * Enthalpy_of_Reaction_dH1 + rate_step2 * Enthalpy_of_Reaction_dH2) * Time_Step_dt * Volume_L

    # Temperature change: dT = Q / (Mass_solution * Specific_Heat_Capacity_solution)
    delta_T = heat_change / (Mass_solution * Specific_Heat_Capacity_solution) # K

    Current_Temperature_K += delta_T

    # Record the result
    Current_Time = i * Time_Step_dt
    Results_Table.append((Current_Time, Current_Concentration_A, Current_Concentration_I, Current_Concentration_P, Current_Temperature_K, k1, k2))

# Convert to DataFrame for better presentation
df_study6 = pd.DataFrame(Results_Table, columns=[
    'Time (s)', 'A Concentration (mol/L)', 'I Concentration (mol/L)', 'P Concentration (mol/L)',
    'Temperature (K)', 'k1 (s^-1)', 'k2 (s^-1)'
])

# 3. Output Generation:
print("--- Study 6 Simulation Results (Multi-step Reaction with Dynamic Temperature and UBP Operator) ---")
print(f"Initial Temperature: {Initial_Temperature_K} K")
print(f"Enthalpy of Reaction Step 1 (A->I): {Enthalpy_of_Reaction_dH1} J/mol")
print(f"Enthalpy of Reaction Step 2 (I->P): {Enthalpy_of_Reaction_dH2} J/mol")
print(f"Specific Heat Capacity (solution): {Specific_Heat_Capacity_solution} J/(g*K)")
print(f"UBP Operator Type (on k1): {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(df_study6.to_string(index=False))

# Save results to CSV
output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study6_multi_step_mechanism_results.csv"
df_study6.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

