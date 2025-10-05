
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

# --- Study 5: Reversible Reactions with Dynamic Temperature ---

# 1. Initialization:
R_initial = 1.0  # Initial concentration of reactant R in mol/L
P_initial = 0.0  # Initial concentration of product P in mol/L
Time_Step_dt = 1.0  # Time step duration in seconds
Num_Steps = 10  # Number of simulation steps

# Arrhenius parameters for forward reaction (R -> P)
Pre_exponential_Factor_A_f = 1.0e10  # s^-1
Activation_Energy_Ea_f = 60000.0  # J/mol

# Arrhenius parameters for reverse reaction (P -> R)
Pre_exponential_Factor_A_r = 1.0e9  # s^-1 (typically smaller for reverse)
Activation_Energy_Ea_r = 70000.0  # J/mol (typically higher for reverse to favor forward)

Gas_Constant_R_val = 8.314  # J/(mol*K)

# Dynamic Temperature Parameters (from Study 4)
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K)
Density_solution = 1000.0  # g/L
Enthalpy_of_Reaction_dH = -50000.0  # J/mol (for R -> P, negative for exothermic)
Volume_L = 1.0 # Volume of the reactor in Liters
Mass_solution = Volume_L * Density_solution # g

# UBP Parameters (applied to forward rate constant for demonstration)
UBP_OPERATOR_TYPE = 'quadratic' # Example: apply quadratic UBP operator
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

Current_Time = 0.0
Current_Temperature_K = Initial_Temperature_K
Current_Concentration_R = R_initial
Current_Concentration_P = P_initial

Results_Table = []  # List to store (Time, R_Concentration, P_Concentration, Temperature, kf, kr) tuples

# Add initial state (T=0)
initial_kf = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Initial_Temperature_K))
initial_kf_modified = apply_ubp_operator(initial_kf, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)
initial_kr = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Initial_Temperature_K))
Results_Table.append((Current_Time, Current_Concentration_R, Current_Concentration_P, Current_Temperature_K, initial_kf_modified, initial_kr))

# 2. Simulation Loop (using numerical integration for concentrations and temperature):
for i in range(1, Num_Steps + 1):
    # Calculate current forward rate constant kf using Arrhenius equation with current temperature
    kf_base = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Current_Temperature_K))
    kf = apply_ubp_operator(kf_base, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)

    # Calculate current reverse rate constant kr using Arrhenius equation with current temperature
    kr = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Current_Temperature_K))

    # Calculate net change in concentration for R (using Euler method)
    # dR/dt = -kf*R + kr*P
    rate_forward = kf * Current_Concentration_R
    rate_reverse = kr * Current_Concentration_P
    net_rate_R = -rate_forward + rate_reverse

    delta_R_concentration = net_rate_R * Time_Step_dt

    # Ensure concentrations remain non-negative and sum to total initial concentration
    next_R = Current_Concentration_R + delta_R_concentration
    next_P = Current_Concentration_P - delta_R_concentration # P changes by -delta_R

    # Simple check to prevent negative concentrations and conserve total mass
    if next_R < 0:
        delta_R_concentration = -Current_Concentration_R
        next_R = 0.0
        next_P = Current_Concentration_R + Current_Concentration_P
    elif next_P < 0:
        delta_R_concentration = Current_Concentration_P
        next_P = 0.0
        next_R = Current_Concentration_R + Current_Concentration_P

    # Moles reacted (forward direction) in this time step for energy balance
    # This is the amount of R that actually converted to P, considering net change
    moles_converted_net = -delta_R_concentration * Volume_L # If R decreases, moles_converted_net is positive

    # Heat change due to reaction (based on net conversion)
    heat_change = moles_converted_net * Enthalpy_of_Reaction_dH # J

    # Temperature change: dT = Q / (Mass_solution * Specific_Heat_Capacity_solution)
    delta_T = heat_change / (Mass_solution * Specific_Heat_Capacity_solution) # K

    Current_Concentration_R += delta_R_concentration
    Current_Concentration_P -= delta_R_concentration # P changes by -delta_R
    Current_Temperature_K += delta_T

    # Record the result
    Current_Time = i * Time_Step_dt
    Results_Table.append((Current_Time, Current_Concentration_R, Current_Concentration_P, Current_Temperature_K, kf, kr))

# Convert to DataFrame for better presentation
df_study5 = pd.DataFrame(Results_Table, columns=[
    'Time (s)', 'R Concentration (mol/L)', 'P Concentration (mol/L)',
    'Temperature (K)', 'kf (s^-1)', 'kr (s^-1)'
])

# 3. Output Generation:
print("--- Study 5 Simulation Results (Reversible Reactions with Dynamic Temperature) ---")
print(f"Initial Temperature: {Initial_Temperature_K} K")
print(f"Enthalpy of Reaction (R->P): {Enthalpy_of_Reaction_dH} J/mol")
print(f"Specific Heat Capacity (solution): {Specific_Heat_Capacity_solution} J/(g*K)")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(df_study5.to_string(index=False))

# Save results to CSV
output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study5_reversible_reactions_results.csv"
df_study5.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

