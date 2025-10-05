import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

# --- Simulation function for optimization (based on Study 5/8, with full results) ---
def simulate_reaction_full_output(C_rate, M_constant, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt):
    R_initial, P_initial = initial_conditions
    Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val = arrhenius_params
    Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L = temp_params

    Mass_solution = Volume_L * Density_solution

    Current_Temperature_K = Initial_Temperature_K
    Current_Concentration_R = R_initial
    Current_Concentration_P = P_initial

    temperatures = [Initial_Temperature_K]
    r_concentrations = [R_initial]
    p_concentrations = [P_initial]

    for i in range(1, num_steps + 1):
        kf_base = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Current_Temperature_K))
        kr_base = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Current_Temperature_K))

        kf = apply_ubp_operator(kf_base, UBP_OPERATOR_TYPE, C_rate, M_constant)
        kr = kr_base # UBP operator only on kf for this optimization

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
        p_concentrations.append(Current_Concentration_P)

    return r_concentrations, p_concentrations, temperatures

# --- Multi-objective function for minimization ---
def multi_objective_function(params, target_R, target_temp_std, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt, weight_conc, weight_temp):
    C_rate, M_constant = params
    r_concs, p_concs, temps = simulate_reaction_full_output(C_rate, M_constant, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt)

    # Objective 1: Minimize normalized squared difference from target final R concentration
    final_R_concentration = r_concs[-1]
    obj1 = ((final_R_concentration - target_R) / target_R)**2 if target_R != 0 else (final_R_concentration - target_R)**2

    # Objective 2: Minimize normalized squared difference from target temperature standard deviation
    temp_std = np.std(temps)
    obj2 = ((temp_std - target_temp_std) / target_temp_std)**2 if target_temp_std != 0 else (temp_std - target_temp_std)**2

    # Combined objective (weighted sum)
    return weight_conc * obj1 + weight_temp * obj2

# --- Study 13: Multi-objective Optimization with Alternative Weighting Schemes ---

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

# Determine baseline for target R and target temperature standard deviation
_, _, baseline_temps = simulate_reaction_full_output(0.0, math.e, None, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)
baseline_final_R, _, _ = simulate_reaction_full_output(0.0, math.e, None, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)

target_R = baseline_final_R[-1] * 0.8 # Aim for 20% lower R concentration
target_temp_std = np.std(baseline_temps) * 0.5 # Aim for 50% lower temperature fluctuation

print(f"Baseline final R concentration without UBP: {baseline_final_R[-1]:.4f} mol/L")
print(f"Baseline temperature standard deviation: {np.std(baseline_temps):.4f} K")
print(f"Target final R concentration for optimization: {target_R:.4f} mol/L")
print(f"Target temperature standard deviation for optimization: {target_temp_std:.4f} K")

operator_types = ['linear', 'quadratic', 'compositional']
weighting_schemes = [
    {'weight_conc': 0.9, 'weight_temp': 0.1, 'name': 'Heavy_Conc'},
    {'weight_conc': 0.5, 'weight_temp': 0.5, 'name': 'Balanced'},
    {'weight_conc': 0.1, 'weight_temp': 0.9, 'name': 'Heavy_Temp'}
]

all_optimization_results = []

for op_type in operator_types:
    for scheme in weighting_schemes:
        weight_conc = scheme['weight_conc']
        weight_temp = scheme['weight_temp']
        scheme_name = scheme['name']

        x0 = [10.0, math.pi]  # Initial guess for [C_rate, M_constant]
        bounds = [(0.1, 500.0), (0.1, 50.0)] # Bounds for C_rate and M_constant

        print(f"\n--- Optimizing for UBP Operator: {op_type}, Weighting: {scheme_name} (Conc:{weight_conc}, Temp:{weight_temp}) ---")
        result = minimize(multi_objective_function, x0, args=(target_R, target_temp_std, op_type, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt, weight_conc, weight_temp), bounds=bounds, method='L-BFGS-B')

        optimized_C_rate, optimized_M_constant = result.x
        final_r, final_p, final_t = simulate_reaction_full_output(optimized_C_rate, optimized_M_constant, op_type, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)

        all_optimization_results.append({
            'UBP_Operator_Type': op_type,
            'Weighting_Scheme': scheme_name,
            'Weight_Concentration': weight_conc,
            'Weight_Temperature': weight_temp,
            'Optimized_C_rate': optimized_C_rate,
            'Optimized_M_constant': optimized_M_constant,
            'Final_R_Concentration': final_r[-1],
            'Temperature_Std_Dev': np.std(final_t),
            'Optimization_Success': result.success
        })

df_all_optimization_results = pd.DataFrame(all_optimization_results)

output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study13_weighted_optimization_results.csv"
df_all_optimization_results.to_csv(output_path, index=False)
print(f"\nMulti-objective optimization results with varied weighting schemes saved to {output_path}")

print("\nSample of Weighted Optimization Results:")
print(df_all_optimization_results)

