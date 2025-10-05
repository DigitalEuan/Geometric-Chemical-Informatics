
import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

# --- Simulation function for optimization ---
def simulate_reaction(C_rate, M_constant, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt):
    R_initial, P_initial = initial_conditions
    Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val = arrhenius_params
    Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L = temp_params

    Mass_solution = Volume_L * Density_solution

    Current_Temperature_K = Initial_Temperature_K
    Current_Concentration_R = R_initial
    Current_Concentration_P = P_initial

    # Store only the final concentration for the objective function
    final_R_concentration = R_initial

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
        
        final_R_concentration = Current_Concentration_R

    return final_R_concentration

# --- Objective function for minimization ---
def objective_function(params, target_R, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt):
    C_rate, M_constant = params
    simulated_R = simulate_reaction(C_rate, M_constant, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt)
    return (simulated_R - target_R)**2 # Minimize the squared difference from target

# --- Study 8: Parameter Optimization for UBP Operators ---

# 1. Initialization:
R_initial = 1.0  # Initial concentration of reactant R in mol/L
P_initial = 0.0  # Initial concentration of product P in mol/L
Time_Step_dt = 0.1  # Time step duration in seconds
Num_Steps = 100  # Number of simulation steps

# Arrhenius parameters
Pre_exponential_Factor_A_f = 1.0e10  # s^-1
Activation_Energy_Ea_f = 60000.0  # J/mol
Pre_exponential_Factor_A_r = 1.0e9  # s^-1
Activation_Energy_Ea_r = 70000.0  # J/mol
Gas_Constant_R_val = 8.314  # J/(mol*K)

# Dynamic Temperature Parameters
Initial_Temperature_K = 298.15  # Initial temperature in Kelvin
Specific_Heat_Capacity_solution = 4.184  # J/(g*K)
Density_solution = 1000.0  # g/L
Enthalpy_of_Reaction_dH = -50000.0  # J/mol
Volume_L = 1.0 # Volume of the reactor in Liters

# Target for optimization: A specific final concentration of R
# Let's aim for a lower concentration than the baseline without UBP to show amplification
# First, run a baseline simulation without UBP to get a reference final R
base_kf = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Initial_Temperature_K))
base_kr = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Initial_Temperature_K))

def run_baseline_simulation(initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt):
    R_initial, P_initial = initial_conditions
    Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val = arrhenius_params
    Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L = temp_params

    Mass_solution = Volume_L * Density_solution

    Current_Temperature_K = Initial_Temperature_K
    Current_Concentration_R = R_initial
    Current_Concentration_P = P_initial

    for i in range(1, num_steps + 1):
        kf = Pre_exponential_Factor_A_f * math.exp(-Activation_Energy_Ea_f / (Gas_Constant_R_val * Current_Temperature_K))
        kr = Pre_exponential_Factor_A_r * math.exp(-Activation_Energy_Ea_r / (Gas_Constant_R_val * Current_Temperature_K))

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

    return Current_Concentration_R

initial_conditions = (R_initial, P_initial)
arrhenius_params = (Pre_exponential_Factor_A_f, Activation_Energy_Ea_f, Pre_exponential_Factor_A_r, Activation_Energy_Ea_r, Gas_Constant_R_val)
temp_params = (Initial_Temperature_K, Specific_Heat_Capacity_solution, Density_solution, Enthalpy_of_Reaction_dH, Volume_L)

baseline_final_R = run_baseline_simulation(initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)
target_R = baseline_final_R * 0.8 # Aim for 20% lower concentration, implying faster reaction

print(f"Baseline final R concentration without UBP: {baseline_final_R:.4f} mol/L")
print(f"Target final R concentration for optimization: {target_R:.4f} mol/L")

# Optimization for 'quadratic' UBP operator
UBP_OPERATOR_TYPE_OPT = 'quadratic'
x0 = [10.0, math.pi]  # Initial guess for [C_rate, M_constant]
bounds = [(0.1, 100.0), (0.1, 10.0)] # Bounds for C_rate and M_constant

print(f"\n--- Optimizing for UBP Operator: {UBP_OPERATOR_TYPE_OPT} ---")
result_quadratic = minimize(objective_function, x0, args=(target_R, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt), bounds=bounds, method='L-BFGS-B')

optimized_C_rate_quadratic, optimized_M_constant_quadratic = result_quadratic.x
optimized_final_R_quadratic = simulate_reaction(optimized_C_rate_quadratic, optimized_M_constant_quadratic, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)

print(f"Optimization successful: {result_quadratic.success}")
print(f"Optimized C_rate ({UBP_OPERATOR_TYPE_OPT}): {optimized_C_rate_quadratic:.4f}")
print(f"Optimized M_constant ({UBP_OPERATOR_TYPE_OPT}): {optimized_M_constant_quadratic:.4f}")
print(f"Final R concentration with optimized {UBP_OPERATOR_TYPE_OPT} UBP: {optimized_final_R_quadratic:.4f} mol/L")

# Optimization for 'linear' UBP operator
UBP_OPERATOR_TYPE_OPT = 'linear'
x0 = [10.0, math.e]  # Initial guess for [C_rate, M_constant]
bounds = [(0.1, 100.0), (0.1, 10.0)] # Bounds for C_rate and M_constant

print(f"\n--- Optimizing for UBP Operator: {UBP_OPERATOR_TYPE_OPT} ---")
result_linear = minimize(objective_function, x0, args=(target_R, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt), bounds=bounds, method='L-BFGS-B')

optimized_C_rate_linear, optimized_M_constant_linear = result_linear.x
optimized_final_R_linear = simulate_reaction(optimized_C_rate_linear, optimized_M_constant_linear, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)


print(f"Optimization successful: {result_linear.success}")
print(f"Optimized C_rate ({UBP_OPERATOR_TYPE_OPT}): {optimized_C_rate_linear:.4f}")
print(f"Optimized M_constant ({UBP_OPERATOR_TYPE_OPT}): {optimized_M_constant_linear:.4f}")
print(f"Final R concentration with optimized {UBP_OPERATOR_TYPE_OPT} UBP: {optimized_final_R_linear:.4f} mol/L")

# Optimization for 'compositional' UBP operator
UBP_OPERATOR_TYPE_OPT = 'compositional'
x0 = [10.0, math.pi]  # Initial guess for [C_rate, M_constant]
bounds = [(0.1, 100.0), (0.1, 10.0)] # Bounds for C_rate and M_constant

print(f"\n--- Optimizing for UBP Operator: {UBP_OPERATOR_TYPE_OPT} ---")
result_compositional = minimize(objective_function, x0, args=(target_R, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt), bounds=bounds, method='L-BFGS-B')

optimized_C_rate_compositional, optimized_M_constant_compositional = result_compositional.x
optimized_final_R_compositional = simulate_reaction(optimized_C_rate_compositional, optimized_M_constant_compositional, UBP_OPERATOR_TYPE_OPT, initial_conditions, arrhenius_params, temp_params, Num_Steps, Time_Step_dt)

print(f"Optimization successful: {result_compositional.success}")
print(f"Optimized C_rate ({UBP_OPERATOR_TYPE_OPT}): {optimized_C_rate_compositional:.4f}")
print(f"Optimized M_constant ({UBP_OPERATOR_TYPE_OPT}): {optimized_M_constant_compositional:.4f}")
print(f"Final R concentration with optimized {UBP_OPERATOR_TYPE_OPT} UBP: {optimized_final_R_compositional:.4f} mol/L")

# Store optimization results
optimization_results = {
    'UBP_Operator_Type': ["quadratic", "linear", "compositional"],
    'Optimized_C_rate': [optimized_C_rate_quadratic, optimized_C_rate_linear, optimized_C_rate_compositional],
    'Optimized_M_constant': [optimized_M_constant_quadratic, optimized_M_constant_linear, optimized_M_constant_compositional],
    'Final_R_Concentration': [optimized_final_R_quadratic, optimized_final_R_linear, optimized_final_R_compositional],
    'Optimization_Success': [result_quadratic.success, result_linear.success, result_compositional.success]
}
df_optimization = pd.DataFrame(optimization_results)

output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study8_optimization_results.csv"
df_optimization.to_csv(output_path, index=False)
print(f"\nOptimization results saved to {output_path}")


