
import math
import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling as get_sampling
from pymoo.operators.crossover.sbx import SBX as get_crossover
from pymoo.operators.mutation.pm import PolynomialMutation as get_mutation
import matplotlib.pyplot as plt
import os

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

# --- Problem Definition for pymoo (NSGA-II) ---
class ChemicalReactionProblem(Problem):

    def __init__(self, UBP_OPERATOR_TYPE, initial_conditions, arrhenius_params, temp_params, num_steps, time_step_dt, target_R, target_temp_std):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0.1, 0.1]), xu=np.array([500.0, 50.0]))
        self.UBP_OPERATOR_TYPE = UBP_OPERATOR_TYPE
        self.initial_conditions = initial_conditions
        self.arrhenius_params = arrhenius_params
        self.temp_params = temp_params
        self.num_steps = num_steps
        self.time_step_dt = time_step_dt
        self.target_R = target_R
        self.target_temp_std = target_temp_std

    def _evaluate(self, x, out, *args, **kwargs):
        obj1_values = [] # Final R concentration (minimize)
        obj2_values = [] # Temperature standard deviation (minimize)

        for i in range(len(x)):
            C_rate = x[i, 0]
            M_constant = x[i, 1]

            r_concs, p_concs, temps = simulate_reaction_full_output(
                C_rate, M_constant, self.UBP_OPERATOR_TYPE, self.initial_conditions,
                self.arrhenius_params, self.temp_params, self.num_steps, self.time_step_dt
            )

            # Objective 1: Final R concentration (we want to minimize this, so it's directly an objective)
            final_R_concentration = r_concs[-1]

            # Objective 2: Temperature standard deviation (we want to minimize this)
            temp_std = np.std(temps)

            obj1_values.append(final_R_concentration)
            obj2_values.append(temp_std)

        out["F"] = np.column_stack([obj1_values, obj2_values])

# --- Study 14: NSGA-II Multi-objective Optimization ---

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
all_pareto_fronts = {}

for op_type in operator_types:
    print(f"\n--- Running NSGA-II for UBP Operator: {op_type} ---")

    problem = ChemicalReactionProblem(
        UBP_OPERATOR_TYPE=op_type,
        initial_conditions=initial_conditions,
        arrhenius_params=arrhenius_params,
        temp_params=temp_params,
        num_steps=Num_Steps,
        time_step_dt=Time_Step_dt,
        target_R=target_R,
        target_temp_std=target_temp_std
    )

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=10,
        sampling=get_sampling(),
        crossover=get_crossover(prob=0.9, eta=15),
        mutation=get_mutation(prob=1.0 / problem.n_var, eta=20),
        eliminate_duplicates=True
    )

    res = pymoo_minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=True, save_history=True)

    # Extract Pareto front (non-dominated solutions)
    F = res.F
    X = res.X

    all_pareto_fronts[op_type] = {'F': F, 'X': X}

    print(f"Found {len(F)} non-dominated solutions for {op_type} operator.")

    # Save Pareto front to CSV
    df_pareto = pd.DataFrame(F, columns=['Final_R_Concentration', 'Temperature_Std_Dev'])
    df_pareto['UBP_Operator_Type'] = op_type
    df_pareto['C_rate'] = X[:, 0]
    df_pareto['M_constant'] = X[:, 1]

    output_path = os.path.join("/home/ubuntu/ChemicalReactionKinetics/results/", f"study14_nsga2_pareto_front_{op_type}_results.csv")
    df_pareto.to_csv(output_path, index=False)
    print(f"Pareto front for {op_type} saved to {output_path}")

# Plotting Pareto fronts
plt.figure(figsize=(10, 8))
for op_type, data in all_pareto_fronts.items():
    plt.scatter(data['F'][:, 0], data['F'][:, 1], label=f'{op_type} Operator', alpha=0.7)

plt.title('Study 14: Pareto Fronts for Multi-objective Optimization (NSGA-II)')
plt.xlabel('Final R Concentration (mol/L)')
plt.ylabel('Temperature Standard Deviation (K)')
plt.axvline(x=target_R, color='r', linestyle='--', label='Target R Concentration')
plt.axhline(y=target_temp_std, color='g', linestyle='--', label='Target Temp Std Dev')
plt.legend()
plt.grid(True)

plot_path = os.path.join("/home/ubuntu/ChemicalReactionKinetics/results/", 'study14_nsga2_pareto_fronts_plot.png')
plt.savefig(plot_path)
print(f"Pareto fronts plot saved to {plot_path}")

print("\nNSGA-II optimization complete.")

