
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

# --- Study 10: Gillespie Algorithm with UBP Operators (Isothermal for simplicity) ---

# 1. Initialization:
# Reaction: R -> P (irreversible, first-order)
# We will use an irreversible reaction for the initial Gillespie implementation for simplicity.

# Initial number of molecules (instead of concentration for Gillespie)
N_R_initial = 6000  # Initial number of R molecules
N_P_initial = 0     # Initial number of P molecules

Volume_L = 1.0 # Volume of the system in Liters
Avogadro_Constant = 6.022e23 # molecules/mol

# Base rate constant (from previous studies, assuming isothermal for now)
# Let's use a k value that results in a reasonable number of events over the simulation time
Base_Rate_Constant_k = 0.1 # s^-1 (This is a macroscopic rate constant)

# UBP Parameters
UBP_OPERATOR_TYPE = 'quadratic'
UBP_C_RATE = 10.0
UBP_M_CONSTANT = math.pi

# Simulation time parameters
Max_Simulation_Time = 10.0 # seconds

# Convert macroscopic rate constant to microscopic rate constant (c_i for Gillespie)
# For a first-order reaction R -> P, the propensity function is a_1 = c_1 * N_R
# where c_1 = k / Volume_L (if k is in L/mol/s for second order) or k (if k is in s^-1 for first order)
# Here, k is already first order (s^-1), so c_1 = k

# Apply UBP operator to the base rate constant
modified_k = apply_ubp_operator(Base_Rate_Constant_k, UBP_OPERATOR_TYPE, UBP_C_RATE, UBP_M_CONSTANT)

c1 = modified_k # Microscopic rate constant for R -> P

# Stoichiometry matrix (change in species counts for each reaction)
# Reactions: [R -> P]
# Species:   [R, P]
stoichiometry = np.array([[-1, 1]])

# Current state variables
Current_Time = 0.0
N_R = N_R_initial
N_P = N_P_initial

Results_Table = []

# Record initial state
Results_Table.append({
    'Time (s)': Current_Time,
    'N_R': N_R,
    'N_P': N_P,
    'R Concentration (mol/L)': N_R / (Avogadro_Constant * Volume_L),
    'P Concentration (mol/L)': N_P / (Avogadro_Constant * Volume_L),
    'k (s^-1)': modified_k
})

# 2. Gillespie Simulation Loop:
step_count = 0
while Current_Time < Max_Simulation_Time and N_R > 0:
    step_count += 1

    # Calculate propensities (a_i)
    # For R -> P, a_1 = c_1 * N_R
    a1 = c1 * N_R
    propensities = np.array([a1])

    a0 = np.sum(propensities) # Sum of all propensities

    if a0 == 0: # No reactions can occur, system is static
        break

    # Generate two random numbers (r1, r2) from a uniform distribution [0, 1)
    r1, r2 = np.random.rand(2)

    # Calculate time until next reaction (tau)
    tau = (1.0 / a0) * math.log(1.0 / r1)

    # Determine which reaction occurs (j)
    # Find j such that sum(a_i for i=1 to j-1) < r2*a0 <= sum(a_i for i=1 to j)
    reaction_index = np.where(r2 * a0 <= np.cumsum(propensities))[0][0]

    # Update time and species counts
    Current_Time += tau
    N_R += stoichiometry[reaction_index, 0]
    N_P += stoichiometry[reaction_index, 1]

    # Record state at each reaction event (or at fixed intervals for smoother plots)
    # For now, record at each event. We can interpolate later for fixed intervals if needed.
    Results_Table.append({
        'Time (s)': Current_Time,
        'N_R': N_R,
        'N_P': N_P,
        'R Concentration (mol/L)': N_R / (Avogadro_Constant * Volume_L),
        'P Concentration (mol/L)': N_P / (Avogadro_Constant * Volume_L),
        'k (s^-1)': modified_k
    })

# Convert to DataFrame for better presentation
df_study10 = pd.DataFrame(Results_Table)

# 3. Output Generation:
print("--- Study 10 Simulation Results (Gillespie Algorithm with UBP Operators) ---")
print(f"Initial Number of R molecules: {N_R_initial}")
print(f"Volume: {Volume_L} L")
print(f"Base Rate Constant: {Base_Rate_Constant_k} s^-1")
print(f"UBP Operator Type: {UBP_OPERATOR_TYPE}")
print(f"UBP C_rate: {UBP_C_RATE}")
print(f"Modified Rate Constant (k): {modified_k:.4f} s^-1")
print(f"Total Gillespie Steps: {step_count}")
print(df_study10.head())
print(df_study10.tail())

# Save results to CSV
output_path = "/home/ubuntu/ChemicalReactionKinetics/results/study10_gillespie_algorithm_results.csv"
df_study10.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

