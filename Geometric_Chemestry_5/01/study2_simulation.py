
import math

# --- UBP-inspired constants (for future integration, currently placeholders) ---
# These are conceptual and would need concrete definitions within the UBP framework
# For now, they are not directly used in the Arrhenius calculation but serve as a reminder
UBP_A_FACTOR = 1.0 # Placeholder for UBP pre-exponential factor influence
UBP_E_ACTIVATION_MODIFIER = 1.0 # Placeholder for UBP activation energy modification

# 1. Initialization (Study 2 - with Arrhenius Equation):
# Numerical Values from Study 1
R_initial = 100.0  # Initial concentration in units
Time_Step_dt = 1.0  # Time step duration in seconds
Num_Steps = 10  # Number of simulation steps

# New variables for Arrhenius Equation (as suggested in Study 1's Future Refinement)
# Adjusted values to get a rate constant closer to 0.1 s^-1 for illustrative purposes
Pre_exponential_Factor_A = 1.0e5  # Adjusted from 1.0e10 to 1.0e5
Activation_Energy_Ea = 30000.0  # Adjusted from 50000.0 to 30000.0 Joules/mol
Gas_Constant_R = 8.314  # J/(mol*K)
Temperature_K = 298.15  # Kelvin (25 degrees Celsius)

# Calculate the rate constant k using the Arrhenius equation
# k = A * exp(-Ea / (R * T))
Rate_Constant_k = Pre_exponential_Factor_A * math.exp(-Activation_Energy_Ea / (Gas_Constant_R * Temperature_K))

# Apply UBP modifiers (currently no effect as modifiers are 1.0)
Rate_Constant_k *= UBP_A_FACTOR
Activation_Energy_Ea *= UBP_E_ACTIVATION_MODIFIER

Current_Time = 0.0
Results_Table = []  # List to store (Time, Concentration) tuples

# Add initial state (T=0)
Results_Table.append((Current_Time, R_initial))

# 2. Simulation Loop (Pseudo-Code - using analytical solution for first-order kinetics):
for i in range(1, Num_Steps + 1):
    # Calculate current time
    Current_Time = i * Time_Step_dt

    # Calculate concentration using the analytical solution
    # R(T) = R_initial * exp(-Rate_Constant_k * Current_Time)
    Concentration_R = R_initial * math.exp(-Rate_Constant_k * Current_Time)

    # Record the result
    Results_Table.append((Current_Time, Concentration_R))

# 3. Output Generation:
print("--- Study 2 Simulation Results (with Arrhenius Equation) ---")
print(f"Temperature: {Temperature_K} K")
print(f"Calculated Rate Constant (k): {Rate_Constant_k:.4e} s^-1")
print("\nTime (s) | Concentration (units)")
print("---------|----------------------")
for Time, Concentration in Results_Table:
    print(f"{Time:<9.2f}| {Concentration:<20.4f}")

