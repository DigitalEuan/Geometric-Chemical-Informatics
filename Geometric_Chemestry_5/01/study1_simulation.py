import math

# 1. Initialization:
R_initial = 100.0  # Initial concentration in units
Rate_Constant_k = 0.1  # Rate constant in s^-1
Time_Step_dt = 1.0  # Time step duration in seconds
Num_Steps = 10  # Number of simulation steps

Current_Time = 0.0
Results_Table = []  # List to store (Time, Concentration) tuples

# Add initial state (T=0)
Results_Table.append((Current_Time, R_initial))

# 2. Simulation Loop (Pseudo-Code):
for i in range(1, Num_Steps + 1):
    # Calculate current time
    Current_Time = i * Time_Step_dt

    # Calculate concentration using the analytical solution
    # R(T) = R_initial * exp(-Rate_Constant_k * Current_Time)
    Concentration_R = R_initial * math.exp(-Rate_Constant_k * Current_Time)

    # Record the result
    Results_Table.append((Current_Time, Concentration_R))

# 3. Output Generation:
print("Time (s) | Concentration (units)")
print("---------|----------------------")
for Time, Concentration in Results_Table:
    print(f"{Time:<9.2f}| {Concentration:<20.4f}")

