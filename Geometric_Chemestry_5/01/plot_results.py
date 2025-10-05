
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the results directory
results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"

# Load the data from CSV files
df_study1 = pd.read_csv(os.path.join(results_dir, 'study1_results.csv'))
df_study2 = pd.read_csv(os.path.join(results_dir, 'study2_results.csv'))
df_study3_quadratic = pd.read_csv(os.path.join(results_dir, 'study3_quadratic_results.csv'))
df_study3_linear = pd.read_csv(os.path.join(results_dir, 'study3_linear_results.csv'))
df_study3_compositional = pd.read_csv(os.path.join(results_dir, 'study3_compositional_results.csv'))

# Create the plot
plt.figure(figsize=(12, 8))

plt.plot(df_study1['Time (s)'], df_study1['Concentration (units)'], label='Study 1 (Basic Kinetics)', marker='o')
plt.plot(df_study2['Time (s)'], df_study2['Concentration (units)'], label='Study 2 (Arrhenius)', marker='x')
plt.plot(df_study3_linear['Time (s)'], df_study3_linear['Concentration (units)'], label='Study 3 (UBP Linear)', marker='s')
plt.plot(df_study3_quadratic['Time (s)'], df_study3_quadratic['Concentration (units)'], label='Study 3 (UBP Quadratic)', marker='^')
plt.plot(df_study3_compositional['Time (s)'], df_study3_compositional['Concentration (units)'], label='Study 3 (UBP Compositional)', marker='d')

plt.title('Chemical Reaction Kinetics: Concentration Decay Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (units)')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(results_dir, 'concentration_decay_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

