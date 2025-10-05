
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_optimization = pd.read_csv(os.path.join(results_dir, 'study8_optimization_results.csv'))

# Create a bar plot for optimized C_rate values
plt.figure(figsize=(10, 6))
plt.bar(df_optimization['UBP_Operator_Type'], df_optimization['Optimized_C_rate'], color=['blue', 'green', 'red'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Optimized C_rate')
plt.title('Study 8: Optimized C_rate for Different UBP Operators')
plt.grid(axis='y', linestyle='--')
plot_path_c_rate = os.path.join(results_dir, 'study8_optimized_c_rate_plot.png')
plt.savefig(plot_path_c_rate)
print(f"Plot saved to {plot_path_c_rate}")

# Create a bar plot for optimized M_constant values
plt.figure(figsize=(10, 6))
plt.bar(df_optimization['UBP_Operator_Type'], df_optimization['Optimized_M_constant'], color=['cyan', 'magenta', 'yellow'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Optimized M_constant')
plt.title('Study 8: Optimized M_constant for Different UBP Operators')
plt.grid(axis='y', linestyle='--')
plot_path_m_constant = os.path.join(results_dir, 'study8_optimized_m_constant_plot.png')
plt.savefig(plot_path_m_constant)
print(f"Plot saved to {plot_path_m_constant}")

# Create a bar plot for final R concentrations
plt.figure(figsize=(10, 6))
plt.bar(df_optimization['UBP_Operator_Type'], df_optimization['Final_R_Concentration'], color=['purple', 'orange', 'brown'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Final R Concentration (mol/L)')
plt.title('Study 8: Final R Concentration with Optimized UBP Operators')
plt.grid(axis='y', linestyle='--')
plot_path_final_r = os.path.join(results_dir, 'study8_optimized_final_r_plot.png')
plt.savefig(plot_path_final_r)
print(f"Plot saved to {plot_path_final_r}")

