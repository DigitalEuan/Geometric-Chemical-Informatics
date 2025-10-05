
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_optimization_multi = pd.read_csv(os.path.join(results_dir, 'study11_multi_objective_optimization_results.csv'))

plt.figure(figsize=(14, 8))

# Plot Optimized C_rate
plt.subplot(2, 2, 1)
plt.bar(df_optimization_multi['UBP_Operator_Type'], df_optimization_multi['Optimized_C_rate'], color=['blue', 'green', 'red'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Optimized C_rate')
plt.title('Study 11: Optimized C_rate for Multi-objective Optimization')
plt.grid(axis='y', linestyle='--')

# Plot Optimized M_constant
plt.subplot(2, 2, 2)
plt.bar(df_optimization_multi['UBP_Operator_Type'], df_optimization_multi['Optimized_M_constant'], color=['cyan', 'magenta', 'yellow'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Optimized M_constant')
plt.title('Study 11: Optimized M_constant for Multi-objective Optimization')
plt.grid(axis='y', linestyle='--')

# Plot Final R Concentration
plt.subplot(2, 2, 3)
plt.bar(df_optimization_multi['UBP_Operator_Type'], df_optimization_multi['Final_R_Concentration'], color=['purple', 'orange', 'brown'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Final R Concentration (mol/L)')
plt.title('Study 11: Final R Concentration with Multi-objective Optimization')
plt.grid(axis='y', linestyle='--')

# Plot Temperature Standard Deviation
plt.subplot(2, 2, 4)
plt.bar(df_optimization_multi['UBP_Operator_Type'], df_optimization_multi['Temperature_Std_Dev'], color=['gray', 'black', 'lime'])
plt.xlabel('UBP Operator Type')
plt.ylabel('Temperature Std Dev (K)')
plt.title('Study 11: Temperature Std Dev with Multi-objective Optimization')
plt.grid(axis='y', linestyle='--')

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study11_multi_objective_optimization_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

