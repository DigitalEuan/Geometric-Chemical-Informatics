
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_refined_optimization = pd.read_csv(os.path.join(results_dir, 'study15_refined_ubp_operators_results.csv'))

operator_types = df_refined_optimization['UBP_Operator_Type'].unique()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Study 15: Refined UBP Operators Optimization Results (Balanced Weighting)', fontsize=16)

bar_width = 0.2
index = np.arange(len(operator_types))

# Plot Final R Concentration
ax1 = axes[0]
ax1.bar(index, df_refined_optimization['Final_R_Concentration'], bar_width, label='Final R Concentration', color='skyblue')
ax1.set_title('Final R Concentration')
ax1.set_ylabel('Final R Concentration (mol/L)')
ax1.set_xticks(index)
ax1.set_xticklabels(operator_types, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--')

# Plot Temperature Standard Deviation
ax2 = axes[1]
ax2.bar(index, df_refined_optimization['Temperature_Std_Dev'], bar_width, label='Temperature Std Dev', color='lightcoral')
ax2.set_title('Temperature Standard Deviation')
ax2.set_ylabel('Temperature Std Dev (K)')
ax2.set_xticks(index)
ax2.set_xticklabels(operator_types, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plot_path = os.path.join(results_dir, 'study15_refined_ubp_operators_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

