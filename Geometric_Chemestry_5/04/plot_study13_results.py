
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_weighted_optimization = pd.read_csv(os.path.join(results_dir, 'study13_weighted_optimization_results.csv'))

operator_types = df_weighted_optimization['UBP_Operator_Type'].unique()
weighting_schemes = ['Heavy_Conc', 'Balanced', 'Heavy_Temp'] # Ensure consistent order

fig, axes = plt.subplots(len(operator_types), 2, figsize=(16, 6 * len(operator_types)), sharex=False)
fig.suptitle('Study 13: Multi-objective Optimization with Alternative Weighting Schemes', fontsize=16)

bar_width = 0.25
index = np.arange(len(weighting_schemes))

for i, op_type in enumerate(operator_types):
    df_op_type = df_weighted_optimization[df_weighted_optimization['UBP_Operator_Type'] == op_type]

    # Prepare data for plotting
    final_r_concs = [df_op_type[df_op_type['Weighting_Scheme'] == scheme]['Final_R_Concentration'].iloc[0] for scheme in weighting_schemes]
    temp_std_devs = [df_op_type[df_op_type['Weighting_Scheme'] == scheme]['Temperature_Std_Dev'].iloc[0] for scheme in weighting_schemes]

    # Plot Final R Concentration
    ax1 = axes[i, 0]
    ax1.bar(index, final_r_concs, bar_width, label='Final R Concentration', color='skyblue')
    ax1.set_title(f'{op_type} Operator: Final R Concentration')
    ax1.set_ylabel('Final R Concentration (mol/L)')
    ax1.set_xticks(index)
    ax1.set_xticklabels(weighting_schemes, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--')

    # Plot Temperature Standard Deviation
    ax2 = axes[i, 1]
    ax2.bar(index, temp_std_devs, bar_width, label='Temperature Std Dev', color='lightcoral')
    ax2.set_title(f'{op_type} Operator: Temperature Std Dev')
    ax2.set_ylabel('Temperature Std Dev (K)')
    ax2.set_xticks(index)
    ax2.set_xticklabels(weighting_schemes, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plot_path = os.path.join(results_dir, 'study13_weighted_optimization_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

