
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_sensitivity = pd.read_csv(os.path.join(results_dir, 'study12_sensitivity_analysis_results.csv'))

operator_types = df_sensitivity['Operator_Type'].unique()

fig, axes = plt.subplots(len(operator_types), 2, figsize=(16, 6 * len(operator_types)), sharex=True)
fig.suptitle('Study 12: Sensitivity Analysis of UBP Operator Parameters', fontsize=16)

for i, op_type in enumerate(operator_types):
    df_op = df_sensitivity[df_sensitivity['Operator_Type'] == op_type]

    # Plot Final R Concentration
    ax1 = axes[i, 0]
    for m_const in df_op['M_constant'].unique():
        df_m = df_op[df_op['M_constant'] == m_const]
        ax1.plot(df_m['C_rate'], df_m['Final_R_Concentration'], label=f'M_constant={m_const:.1f}')
    ax1.set_title(f'{op_type} Operator: Final R Concentration vs. C_rate')
    ax1.set_ylabel('Final R Concentration (mol/L)')
    ax1.legend()
    ax1.grid(True)

    # Plot Temperature Standard Deviation
    ax2 = axes[i, 1]
    for m_const in df_op['M_constant'].unique():
        df_m = df_op[df_op['M_constant'] == m_const]
        ax2.plot(df_m['C_rate'], df_m['Temperature_Std_Dev'], label=f'M_constant={m_const:.1f}')
    ax2.set_title(f'{op_type} Operator: Temperature Std Dev vs. C_rate')
    ax2.set_ylabel('Temperature Std Dev (K)')
    ax2.legend()
    ax2.grid(True)

axes[-1, 0].set_xlabel('C_rate')
axes[-1, 1].set_xlabel('C_rate')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plot_path = os.path.join(results_dir, 'study12_sensitivity_analysis_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

