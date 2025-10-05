
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study6 = pd.read_csv(os.path.join(results_dir, 'study6_multi_step_mechanism_results.csv'))

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(df_study6['Time (s)'], df_study6['A Concentration (mol/L)'], label='A Concentration', color='blue')
plt.plot(df_study6['Time (s)'], df_study6['I Concentration (mol/L)'], label='I Concentration', color='orange')
plt.plot(df_study6['Time (s)'], df_study6['P Concentration (mol/L)'], label='P Concentration', color='green')
plt.title('Study 6: Concentrations in a Multi-step Reaction Mechanism')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df_study6['Time (s)'], df_study6['Temperature (K)'], label='Temperature', color='red')
plt.title('Study 6: Temperature Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df_study6['Time (s)'], df_study6['k1 (s^-1)'], label='k1 (A->I)', color='purple')
plt.plot(df_study6['Time (s)'], df_study6['k2 (s^-1)'], label='k2 (I->P)', color='brown')
plt.title('Study 6: Rate Constants for Elementary Steps Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Rate Constant (s^-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study6_multi_step_mechanism_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

