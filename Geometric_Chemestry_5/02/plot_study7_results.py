
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study7 = pd.read_csv(os.path.join(results_dir, 'study7_stochastic_effects_results.csv'))

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(df_study7['Time (s)'], df_study7['R Concentration (mol/L)'], label='R Concentration', color='blue')
plt.plot(df_study7['Time (s)'], df_study7['P Concentration (mol/L)'], label='P Concentration', color='green')
plt.title('Study 7: Reactant and Product Concentrations with Stochastic Effects')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df_study7['Time (s)'], df_study7['Temperature (K)'], label='Temperature', color='red')
plt.title('Study 7: Temperature Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df_study7['Time (s)'], df_study7['kf (s^-1)'], label='kf (forward rate constant)', color='purple')
plt.plot(df_study7['Time (s)'], df_study7['kr (s^-1)'], label='kr (reverse rate constant)', color='orange')
plt.title('Study 7: Forward and Reverse Rate Constants with Stochastic Effects')
plt.xlabel('Time (s)')
plt.ylabel('Rate Constant (s^-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study7_stochastic_effects_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

