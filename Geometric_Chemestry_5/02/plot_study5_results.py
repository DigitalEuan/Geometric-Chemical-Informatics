
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study5 = pd.read_csv(os.path.join(results_dir, 'study5_reversible_reactions_results.csv'))

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(df_study5['Time (s)'], df_study5['R Concentration (mol/L)'], label='R Concentration', color='blue')
plt.plot(df_study5['Time (s)'], df_study5['P Concentration (mol/L)'], label='P Concentration', color='green')
plt.title('Study 5: Reactant and Product Concentrations (Reversible Reaction)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df_study5['Time (s)'], df_study5['Temperature (K)'], label='Temperature', color='red')
plt.title('Study 5: Temperature Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df_study5['Time (s)'], df_study5['kf (s^-1)'], label='kf (forward rate constant)', color='purple')
plt.plot(df_study5['Time (s)'], df_study5['kr (s^-1)'], label='kr (reverse rate constant)', color='orange')
plt.title('Study 5: Forward and Reverse Rate Constants Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Rate Constant (s^-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study5_reversible_reactions_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

