
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study9 = pd.read_csv(os.path.join(results_dir, 'study9_coupled_ubp_operators_results.csv'))

plt.figure(figsize=(14, 12))

# Plot Concentrations
plt.subplot(3, 1, 1)
plt.plot(df_study9['Time (s)'], df_study9['A Concentration (mol/L)'], label='A Concentration', color='blue')
plt.plot(df_study9['Time (s)'], df_study9['I Concentration (mol/L)'], label='I Concentration', color='orange')
plt.plot(df_study9['Time (s)'], df_study9['P Concentration (mol/L)'], label='P Concentration', color='green')
plt.title('Study 9: Concentrations with Coupled UBP Operators in Multi-step Reversible Mechanism')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

# Plot Temperature
plt.subplot(3, 1, 2)
plt.plot(df_study9['Time (s)'], df_study9['Temperature (K)'], label='Temperature', color='red')
plt.title('Study 9: Temperature Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

# Plot Rate Constants
plt.subplot(3, 1, 3)
plt.plot(df_study9['Time (s)'], df_study9['kf1 (s^-1)'], label='kf1 (A<=>I, forward)', color='purple')
plt.plot(df_study9['Time (s)'], df_study9['kr1 (s^-1)'], label='kr1 (A<=>I, reverse)', color='magenta')
plt.plot(df_study9['Time (s)'], df_study9['kf2 (s^-1)'], label='kf2 (I<=>P, forward)', color='brown')
plt.plot(df_study9['Time (s)'], df_study9['kr2 (s^-1)'], label='kr2 (I<=>P, reverse)', color='cyan')
plt.title('Study 9: Rate Constants Over Time with Coupled UBP Operators')
plt.xlabel('Time (s)')
plt.ylabel('Rate Constant (s^-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study9_coupled_ubp_operators_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

