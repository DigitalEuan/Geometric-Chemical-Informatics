
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study4 = pd.read_csv(os.path.join(results_dir, 'study4_dynamic_temperature_results.csv'))

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df_study4['Time (s)'], df_study4['Concentration (mol/L)'], label='R Concentration', color='blue')
plt.title('Study 4: Reactant Concentration with Dynamic Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df_study4['Time (s)'], df_study4['Temperature (K)'], label='Temperature', color='red')
plt.title('Study 4: Temperature Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study4_dynamic_temperature_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

