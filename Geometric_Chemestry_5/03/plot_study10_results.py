
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = "/home/ubuntu/ChemicalReactionKinetics/results/"
df_study10 = pd.read_csv(os.path.join(results_dir, 'study10_gillespie_algorithm_results.csv'))

plt.figure(figsize=(12, 6))

plt.plot(df_study10['Time (s)'], df_study10['R Concentration (mol/L)'], label='R Concentration (Gillespie)', color='blue', drawstyle='steps-post')
plt.plot(df_study10['Time (s)'], df_study10['P Concentration (mol/L)'], label='P Concentration (Gillespie)', color='green', drawstyle='steps-post')

plt.title('Study 10: Reactant and Product Concentrations (Gillespie Algorithm)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'study10_gillespie_algorithm_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

