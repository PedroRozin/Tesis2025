import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your .dat file
path = '/home/pedrorozin/scripts/class_public/output/test_delta_prime_01_perturbations_k0_s.dat'

# Read the .dat file, skipping comment lines and using whitespace as delimiter
df = pd.read_csv(
    path,
    comment='#',
    delim_whitespace=True,
    header=None,
    skiprows=1  # Adjust if your header is not on the third line
)

# Optionally, set column names from the header line in your .dat file
with open(path) as f:
    for line in f:
        if line.startswith('#') and ':' in line:
            columns = [col.split(':')[1].strip() for col in line[1:].split('  ') if ':' in col]
            df.columns = columns
            break
#rename tau
df.rename(columns={'tau [Mpc]': 'tau'}, inplace=True)
#keep columns
df = df[['tau', 'delta_b', 'delta_cdm','a',
          'delta_prime_b', 'delta_prime_cdm']]

fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex= True)
# Plot delta_b and delta_cdm vs a in the first subplot
#plot delta_prime_b and delta_prime_cdm vs a in the second subplot
axs[0].plot(df['a'], df['delta_b'], label='delta_b', color='blue')
axs[0].plot(df['a'], df['delta_cdm'], label='delta_cdm', color='red')
#logscale in x
axs[0].set_xscale('log')
axs[0].set_ylabel('delta')
# axs[0].set_title('delta_b and delta_cdm vs a')
axs[0].legend()
axs[0].grid()
#axs[0].set_xlim(0, 1)
axs[1].plot(df['a'], df['delta_prime_b'], label='delta_prime_b', color='blue')
axs[1].plot(df['a'], df['delta_prime_cdm'], label='delta_prime_cdm', color='red')
axs[1].set_xscale('log')
axs[1].set_xlabel('a')
axs[1].set_ylabel('delta_prime')
axs[1].legend()
axs[1].grid()

#save figure
plt.savefig('delta_prime_b_cdm_vs_a.png', dpi=300)
plt.show()
# axs[1].set_title('delta_prime_b and delta_prime_cdm vs a')

