import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('covtype_results/table.csv')

# Create a combined label for each method and balancing strategy
df['label'] = df['method'] + '_' + df['query_balancing']

# Determine sorted unique process counts for x-ticks
xticks = sorted(df['num_procs'].unique())

# Plot runtime vs number of processes for each label on log–log scales
plt.figure(figsize=(8,6))
for label, grp in df.groupby('label'):
    grp_sorted = grp.sort_values('num_procs')
    plt.plot(grp_sorted['num_procs'],
             grp_sorted['runtime'],
             marker='o',
             linestyle='-',
             label=label)

plt.xscale('log')
plt.yscale('log')

# Set ticks at each distinct proc count
plt.xticks(xticks, xticks)

plt.xlabel('Number of Processes (log scale)')
plt.ylabel('Runtime (s, log scale)')
plt.title('Strong Scaling Plot (Log–Log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Method – Balancing', loc='best')
plt.tight_layout()

plt.savefig("covtype.strong_scaling.png", dpi=200)
#  plt.show()
