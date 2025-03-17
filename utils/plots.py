import matplotlib.pyplot as plt
import numpy as np

# Define the lists
L1 = [True, False, False, False]
L2 = [False, True, True, False]

# Convert to integers for plotting
L1_int = np.array(L1).astype(int)
L2_int = np.array(L2).astype(int)

# Define the indices for the x-axis
x = np.arange(len(L1))

# Plot the lists
plt.figure(figsize=(8, 2))
plt.plot(x, L1_int, marker='o', label='L1', linestyle='-', color='b')
plt.plot(x, L2_int, marker='s', label='L2', linestyle='--', color='r')

# Add labels and legend
plt.yticks([0, 1], ['False', 'True'])
plt.xticks(x)
plt.xlabel('Index')
plt.title('Element-wise Plot')
plt.legend()

# Show grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()