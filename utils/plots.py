import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import pandas as pd

from main import read_srt

def get_srt_files(directory):
    # Use glob to find all .json files in the specified directory
    json_files = glob.glob(os.path.join(directory, '*.srt'))
    return json_files
def plot_fig(pred_timing, ref_timing):
    # Convert to integers for plotting
    pred_timing = np.array(pred_timing).astype(int)
    ref_timing = np.array(ref_timing).astype(int)

    # Define the indices for the x-axis
    x = np.arange(len(pred_timing))

    # Plot the lists
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(x, pred_timing, marker='o', label='pred_timing', linestyle='-', color='b')
    axs[0].set_title('')
    axs[0].legend()

    axs[1].plot(x, ref_timing, marker='o', label='ref_timing', linestyle='-', color='g')
    axs[1].set_title('')
    axs[1].legend()

    # Add labels and legend
    # plt.yticks([0, 1], ['False', 'True'])
    # plt.xticks(x)

    plt.title(f"{os.path.basename(logs_file).replace('.csv', '')}")
    plt.legend()

    # Show grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

logs_file = "/Users/anumafzal/PycharmProjects/video2Text/utils/plot_data/pred_timing.csv"
df = pd.read_csv(logs_file)
print (df['baseline_pred_timing'])
baseline_pred_timing = [str(i) for i in df['baseline_pred_timing'][0].split(',')]
ref_timing = [str(i)  for i in df['baseline_ref_timing'][0].split(',')]

baseline_pred_timing = [1 if i == 'true' else 0 for i in baseline_pred_timing]
ref_timing = [1 if i == 'true' else 0 for i in ref_timing]

plot_fig(baseline_pred_timing, ref_timing)

