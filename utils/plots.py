import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
def get_json_files(directory):
    # Use glob to find all .json files in the specified directory
    json_files = glob.glob(os.path.join(directory, '*.json'))
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

    plt.title(f"{os.path.basename(log).replace('.json', '')}_{logs_folder.split('/')[-1]}")
    plt.legend()

    # Show grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

logs_folder = "/Users/anumafzal/PycharmProjects/video2Text/logs/llava-hf_LLaVA-NeXT-Video-7B-hf/step_5"
logs = get_json_files(logs_folder)
for log in logs:
    with open(log, 'r') as file:
        data = json.load(file)
    print (f"File: {log}")
    print (f"ROUGE: {data['rouge']}")
    print(f"BLUE: {data['blue']}")
    pred_timing = data['pred_timing']
    ref_timing = data['ref_timing']

    plot_fig(pred_timing, ref_timing)

