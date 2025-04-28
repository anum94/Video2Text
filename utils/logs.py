import os
import wandb
from typing import List
from dotenv import load_dotenv
import numpy as np
load_dotenv()
import matplotlib.pyplot as plt
def wandb_setup():

    wandb_token_key: str = "WANDB_TOKEN"

    # wandb setup
    wandb_tok = os.environ.get(wandb_token_key)
    assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
    wandb.login( key=wandb_tok)

def write_to_wb(run_name, baseline_output:tuple, feedback_output:tuple, icl_output:tuple,  config):
    project_name = "CommGen"
    entity = "anum-afzal-technical-university-of-munich"
    wandb_setup()
    wandb_mode = "online"

    wandb.init(project=project_name, entity=entity, config=config, name=run_name,
           mode=wandb_mode)

    b_eval_metrics = baseline_output[2]
    b_generations = b_eval_metrics["pred_utterences"]
    b_pred_timing =b_eval_metrics["pred_timing"]
    ref_timing = b_eval_metrics["ref_timing"]
    ref_generations = b_eval_metrics["ref_utterences"]

    f_eval_metrics = feedback_output[2]
    f_pred_timing = f_eval_metrics["pred_timing"]
    f_generations = f_eval_metrics["pred_utterences"]

    i_eval_metrics = icl_output[2]
    i_pred_timing = i_eval_metrics["pred_timing"]
    i_generations = i_eval_metrics["pred_utterences"]


    metrics_columns = (
            ["model_name", "# frame", "step"] +
                       [f"baseline_{key}" for key in b_eval_metrics.keys()] +
                       [f"feedback_{key}" for key in f_eval_metrics.keys()] +
                       [f"icl_{key}" for key in i_eval_metrics.keys()]
    )

    metrics_data = (
            [config['model'], config['# frame'], config['step']] +
                    list(b_eval_metrics.values()) +
                    list(f_eval_metrics.values()) +
                    list(i_eval_metrics.values())
                    )
    table = wandb.Table(columns=metrics_columns,data = [metrics_data] )
    wandb.log({"metrics_table": table}, commit=True)

    # Plot Prediction Timing
    b_pred_timing = np.array(b_pred_timing).astype(int)
    f_pred_timing = np.array(f_pred_timing).astype(int)
    i_pred_timing = np.array(i_pred_timing).astype(int)
    ref_timing = np.array(ref_timing).astype(int)

    # Define the indices for the x-axis
    x = np.arange(len(b_pred_timing))

    # Plot the lists
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0,0].plot(x, b_pred_timing, marker='o', label='b_pred_timing', linestyle='-', color='b')
    axs[0,0].legend()

    axs[0,1].plot(x, ref_timing, marker='o', label='ref_timing', linestyle='-', color='g')
    axs[0,1].legend()

    axs[1,0].plot(x, f_pred_timing, marker='o', label='feedback_pred_timing', linestyle='-', color='r')
    axs[1,0].legend()


    axs[1,1].plot(x, i_pred_timing, marker='o', label='icl_pred_timing', linestyle='-', color='y')
    axs[1,1].legend()

    # Add labels and legend
    # plt.yticks([0, 1], ['False', 'True'])
    # plt.xticks(x)

    #plt.title(f"{run_name}")
    plt.legend()

    # Show grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    wandb.log({f"plot_timing_{run_name}": wandb.Image(fig)})

    # Plot Prediction Token Stats
    b_gen_words = np.array([len(gen.split()) for gen in b_generations]).astype(int)
    f_gen_words = np.array([len(gen.split()) for gen in f_generations]).astype(int)
    i_gen_words = np.array([len(gen.split()) for gen in i_generations]).astype(int)
    ref_gen_words = np.array([len(gen.split()) for gen in ref_generations]).astype(int)

    # Define the indices for the x-axis
    x = np.arange(len(b_gen_words))

    # Plot the lists
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].plot(x, b_gen_words, marker='o', label='baseline_gen_words', linestyle='-', color='b')
    axs[0, 0].legend()

    axs[0, 1].plot(x, ref_gen_words, marker='o', label='ref_gen_words', linestyle='-', color='g')
    axs[0, 1].legend()

    axs[1, 0].plot(x, f_gen_words, marker='o', label='feedback_gen_words', linestyle='-', color='r')
    axs[1, 0].legend()

    axs[1, 1].plot(x, i_gen_words, marker='o', label='icl_gen_words', linestyle='-', color='y')
    axs[1, 1].legend()

    # Add labels and legend
    # plt.yticks([0, 1], ['False', 'True'])
    # plt.xticks(x)

    # plt.title(f"{run_name}")
    plt.legend()

    # Show grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    wandb.log({f"plot_worddist_{run_name}": wandb.Image(fig)})

    wandb.finish()
