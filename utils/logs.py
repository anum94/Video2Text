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

def write_to_wb(run_name, baseline_output:tuple, feedback_output:tuple, icl_output:tuple, realtime_output:tuple, config, WB = True):
    if WB:
        project_name = "CommGen"
        entity = "anum-afzal-technical-university-of-munich"
        wandb_setup()
        wandb_mode = "online"

        wandb.init(project=project_name, entity=entity, config=config, name=run_name,
               mode=wandb_mode)

    b_eval_metrics = baseline_output[2]
    b_generations = baseline_output[0]
    b_pred_timing =b_eval_metrics["pred_timing"]
    ref_timing = b_eval_metrics["ref_timing"]
    ref_generations = baseline_output[3]

    f_eval_metrics = feedback_output[2]
    f_pred_timing = f_eval_metrics["pred_timing"]
    f_generations = feedback_output[0]

    r_eval_metrics = realtime_output[2]
    r_pred_timing = r_eval_metrics["pred_timing"]
    r_generations = realtime_output[0]

    i_eval_metrics = icl_output[2]
    i_pred_timing = i_eval_metrics["pred_timing"]
    i_generations = icl_output[0]

    additional_columns = ["model_name", "sample","# frame", "step"]
    metrics_columns = (
            additional_columns +
                       [f"baseline_{key}" for key in b_eval_metrics.keys()] +
                       [f"feedback_{key}" for key in f_eval_metrics.keys()] +
                        [f"realtime_{key}" for key in r_eval_metrics.keys()] +
                       [f"icl_{key}" for key in i_eval_metrics.keys()]
    )

    metrics_data = (
            [config['model'], run_name, config['# frame'], config['step']] +
                    list(b_eval_metrics.values()) +
                    list(f_eval_metrics.values()) +
                    list(r_eval_metrics.values()) +
                    list(i_eval_metrics.values())
                    )

    if WB:
        table = wandb.Table(columns=metrics_columns,data = [metrics_data] )
        wandb.log({"metrics_table": table}, commit=True)

        # Plot Prediction Timing
        b_pred_timing = np.array(b_pred_timing).astype(int)
        f_pred_timing = np.array(f_pred_timing).astype(int)
        r_pred_timing = np.array(r_pred_timing).astype(int)
        i_pred_timing = np.array(i_pred_timing).astype(int)
        ref_timing = np.array(ref_timing).astype(int)

        # Define the indices for the x-axis
        x = np.arange(len(b_pred_timing))

        # Plot the lists
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        axs[0,0].plot(x, b_pred_timing, marker='o', label='b_pred_timing', linestyle='-', color='b')
        axs[0,0].legend()

        axs[0,1].plot(x, ref_timing, marker='o', label='ref_timing', linestyle='-', color='g')
        axs[0,1].legend()

        axs[1,0].plot(x, f_pred_timing, marker='o', label='feedback_pred_timing', linestyle='-', color='r')
        axs[1,0].legend()


        axs[1,1].plot(x, i_pred_timing, marker='o', label='icl_pred_timing', linestyle='-', color='y')
        axs[1,1].legend()

        #print (len(x), len(r_pred_timing))
        x = np.arange(len(r_pred_timing))
        #print(len(x), len(r_pred_timing))

        axs[2, 0].plot(x, r_pred_timing, marker='o', label='realtime_pred_timing', linestyle='-', color='m')
        axs[2, 0].legend()

        # Add labels and legend
        # plt.yticks([0, 1], ['False', 'True'])
        # plt.xticks(x)

        plt.title(f"{run_name}")
        plt.legend()

        # Show grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        wandb.log({f"plot_timing": wandb.Image(fig)})

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

        plt.title(f"{run_name}")
        plt.legend()

        # Show grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        wandb.log({f"plot_word_dist": wandb.Image(fig)})

        wandb.finish()
    metrics = dict(zip(metrics_columns, metrics_data))
    additional_columns+= ["baseline_ref_timing", "baseline_pred_timing", "feedback_ref_timing",
                          "feedback_pred_timing", "icl_ref_timing", "icl_pred_timing",
                          "feedback_bins", "realtime_ref_timing", "realtime_pred_timing",
                          "baseline_bins", "icl_bins" , "realtime_bins"]
    for k,v in metrics["feedback_bins"].items():
        metrics[f"feedback_{k}"] = v
    for k,v in metrics["baseline_bins"].items():
        metrics[f"baseline_{k}"] = v
    for k,v in metrics["icl_bins"].items():
        metrics[f"icl_{k}"] = v
    for k,v in metrics["realtime_bins"].items():
        metrics[f"realtime_{k}"] = v
    return {k: v for k, v in metrics.items() if k not in additional_columns}
