import os

import numpy as np
import pandas as pd
import shutil
import argparse
SAMPLES_PER_MODEL = 2

model_dict = {"llava-hf_LLaVA-NeXT-Video-7B-hf": "M1",
              "gpt-4o-mini-2024-07-18": "M3",
                "Qwen_Qwen2.5-VL-7B-Instruct": "M2",
              "anumafzal94_LLaVa-NeXT-Video-_step_2_frames_1_n_40000": "M4"
              }

srt_dict = { "realtime_srt": "G1.srt", "icl_srt": "G2.srt", "ft_str": "G3.srt"}
def findDirWithFileInLevel(path, level=3):
    c = path.count(os.sep)
    for root, dirs, files in os.walk(path):
        for name in files:
            if root.count(os.sep) - c - 1 <= level:
                yield root
def get_video_path(sample):
    game_path = os.path.join(VIDEO_DIR, sample)
    try:
        mp4_file = [os.path.join(game_path, file) for file in os.listdir(game_path) if
                    file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
        return mp4_file
    except:
        #print(sample)
        return np.nan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Prepares samples for Human Evaluation from previously conducted experiments."
    )

    parser.add_argument("--video_dir", required=True, type=str, help="directoty with all the videos")
    parser.add_argument("--logs_dir", required=True, type=str, help="log directoty with all .srt files")
    parser.add_argument("--ds", required=True, type=str, help="name of the dataset")


    args = parser.parse_args()

    VIDEO_DIR = args.video_dir
    logs_directory = args.logs_dir
    ds = args.ds

    logs_list = []
    for file_path in findDirWithFileInLevel(logs_directory, 3):
         sample_dict = {}
         nested_paths = file_path.split('/')
         if len(nested_paths) < 6 and ds not in nested_paths:
             continue
         #print(nested_paths)
         sample_dict["model"] = nested_paths[-3]
         config = nested_paths[-1].split("_")
         sample_dict["step"], sample_dict["frames_used"], sample_dict["k"] = config[1], config[3], config[5]
         sample_dict["sample"] = nested_paths[-2]
         sample_dict["video_path"] = get_video_path(sample_dict["sample"])


         for srt_file in os.listdir(file_path):
             sample_dict["realtime_srt"] = "NA"
             sample_dict["icl_srt"] = "NA"
             sample_dict["feedback_srt"] = "NA"
    else:
        sample_dict["baseline_srt"] = os.path.join(file_path, srt_file)

        if ".srt" in srt_file:
            if "realtime" in srt_file:
                sample_dict["realtime_srt"] = os.path.join(file_path,srt_file)
            elif "icl" in srt_file:
                sample_dict["icl_srt"] = os.path.join(file_path,srt_file)
            elif "feedback" in srt_file:
                sample_dict["feedback_srt"] = os.path.join(file_path,srt_file)
            else:
                sample_dict["baseline_srt"] = os.path.join(file_path,srt_file)

        logs_list.append(sample_dict)
        #print(sample_dict)
    evaluation_metrics = ["KEI", "WAIT-NESS", "Naturalness", "Logical_Coherence"]
    print(len(logs_list))
    df = pd.DataFrame(logs_list)#.dropna()
    df = df[((df['step'] == '2') & (df['frames_used'] == '1')) & (df['k'] == '8') ]

    print (len(df))
    df_samples = df.groupby('sample')
    excel_columns = []
    samples = []
    for sample_name, group_sample in df_samples:
        if len(group_sample) >= SAMPLES_PER_MODEL:
            samples.append(sample_name)
            print(sample_name)
            group_sample = group_sample.head(SAMPLES_PER_MODEL)
            eval_samples_dir = os.path.join("evaluation_samples", ds, sample_name)
            os.makedirs(eval_samples_dir, exist_ok=True)

            # Copy video into the directory
            source = group_sample.iloc[0]['video_path']
            destination = os.path.join(eval_samples_dir, os.path.basename(source))
            dest = shutil.copyfile(source, destination)

            # iteration over each model generations
            df_models = df.groupby('model')
            for model_name, group_model in df_models:
                print (model_name)
                group_model.to_excel(f"{model_name}.xlsx")
                if "anumafzal94" in model_name:
                    eval_model_dir = os.path.join(eval_samples_dir, model_dict[model_name])
                    os.makedirs(eval_model_dir, exist_ok=True)

                    srt_mode = 'feedback_srt'
                    source = group_model.iloc[0][srt_mode]
                    destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode]}")
                    #dest = shutil.copyfile(source, destination)
                    prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                    eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                    excel_columns += eval_col
                else:
                    eval_model_dir = os.path.join(eval_samples_dir, model_dict[model_name])
                    os.makedirs(eval_model_dir, exist_ok=True)
                    # Copy srt files into the respective sample/model directory

                    # Baseline
                    srt_mode = 'icl_srt'
                    source = group_model.iloc[0][srt_mode]
                    destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode]}")
                    #dest = shutil.copyfile(source, destination)
                    prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                    eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                    excel_columns += eval_col

                    # Realtime
                    srt_mode = 'realtime_srt'
                    source = group_model.iloc[0][srt_mode]
                    destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode]}")
                    #dest = shutil.copyfile(source, destination)
                    prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                    eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                    excel_columns += eval_col

    eval_df = pd.DataFrame(0, index=np.arange(len(samples)), columns=excel_columns)
    eval_df["sample"] = samples
    print(eval_df)
    eval_df.to_excel("evaluation_samples.xlsx")









