import os

import numpy as np
import pandas as pd
import shutil
from main import sample_frames
SAMPLES_PER_MODEL = 2
VIDEO_DIR = "RaceCommentary/recordings"
model_dict = {"llava-hf_LLaVA-NeXT-Video-7B-hf": "M1"}
srt_dict = {"baseline_srt": "G1.srt", "realtime_srt": "G2.srt", "icl_srt": "G3.srt"}
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


logs_directory = "/Users/anumafzal/PycharmProjects/video2Text/logs/logs"
logs_list = []
for file_path in findDirWithFileInLevel(logs_directory, 3):
     sample_dict = {}
     nested_paths = file_path.split('/')
     sample_dict["model"] = nested_paths[-3]
     config = nested_paths[-1].split("_")
     sample_dict["step"], sample_dict["frames_used"], sample_dict["k"] = config[1], config[3], config[5]
     sample_dict["sample"] = nested_paths[-2]
     sample_dict["video_path"] = get_video_path(sample_dict["sample"])
     FT = True if "FT" in "model" else False
     for srt_file in os.listdir(file_path):
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
evaluation_metrics = ["KEI", "WAIT-NESS", "Naturalness", "Logical_Coherence"]
df = pd.DataFrame(logs_list).dropna()
df = df.loc[(df['step'] == 2) & (df['frames_used'] == 1) & (df['k'].isin([6,8]))]
df_samples = df.groupby('sample')
excel_columns = []
samples = []
for sample_name, group_sample in df_samples:
    if len(group_sample) >= SAMPLES_PER_MODEL:
        samples.append(sample_name)
        print(sample_name)
        group_sample = group_sample.head(SAMPLES_PER_MODEL)
        eval_samples_dir = os.path.join("evaluation_samples", sample_name)
        os.makedirs(eval_samples_dir, exist_ok=True)

        # Copy video into the directory
        source = group_sample.iloc[0]['video_path']
        destination = os.path.join(eval_samples_dir, os.path.basename(source))
        dest = shutil.copyfile(source, destination)

        # iteration over each model generations
        df_models = df.groupby('model')
        for model_name, group_model in df_models:
            eval_model_dir = os.path.join(eval_samples_dir, model_dict[model_name])
            os.makedirs(eval_model_dir, exist_ok=True)
            # Copy srt files into the respective sample/model directory

            # Baseline
            srt_mode = 'baseline_srt'
            source = group_model.iloc[0][srt_mode]
            destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode]}")
            #dest = shutil.copyfile(source, destination)
            prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
            eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
            excel_columns += eval_col

            # Realtime
            srt_mode = 'realtime_srt'
            srt_mode = 'icl_srt'
            source = group_model.iloc[0][srt_mode]
            destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode]}")
            #dest = shutil.copyfile(source, destination)
            prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
            eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
            excel_columns += eval_col
eval_df = df = pd.DataFrame(0, index=np.arange(len(samples)), columns=eval_col)
eval_df["sample"] = samples
print(eval_df)









