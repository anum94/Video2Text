
import os
from datetime import datetime

import numpy as np
import pandas as pd
import shutil
import argparse
import random
import time
SAMPLES_PER_MODEL = 10
from utils.video_utils import get_video_info
model_dict = {"llava-hf_LLaVA-NeXT-Video-7B-hf": "M1",
              "gpt-4o-mini-2024-07-18": "M3",
                "Qwen_Qwen2.5-VL-7B-Instruct": "M2",
              "anumafzal94_LLaVa-NeXT-Video-_step_2_frames_1_n_40000": "M4",
              "llava-hf_LLaVA-NeXT-Video-7B-hfFP": "M5"
              }

srt_dict = { "realtime_srt": "G1.srt", "icl_srt": "G2.srt", "feedback_srt": "G3.srt"}

import copy

#from ffmpeg import FFmpeg
import webvtt  # https://github.com/glut23/webvtt-py
from webvtt.models import Timestamp


def secs_to_timestamp(secs):
    if secs is None:
        return None
    hours, secs = divmod(secs, 3600)
    mins, secs = divmod(secs, 60)
    secs, msecs = divmod(secs, 1)
    return str(Timestamp(int(hours), int(mins), int(secs), int(msecs * 1000)))


def timestamp_to_secs(ts):
    ts = Timestamp.from_string(ts)
    return ts.hours * 3600 + ts.minutes * 60 + ts.seconds + ts.milliseconds / 1000.0


def cut_video(video_in, video_out, start, end=None):
    t_opt = { "t": end - start } if end else {}
    out_opts = {
        "ss": start,
        **t_opt,
    }

    ffmpeg = FFmpeg()
    ffmpeg = ffmpeg.option('y')
    ffmpeg = ffmpeg.input(
        video_in,
    )
    ffmpeg = ffmpeg.output(
        video_out,
        # map=['0'],
        ss=start,
        **t_opt,
        c='copy',
    )
    ffmpeg.execute()


def cut_subtitles(subs_in, subs_out, start, end=None):
    all_subs = webvtt.from_srt(subs_in)
    subs = webvtt.WebVTT()
    for orig_caption in all_subs.iter_slice(secs_to_timestamp(start), secs_to_timestamp(end)):
        caption = copy.copy(orig_caption)
        caption.start = orig_caption.start and str(secs_to_timestamp(timestamp_to_secs(orig_caption.start) - start))
        caption.end = orig_caption.end and str(secs_to_timestamp(timestamp_to_secs(orig_caption.end) - start))
        subs.captions.append(caption)
    subs.save_as_srt(subs_out)


def cut(video_in, subs_in, video_out, subs_out, start, end=None):
    cut_video(video_in, video_out, start, end)
    cut_subtitles(subs_in, subs_out, start, end)


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
         print (nested_paths)
         if len(nested_paths) < 6 and ds not in nested_paths:
             continue
         #print(nested_paths)
         sample_dict["model"] = nested_paths[-3]

         config = nested_paths[-1].split("_")
         sample_dict["step"], sample_dict["frames_used"], sample_dict["k"] = config[1], config[3], config[5]
         sample_dict["sample"] = nested_paths[-2]
         sample_dict["video_path"] = get_video_path(sample_dict["sample"])


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

                ti_m = os.path.getmtime(os.path.join(file_path,srt_file))
                t_obj = time.ctime(ti_m)
                sample_dict["time"] = datetime.strptime(t_obj, '%a %b %d %H:%M:%S %Y')

         if "anumafzal94" in sample_dict["model"]:
            sample_dict["realtime_srt"] = "NA"
            sample_dict["icl_srt"] = "NA"
            sample_dict["baseline_srt"] = "NA"


         logs_list.append(sample_dict)
        #print(sample_dict)
    evaluation_metrics = ["KEI", "WAIT-NESS", "Naturalness", "Coherence"]
    df = pd.DataFrame(logs_list)#.dropna()
    df = df.dropna(subset=['icl_srt'])
    df = df.dropna(subset=['feedback_srt'])
    df = df.dropna(subset=['realtime_srt'])
    df = df.dropna(subset=['video_path'])


    df = df[((df['step'] == '2') & (df['frames_used'] == '1')) & (df['k'].isin(['8', '0'])) ]
    df.sample(frac=1)

    df_samples = df.groupby('sample')
    excel_columns = []
    samples = []
    for sample_name, group_sample in df_samples:
        if len(samples) == 10:
            break


        eval_samples_dir = os.path.join("evaluation_samples", ds, sample_name)
        os.makedirs(eval_samples_dir, exist_ok=True)

        # Copy video into the directory
        source = group_sample.iloc[0]['video_path']
        video_metadata = get_video_info(source)
        start = random.randint(0,video_metadata["duration"]-11)
        end = start + 10
        destination = os.path.join(eval_samples_dir, f"{os.path.basename(source).replace('.mp4', f'_{start}-{end}.mp4')}")
        #cut_video(video_in=source,video_out=destination,start=start,end=end)

        # iteration over each model generations
        df_models = group_sample.groupby('model')
        if len(df_models) < 4:
            continue
        samples.append(sample_name)
        for model_name, group_model in df_models:
            group_model = group_model.sort_values(by='time', ascending=False)

            if "anumafzal94" in model_name and "LLaVa" in model_name:
                #This is a fine-tuned llava model so we would handle this case separately
                eval_model_dir = os.path.join(eval_samples_dir, model_dict[model_name])
                os.makedirs(eval_model_dir, exist_ok=True)

                srt_mode = 'feedback_srt' #for fetching the correct srt file
                source = group_model.iloc[0][srt_mode]
                destination = os.path.join(eval_model_dir, f"{srt_dict[srt_mode].replace('.srt', f'_{start}-{end}.srt')}")
                cut_subtitles(subs_in=source, subs_out=destination,start=start,end=end)
                prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                excel_columns += eval_col
            else:
                eval_model_dir = os.path.join(eval_samples_dir, model_dict[model_name])
                os.makedirs(eval_model_dir, exist_ok=True)
                # Copy srt files into the respective sample/model directory

                # ICL
                srt_mode = 'icl_srt'
                source = group_model.iloc[0][srt_mode]
                destination = os.path.join(eval_model_dir,
                                           f"{srt_dict[srt_mode].replace('.srt', f'_{start}-{end}.srt')}")
                cut_subtitles(subs_in=source, subs_out=destination, start=start, end=end)
                prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                excel_columns += eval_col

                # Realtime
                srt_mode = 'realtime_srt'
                source = group_model.iloc[0][srt_mode]
                destination = os.path.join(eval_model_dir,
                                           f"{srt_dict[srt_mode].replace('.srt', f'_{start}-{end}.srt')}")
                cut_subtitles(subs_in=source, subs_out=destination, start=start, end=end)
                prefix = f"{model_dict[model_name]}_{srt_dict[srt_mode]}"
                eval_col = [f"{prefix}_{e}" for e in evaluation_metrics]
                excel_columns += eval_col

    eval_df = pd.DataFrame(0, index=np.arange(len(samples)), columns=excel_columns)
    eval_df["sample"] = samples
    eval_df.to_excel(f"evaluation_samples/{ds}_evaluation_samples.xlsx")
    print(eval_df)









