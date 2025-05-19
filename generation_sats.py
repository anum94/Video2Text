import os
import re
from janome.tokenizer import Tokenizer
import pandas as pd
model_dict = {"llava-hf_LLaVA-NeXT-Video-7B-hf": "M1",
              "gpt-4o-mini-2024-07-18": "M3",
                "Qwen_Qwen2.5-VL-7B-Instruct": "M2",
              #"anumafzal94_LLaVa-NeXT-Video-_step_2_frames_1_n_40000": "M4", #Fine-tuned on Race commetary english
              #"anumafzal94/FT_LLaVa-NeXT-Video-_step_2_frames_1_n_40000": "M5" #FT race commentary ja
              }
import os
from utils.data_utils import read_srt
def get_subdirs_at_depth(root_dir, target_depth=4):
    result = []
    root_depth = root_dir.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_depth = dirpath.rstrip(os.sep).count(os.sep)
        if current_depth - root_depth == target_depth:
            result.append(dirpath)

    return [res for res in result if "step_" in res]


if __name__ == '__main__':
    logs_directory = "logs/"
    datasets = ["RaceCommentaryEn", "RaceCommentaryJa", "SmabraDataJa",]
    logs_list = []
    for ds in datasets:
        print (ds)
        logs_list = []
        for file_path in get_subdirs_at_depth(logs_directory, 5):
             sample_dict = {}
             nested_paths = file_path.split('/')

             if len(nested_paths) < 6 or ds not in nested_paths:
                 continue
             #print(nested_paths)

             #print(nested_paths)
             sample_dict["model"] = nested_paths[-3]

             config = nested_paths[-1].split("_")
             sample_dict["step"], sample_dict["frames_used"], sample_dict["k"] = config[1], config[3], config[5]
             sample_dict["sample"] = nested_paths[-2]


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

        df = pd.DataFrame(logs_list)#.dropna()
        df = df.dropna(subset=['icl_srt'])
        df = df.dropna(subset=['baseline_srt'])
        df = df.dropna(subset=['feedback_srt'])
        df = df.dropna(subset=['realtime_srt'])


        df = df[((df['step'] == '2') & (df['frames_used'] == '1')) & (df['k'].isin(['8'])) ]
        df.sample(frac=1)

        df_models = df.groupby('model')
        excel_columns = []
        samples = []
        pattern = r'(\d+):\s*(.+)'
        for model_name, group_model in df_models:
            print (model_name)
            srts = []
            group_model.sample(frac=1)
            group_model = group_model.head(19)
            print (len(group_model))
            for item in group_model.iterrows():

                for mode in ["realtime_srt", "feedback_srt", "baseline_srt", "icl_srt"]:
                    #print (item[1][mode])
                    srt = read_srt(item[1][mode])

                    if "Ja" in ds:
                        #print (srt.text)
                        tokenizer = Tokenizer()
                        tokens = list(tokenizer.tokenize(srt.text))
                        srt_count = [token.surface for token in tokens if
                                     token.surface.strip() and token.part_of_speech.split(',')[0] != '記号']
                        srt_count = len(srt_count)
                        srts.append(srt_count)


                    else:
                        c = len(srt.text.split())
                        srts.append(c)
            maximum = max(srts)
            minimum = min(srts)
            average = sum(srts) / len(srts)

            print("Average Commentary words:", average)
            print("Minimum Commentary words:", minimum)
            print("Maximum Commentary words:", maximum)



