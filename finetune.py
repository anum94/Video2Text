# Library Imports
import os
import av
import fsspec
import shutil
import pandas as pd
import numpy as np
from datasets import Dataset
#from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
#from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from main import get_utterence_timing
import torch
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from datasets import load_dataset, concatenate_datasets
from utils.data_utils import read_srt
# Local Module imports
from utils.video_utils import sample_frames, get_video_info

MAX_LENGTH = 256
BATCH_SIZE = 4
NUM_FRAMES = 8 # more frames -> more VRAM needed
DATASET_PATH = "/Users/anumafzal/PycharmProjects/video2Text/RaceCommentary" # path where to save the dataset
OUTPUT_DIR = "/Users/anumafzal/PycharmProjects/video2Text/logs/FT/" # path where to save the checkpoints
MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
REPO_ID = "anumafzal94/LLaVa-NeXT-Video-demo" # Change to your hf-hub repo

USE_LORA = False
USE_QLORA = True
def get_commentary_path(commentary_directory, game_path):
    game_path = os.path.basename(game_path)
    commentary_path = [os.path.join(commentary_directory, file) for file in os.listdir(commentary_directory) if
     file.endswith('.srt') and os.path.isfile(os.path.join(commentary_directory, file)) and "kyakkan" in file
         and game_path in file]
    if len(commentary_path) > 0:
        commentary_path = commentary_path[0]
    else:
        commentary_path = None
    return commentary_path
def collate_fn(example, video_path, config:dict):
    video_clip = sample_frames(video_path, config['num_frames_to_use'], start_frame= config['start_frame'],
                          end_frame=config['end_frame'], format="video")

    # we'll take the overall video caption, not per-scene caption for each frame
    captions_all = [caption for caption in example['captions'] if caption['idx'] == '-1']
    caption = captions_all[0]['content']

    # Let's use chat template to format the prompt correctly
    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a detailed caption for this video."},
                    {"type": "video"},
                    ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption},
                     ],
            },
        ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

    batch = processor(
        text=prompt,
        videos=video_clip,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return batch
def convert_to_hf_dataset(folder):
    dataset_list = []
    # define directory paths
    video_directory = "recordings"
    video_directory = os.path.join(folder,video_directory)

    commentary_directory = "transcriptions_whole_data_english"
    commentary_directory = os.path.join(folder,commentary_directory)

    all_game_path = [os.path.join(video_directory, name) for name in os.listdir(video_directory) if
                     os.path.isdir(os.path.join(video_directory, name))]

    for i, game_path in enumerate(all_game_path):

        transcription_file = get_commentary_path(commentary_directory, game_path)
        if transcription_file is not None:
            mp4_file = [os.path.join(game_path, file) for file in os.listdir(game_path) if
                        file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
        else:
            print(f"kyakkan commentary not available for game: {game_path}")
            continue

        # Baseline without feedback loop
        sample_name = os.path.dirname(mp4_file).split('/')[-1]
        srt = read_srt(transcription_file)
        video_metadata = get_video_info(mp4_file)
        ref_utterences, ref_timing = get_utterence_timing(srt, video_metadata)
        dataset_item = {"video_path": mp4_file , "ref_timing": ref_timing, "ref_utterences":ref_utterences}
        dataset_list.append(dataset_item)

    dataset = Dataset.from_list(dataset_list)
    return dataset

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
processor.tokenizer.padding_side = "right"
dataset =  convert_to_hf_dataset(DATASET_PATH)
print (dataset)

'''
# set num_proc higher for faster processing
small_dataset = small_dataset.map(collate_fn, batched=False, fn_kwargs={"path": f"{directory}/{zip_folder}"}, num_proc=8)
temp_dataset = process_dataset(zip_file)

dataset_processed = dataset_processed.shuffle(seed=42)
dataset = dataset_processed.train_test_split(test_size=0.2)
train_dataset, test_dataset = dataset['train'].with_format("torch"), dataset['test'].with_format("torch")
'''