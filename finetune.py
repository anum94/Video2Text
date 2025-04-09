# Library Imports
import os
import av
import datasets
import fsspec
import shutil
import pandas as pd
from tqdm import tqdm
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
def get_FT_prompt(prev_generation):
    prompt =    ("You are a professional commentator for car racing games.You will be provided with few frames"
                           "from an ongoing game and your task is generate brief Commentary for it."
                "1) Identify if the provided video has any new development as compared to the already provided commentary."
                "2) Ignore the background information and refrain the describing the scenery."
                "3) If the state of the game as compared to the provided commentary has not changed, then generate <WAIT>"
                "4) If there are new developments in the provided video, such as if a new player is in lead, or if one of the players did an "
                "impressive move, or if two players are competing strongly, then generate 1 line of commentary to describe it."
                f"Previous generated Commentary: {prev_generation}"
            )

    return prompt

def collate_fn(example):
    video_clip = example["video"]
    prev_generation = example["prev_generations"]
    ground_truth = example["gt"]

    # Let's use chat template to format the prompt correctly
    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_FT_prompt(prev_generation)},
                    {"type": "video"},
                    ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth},
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
def convert_to_hf_dataset(folder, step = 1, num_frames_to_use = 1):
    dataset = []
    # define directory paths
    video_directory = "recordings"
    video_directory = os.path.join(folder,video_directory)

    commentary_directory = "transcriptions_whole_data_english"
    commentary_directory = os.path.join(folder,commentary_directory)

    all_game_path = [os.path.join(video_directory, name) for name in os.listdir(video_directory) if
                     os.path.isdir(os.path.join(video_directory, name))]

    for i, game_path in tqdm(enumerate(all_game_path), total = len(all_game_path)):

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
        for t in tqdm(range(0, video_metadata["duration"], step), total=video_metadata["duration"] / step):
            video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * video_metadata["frames_per_second"],
                                  end_frame=(t + 1) * video_metadata["frames_per_second"], format="video")
            prev_generations = " ".join(ref_utterences[:(t - step)])
            ground_truth = " ".join([ref_utterences[t - j] for j in reversed(range(step))])
            if not ground_truth.strip():
                ground_truth = "<WAIT>"
            dataset_item = {"sample_name": sample_name,
                            "video": video ,
                            "prev_generations": prev_generations,
                            "gt":ground_truth}
            dataset.append(dataset_item)

    dataset = Dataset.from_list(dataset)
    dataset.save_to_disk(f'CarRacingFT_{len(dataset)}_step_{step}_numframes_{num_frames_to_use}')
    return dataset


config = {"num_frames_to_use": 1, "step":6}
create_dataset = True
if create_dataset:
    dataset =  convert_to_hf_dataset(DATASET_PATH, num_frames_to_use=config["num_frames_to_use"], step=config["step"])
else:
    dataset_path = "CarRacingFT_89_step_4_numframes_1"
    dataset = datasets.load_from_disk(dataset_path)
print (dataset)


# set num_proc higher for faster processing
dataset = dataset.map(collate_fn, batched=False, fn_kwargs={}, num_proc=8)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
processor.tokenizer.padding_side = "right"
dataset_processed = dataset.shuffle(seed=42)
dataset = dataset_processed.train_test_split(test_size=0.2)
train_dataset, test_dataset = dataset['train'].with_format("torch"), dataset['test'].with_format("torch")
print (train_dataset)
print (test_dataset)