# Library Imports
import os
from email.policy import default

import av
import fsspec
import shutil
import pandas as pd
from tqdm import tqdm
import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from main import get_utterence_timing
import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from utils.data_utils import read_srt
# Local Module imports
from utils.video_utils import sample_frames, get_video_info, write_video
import argparse
# Reference tutorial: LLaVA-NeXT-Video/Fine_tune_LLaVa_NeXT_Video_with_HFTrainer.ipynb

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
    prompt =    ("You are a professional commentator for car racing games. You are provided with a video clip"
                "from an ongoing car racing game and commentary generated for the game so far."
                 f"Previous generated Commentary: {prev_generation}"
                 "Your task is to compare the given video with the previously generated commentary. "
                "1) Identify if the video has any new development as compared to the already provided commentary."
                "2) Ignore the background information and refrain the describing the scenery too much."
                "3) If the state of the game as compared to the provided commentary has not changed, then generate <WAIT>"
                "4) If there are new developments in the provided video, then generate 1 - 2 line of commentary to describe it."
            )

    return prompt

def collate_fn(examples):
    video_clips = examples["video"]  # list of video clips
    prev_generations = examples["prev_generations"]
    ground_truths = examples["gt"]
    prompts = []
    for prev_gen, gt in zip(prev_generations, ground_truths):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_FT_prompt(prev_gen)},
                    {"type": "video"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        prompts.append(prompt)

    batch = processor(
        text=prompts,
        videos=video_clips,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return batch
def convert_to_hf_dataset(folder, step = 1, num_frames_to_use = 1):
    dataset = []
    path = f'CarRacingFT_{len(dataset)}_step_{step}_numframes_{num_frames_to_use}'
    os.makedirs(path, exist_ok=True)
    # define directory paths
    video_directory = "recordings"
    video_directory = os.path.join(folder,video_directory)

    commentary_directory = "transcriptions_whole_data_english"
    commentary_directory = os.path.join(folder,commentary_directory)

    all_game_path = [os.path.join(video_directory, name) for name in os.listdir(video_directory) if
                     os.path.isdir(os.path.join(video_directory, name))][:10]

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
        for t in range(0, video_metadata["duration"], step): #tqdm(range(0, video_metadata["duration"], step), total=video_metadata["duration"] / step):
            video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * video_metadata["frames_per_second"],
                                  end_frame=(t + 1) * video_metadata["frames_per_second"], format="video")
            video_path = os.path.join(path, mp4_file.replace('.mp4', f'_{t}.mp4'))
            write_video(video, video_path, video_metadata)
            prev_generations = " ".join(ref_utterences[:(t - step)])
            ground_truth = " ".join([ref_utterences[t - j] for j in reversed(range(step))])
            if not ground_truth.strip():
                ground_truth = "<WAIT>"
            dataset_item = {"sample_name": sample_name,
                            "video": video_path ,
                            "prev_generations": prev_generations,
                            "gt":ground_truth}
            dataset.append(dataset_item)

    dataset = Dataset.from_list(dataset)

    dataset.save_to_disk(path)
    print(f"Dataset saved to {path}")
    return dataset


# ------------------------------------- LLM Fine-tuning ------------------------------------

class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [feat['input_ids'][0] for feat in features], # each element is one batch only so we slice [0]
                "attention_mask": [feat['attention_mask'][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)

        return padded_inputs


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def run_inference(video_clip, model):
    # Let's use chat template to format the prompt correctly, this time without the caption
    user_prompt = ("You are a professional commentator for car racing games. You will be provided with few frames"
                   " from an on-going game and your task is generate brief Commentary."
                   "1) Ignore the background information and refrain the describing the scenery."
                   "2) Do not regenerate information that is already part of the Previous Commentary."
                   "3) Identify new developments if any, in the provided video clip as compared to previous commentary, then generate 1 sentence of commentary."
                   "If nothing has change, then generate <WAIT>. Otherwise a brief commentary"
                   "Previous generated Commentary: "
                   )
    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video"},
                    ],
            },
        ]

    # Set add_generation_prompt to add the "ASSISTANT: " at the end
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    batch = processor(
        text=prompt,
        videos=None, # we have a processed video, passing it again to processor causes errors
        return_tensors="pt"
    ).to(model.device)
    video_clip = video_clip.to(model.device)

    out = model.generate(**batch, pixel_values_videos=video_clip, max_length=MAX_LENGTH, do_sample=True)
    generated_text = processor.batch_decode(out, skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generates commentary as per the defined settings"
        )
    parser.add_argument("--dir", required=True, type=str, help="Directory containing the videos "
                            "and respective commentary in recordings and transcriptions_whole_data_english folder",
                            default="/Users/anumafzal/PycharmProjects/video2Text/RaceCommentary")
    parser.add_argument("--n", required=False, type=int, default=-1, help="Number of samples to run")
    parser.add_argument("--step", required=False, type=int, default=1, help="Time Step for generation")
    parser.add_argument("--frames", required=False, type=int, default=-1,
                        help="Number of frames to use per step of generation")
    parser.add_argument("--context_window", required=False, type=int, default=5120,
                        help="Context Window to be used by LLM")
    args = parser.parse_args()

    folder = args.dir
    n = args.n
    step = args.step

    DATASET_PATH = args.dir
    MAX_LENGTH = args.context_window
    BATCH_SIZE = 1
    NUM_FRAMES = args.frames # more frames -> more VRAM needed
    OUTPUT_DIR = "logs/FT/" # path where to save the checkpoints
    MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
    USE_LORA = False
    USE_QLORA = True
    REPO_ID = "anumafzal94/LLaVa-NeXT-Video-demo" # Change to your hf-hub repo

    config = {"num_frames_to_use": NUM_FRAMES, "step":step, "max_length": MAX_LENGTH}


    create_dataset = True
    if create_dataset:
        dataset =  convert_to_hf_dataset(DATASET_PATH, num_frames_to_use=config["num_frames_to_use"], step=config["step"])
        print (dataset)
        dataset.push_to_hub("anumafzal94/test")
    else:
        dataset_path = "CarRacingFT_89_step_4_numframes_1"
        dataset = datasets.load_from_disk(dataset_path)


    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    processor.tokenizer.padding_side = "right"
    # set num_proc higher for faster processing
    dataset = dataset.map(collate_fn, batched=True, fn_kwargs={}, num_proc=8)


    dataset_processed = dataset.shuffle(seed=42)
    dataset = dataset_processed.train_test_split(test_size=0.2)
    train_dataset, test_dataset = dataset['train'].with_format("torch"), dataset['test'].with_format("torch")
    print (f"{len(train_dataset)} training example, {len(test_dataset)} testing examples")
    example = test_dataset[0]
    processor.batch_decode(example["input_ids"])

    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto",
        )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(

        # args related to training
        output_dir=OUTPUT_DIR,
        eval_strategy='steps',
        eval_steps=20,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=2e-05,
        max_steps=100,  # adjust this depending on your dataset size
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,

        # args related to eval/save
        logging_steps=20,
        save_strategy='steps',
        save_steps=20,
        save_total_limit=1,
        fp16=True,  # we have the model train and eval with fp16 precision
        fp16_full_eval=True,
        optim='adamw_bnb_8bit',
        # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        report_to="wandb",  # install wand to use this
        hub_model_id=REPO_ID,
        push_to_hub=True,  # wel'll push the model to hub after each epoch

        # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
        # so we need to pass lable names explicitly to calculate val loss
        label_names=["labels"],
        dataloader_num_workers=4,  # let's get more workers since iterating on video datasets might be slower in general
    )

    trainer = Trainer(
        model=model,
        tokenizer=processor,
        data_collator=LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
    )

    trainer.train()
    trainer.model.push_to_hub(REPO_ID)

    # ------------------------ Test the trained model -----------------------------------#
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        REPO_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print (run_inference(example["pixel_values_videos"], model))

    old_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(run_inference(example["pixel_values_videos"], old_model))
