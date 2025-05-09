# Library Imports
import os
import av
import fsspec
import numpy as np
import shutil
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from main import get_utterence_timing, extract_until_last_complete_sentence, create_ds, baseline_feedback_loop
import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from utils.data_utils import read_srt
# Local Module imports
from utils.video_utils import sample_frames, get_video_info, write_video, read_video, process_video
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

def collate_fn(example):
    video_clips = read_video(example["video"])
    video_clips= np.transpose(video_clips, (0,3, 1, 2))
    prev_gen = example["prev_generations"]
    gt = example["gt"]

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
    #video_clips = video_clips.to(model.device)
    batch = processor(
        text=prompt,
        videos=video_clips,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return batch
def collate_fn_batch(examples):
    video_clips = [read_video(path) for path in examples["video"]]
    video_clips = [process_video(clip, num_frames=NUM_FRAMES) for clip in video_clips]
    video_clips = np.stack(video_clips, axis=0)# list of video clips
    video_clips= np.transpose(video_clips, (0,1, 4, 2, 3))
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
    for i, v in enumerate(video_clips):
        print(f"Video {i} shape:", v.shape)
    batch = processor(
        text=prompts,
        videos=video_clips,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


    return batch

def create_training_samples(hf_ds, path, step = 1, num_frames_to_use = 1):
    hf_dataset = []
    cache_video_folder = f"{path}_videos"

    os.makedirs(cache_video_folder, exist_ok=True)
    for i in tqdm(range(len(hf_ds))):

        mp4_file = hf_ds[i]["video_path"]
        transcription_file = hf_ds[i]["srt_path"]
        sample_name = hf_ds[i]["sample_name"]

        srt = read_srt(transcription_file)
        video_metadata = get_video_info(mp4_file)
        ref_utterences, ref_timing = get_utterence_timing(srt, video_metadata)
        for t in range(0, video_metadata["duration"], step): #tqdm(range(0, video_metadata["duration"], step), total=video_metadata["duration"] / step):

            video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * video_metadata["frames_per_second"],
                                  end_frame=(t + 1) * video_metadata["frames_per_second"], format="video")

            video_path = os.path.join(cache_video_folder, os.path.basename(mp4_file.replace('.mp4', f'_{t}.mp4')))
            write_video(video, video_path, video_metadata["frames_per_second"])
            prev_generations = " ".join(ref_utterences[:(t - step)])
            ground_truth = " ".join([ref_utterences[t - j] for j in reversed(range(step))])
            if not ground_truth.strip():
                ground_truth = "<WAIT>"
            dataset_item = {"sample_name": sample_name,
                           "video": video_path ,
                          #  "video": video,
                            "prev_generations": prev_generations,
                            "num_frames": video.shape[0],
                            "gt":ground_truth}
            hf_dataset.append(dataset_item)

    hf_dataset = Dataset.from_list(hf_dataset)

    hf_dataset.save_to_disk(path)
    print(f"Dataset saved to {path}")
    return hf_dataset


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


def organize_metrics(feedback_loop_generation, config):
    f_eval_metrics = feedback_loop_generation[2]


    additional_columns = ["model_name", "sample", "# frame", "step"]
    metrics_columns = (
            additional_columns +
            [f"feedback_{key}" for key in f_eval_metrics.keys()]
    )

    metrics_data = (
            [config['model'], config['sample_name'], config['# frame'], config['step']] +
            list(f_eval_metrics.values())
    )
    metrics = dict(zip(metrics_columns, metrics_data))
    additional_columns += ["feedback_ref_timing", "feedback_pred_timing", "feedback_ROUGE_10%", ]
    for k, v in metrics["feedback_ROUGE_10%"].items():
        metrics[f"feedback_{k}"] = v

    metrics_per_sample = {k: v for k, v in metrics.items() if k not in additional_columns}
    return metrics_per_sample


def run_inference(example, model):
    split_word = "ASSISTANT:"
    # Let's use chat template to format the prompt correctly, this time without the caption
    inputs_video = collate_fn(example)
    inputs_video = inputs_video.to(model.device)

    output = model.generate(**inputs_video, do_sample=True, max_new_tokens=50)

    generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
    generated_text = generated_text.split(split_word)[-1]
    generated_text = extract_until_last_complete_sentence(generated_text)
    return generated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generates commentary as per the defined settings"
        )
    parser.add_argument("--dir", required=True, type=str, help="Directory containing the videos "
                            "and respective commentary in recordings and transcriptions_whole_data_english folder",
                            default="/Users/anumafzal/PycharmProjects/video2Text/RaceCommentary")
    parser.add_argument("--n", required=False, type=int, default=10, help="Number of samples to run")
    parser.add_argument("--use_existing", required=False, type=bool, help="Linking to previously preprocessed training/validaiton set", default=None)
    parser.add_argument("--step", required=False, type=int, default=2, help="Time Step for generation")
    parser.add_argument("--frames", required=False, type=int, default=2,
                        help="Number of frames to use per step of generation")
    parser.add_argument("--hf_dataset", required=False, type=str,
                        help="The directory containing hf_Dataset", default = "RaceCommentaryEn/")
    parser.add_argument("--context_window", required=False, type=int, default=5120,
                        help="Context Window to be used by LLM")
    args = parser.parse_args()

    folder = args.dir
    n = args.n
    step = args.step
    hf_dataset_path = args.hf_dataset
    DATASET_PATH = args.dir
    MAX_LENGTH = args.context_window
    BATCH_SIZE = 2
    NUM_FRAMES = args.frames # more frames -> more VRAM needed
    OUTPUT_DIR = "logs/FT/" # path where to save the checkpoints
    MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
    USE_LORA = False
    USE_QLORA = True
    use_existing = args.use_existing


    config = {"num_frames_to_use": NUM_FRAMES, "step":step, "max_length": MAX_LENGTH, "use_lora": USE_LORA,
              "q_lora": USE_QLORA}

    if hf_dataset_path is None:
        hf_dataset_path = create_ds(DATASET_PATH)

    ft_dataset = datasets.load_from_disk(hf_dataset_path)
    print (ft_dataset)
    train_dataset_raw, test_dataset_raw = ft_dataset['train'].with_format("torch"), ft_dataset['test'].with_format("torch")

    # enable this line for testing
    #train_dataset_raw, test_dataset_raw = train_dataset_raw.select(range(2)), test_dataset_raw .select(range(2))

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    processor.tokenizer.padding_side = "right"

    if hf_dataset_path[-1] == '/':
        hf_dataset_path = hf_dataset_path.replace("/", "")
    ft_dataset_path = f"{hf_dataset_path}_FT_frames_{NUM_FRAMES}_step_{step}_n_{len(train_dataset_raw)}"

    if use_existing == True:
        train_dataset = datasets.load_from_disk(ft_dataset_path)
    else:
        print("Creating training data from videos and srt files!")
        train_dataset = create_training_samples(train_dataset_raw, path=ft_dataset_path,
                                                num_frames_to_use=config["num_frames_to_use"], step=config["step"])

    print(train_dataset)
    if n == -1:
        n = len(train_dataset)
    train_dataset = train_dataset.select(range(n))

    # set num_proc higher for faster processing
    #train_dataset = train_dataset.map(collate_fn_batch, batched=True, fn_kwargs={}, num_proc=2)
    dataset_processed = train_dataset.map(collate_fn, batched=False, fn_kwargs={}, num_proc=4)
    #os.makedirs(cache_dir, exist_ok=True)
    #train_dataset.save_to_disk(cache_dir)
    #print (f"collated data saved to {cache_dir}")


    dataset_processed = dataset_processed.shuffle(seed=42)
    dataset_processed = dataset_processed.train_test_split(test_size=0.2)

    train_dataset, validation_dataset = dataset_processed['train'].with_format("torch"), dataset_processed['test'].with_format("torch")
    print (f"{len(train_dataset)} training example, {len(validation_dataset)} validation examples")
    REPO_ID = f"anumafzal94/LLaVa-NeXT-Video-_step_{step}_frames_{NUM_FRAMES}_n_{len(train_dataset)}" # Change to your hf-hub repo

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
        num_train_epochs=1,
        #max_steps=5,  # adjust this depending on your dataset size
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
        eval_dataset=validation_dataset,
        args=args,
    )

    trainer.train()
    trainer.model.push_to_hub(REPO_ID)

    # ------------------------ Test the trained model on Validation Set-----------------------------------#

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Old Model")
    for i in range(1):
        example = validation_dataset[i]
        print(run_inference(example, model))

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            REPO_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    print ("FT Model")
    for i in range(1):
        example = validation_dataset[i]
        print(run_inference(example, model))

    # ------------------------------- Test the trained model on whole Train Set ----------------------- #
    split_word = "ASSISTANT:"
    out_folder = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    out_folder = os.path.join("logs", out_folder)
    os.makedirs(out_folder, exist_ok=True)
    metrics_all_samples = []
    for i in tqdm(range(len(test_dataset_raw))):
        # get sample
        mp4_file = test_dataset_raw[i]["video_path"]
        transcription_file = test_dataset_raw[i]["srt_path"]

        # create folder to store logs for each sample.
        sample_name = os.path.dirname(mp4_file).split('/')[-1]
        #try:
        print (out_folder)
        feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, NUM_FRAMES,
                                                                  init_skip_frames=10, step=step, ICL=False,
                                                                  split_word = split_word, processor=processor,
                                                              model=model, logs_dir=out_folder)

        config = {"model": REPO_ID, "step": step, "# frame": NUM_FRAMES, "sample_name": sample_name,
                          }

        metrics_per_sample =  organize_metrics(feedback_loop_generation, config)
        metrics_all_samples.append(metrics_per_sample)
        #except Exception as e:
        #    print (f"Caught the following exception for the sample \n Video Path:{mp4_file} \n Transcription File: {transcription_file} \n Exception: {e}")


    # Writing per experiments logs
    df = pd.DataFrame(metrics_all_samples)
    means_dict = df.select_dtypes(include='number').mean().to_dict()
    means_dict["n"] = len(df)
    means_dict["model_name"] = REPO_ID
    means_dict["# frame"] = NUM_FRAMES
    means_dict["step"] = step
    print(means_dict)

    import json
    run_name = f"FT_step_{step}_frames_{NUM_FRAMES}_n_{len(df)}"
    with open(f'{run_name}.json', 'w') as fp:
        json.dump(means_dict, fp)


