import os.path
import random

import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
import time
import pandas as pd
import warnings
from openai import OpenAI
import base64
import requests
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
warnings.filterwarnings("ignore")
from utils.logs import *
from utils.data_utils import *
from utils.video_utils import *
import argparse
import json
from datasets import Dataset, load_dataset, concatenate_datasets
import datasets
import re

def get_user_prompt(mode="baseline", context="", step = 1, force=False):
    #todo: move prompts to a yaml file
    if mode == "baseline":
        user_prompt = ("You are a professional commentator for car racing games. You are provided with a video clip"
                       "from an ongoing car racing game and commentary generated for the game so far. \n"
                       "Your task is to generate 1 - 2 line of commentary to describe it. Ignore the background information and refrain the describing the scenery too much."
                       "\nDescribe this scene as a single-sentence commentary for making audience immersed.  If you have nothing to say, generate a <WAIT> token."
                       )

    elif mode == "baseline_ja":
        user_prompt = ("あなたはカーレースのプロの実況者です。以下に示すのは現在進行中のレースのビデオクリップと、これまでに生成された実況です。\n"
                       "このシーンを1文で説明する日本語の実況を生成してください。\n"
                       "観客が没入できるような自然な実況を心がけてください。話すべきことがなければ <WAIT> を出力してください。")
    elif mode == "baseline_smabra":
        user_prompt = ("あなたは大乱闘スマッシュブラザーズのプロの実況者です。以下に示すのは現在進行中の対戦のビデオクリップと、これまでに生成された実況です。\n"
                       "このシーンを1文説明する日本語の実況を生成し視聴者を楽しませてください。\n"
                       "観客が没入できるよう驚きや感嘆句も含めてエキサイティングな実況となるよう心がけてください。話すべきことがなければ <WAIT> を出力してください。")

    elif mode == "feedback_loop_init":
        user_prompt = ("You are a professional commentator for car racing games. You will be provided with a video clip"
                       "which represents the start of a race. Your task is to generate one sentences of commentary. "
                       "1) You should identify the number of players and their names along with cars. "
                       "2) Ignore the background information and refrain the describing the scenery."
                       "3) Initial information about the game without being too verbose."
                       )

    elif mode == "feedback_loop_init_ja":
        user_prompt = ("あなたはカーレースのプロの実況者です。これからレース開始時のビデオクリップが提示されます。\n"
                       "それに対して1文の日本語実況を生成してください。\n"
                       "冗長になりすぎず、レースの初期情報を伝えてください。人名や車種には言及せず「プレイヤー」や車の色を使って説明してください．")

    elif mode == "feedback_loop_init_smabra":
        user_prompt = ("あなたは大乱闘スマッシュブラザーズのプロの実況者です。これから対戦開始時のビデオクリップが提示されます。\n"
                       "このシーンを1文で説明する日本語の実況を生成し視聴者を楽しませてください。\n"
                       "観客が没入できるよう驚きや感嘆句も含めてエキサイティングな実況となるよう心がけてください。話すべきことがなければ <WAIT> を出力してください。")

    elif mode == "feedback_loop":
        if force:
            user_prompt = ("You are a professional commentator for car racing games. You are provided with a video clip"
                           "from an ongoing car racing game and commentary generated for the game so far."
                           f"\nPrevious generated Commentary: \n{context}\n"
                           "Your task is to compare the given video with the previously generated commentary. \n"
                           "1) Identify if the video has any new development as compared to the already provided commentary. \n"
                           "2) Ignore the background information and refrain the describing the scenery too much. \n"
                           "3) If there are new developments in the provided video, then generate 1 - 2 line of commentary to describe it. \n"
                           )
        else:
            user_prompt = ("You are a professional commentator for car racing games. You are provided with a video clip"
                           "from an ongoing car racing game and commentary generated for the game so far."
                           f"\nPrevious generated Commentary: \n{context}\n"
                           "Your task is to compare the given video with the previously generated commentary. \n "
                           "1) Identify if the video has any new development as compared to the already provided commentary. \n"
                           "2) Ignore the background information and refrain the describing the scenery too much.\n"
                           "3) If the state of the game as compared to the provided commentary has not changed, then generate <WAIT>\n"
                           "4) If there are new developments in the provided video, then generate 1 - 2 line of commentary to describe it.\n"
                           )
    elif mode == "feedback_loop_ja":
        if force:
            user_prompt = ("あなたはカーレースのプロの実況者です。以下に示すのは現在進行中のレースのビデオクリップと、これまでに生成された実況です。\n"
                           f"\nこれまでの実況:\n{context}\n"
                           "以下のルールに従って日本語実況を1〜2文生成してください：\n"
                           "1) 新たな展開があるかどうかを特定してください。\n"
                           "2) 背景や風景の描写は避けてください。\n"
                           "3) 変化がある場合は、それを説明する1文の実況を生成してください。\n"
                           "4) 人名や車種には言及せず「プレイヤー」や車の色を使って説明してください．")
        else:
            user_prompt = ("あなたはカーレースのプロの実況者です。以下に示すのは現在進行中のレースのビデオクリップと、これまでに生成された実況です。\n"
                           f"\nこれまでの実況:\n{context}\n"
                           "ビデオに新たな展開があるかどうかを比較・分析し、以下のルールに従って日本語実況を生成してください：\n"
                           "1) 新たな展開があるかどうかを特定してください。\n"
                           "2) 状況に変化がなければ <WAIT> を出力してください。\n"
                           "3) 明確な変化があれば、それを説明する1文の実況を生成してください。\n"
                           "4) 人名や車種には言及せず「プレイヤー」や車の色を使って説明してください．\n")

    elif mode == "feedback_loop_smabra":
        if force:
            user_prompt = ("あなたは大乱闘スマッシュブラザーズのプロの実況者です。以下に示すのは現在進行中のレースのビデオクリップと、これまでに生成された実況です。\n"
                               f"\nこれまでの実況:\n{context}\n"
                                "このシーンを1文で説明する日本語の実況を生成し視聴者を楽しませてください。\n"
                                "観客が没入できるよう驚きや感嘆句も含めてエキサイティングな実況となるよう心がけてください。話すべきことがなければ <WAIT> を出力してください。")

        else:
            user_prompt = ("あなたは大乱闘スマッシュブラザーズのプロの実況者です。以下に示すのは現在進行中のレースのビデオクリップと、これまでに生成された実況です。\n"
                               f"\nこれまでの実況:\n{context}\n"
                                "このシーンを1文で説明する日本語の実況を生成し視聴者を楽しませてください。\n"
                                "観客が没入できるよう驚きや感嘆句も含めてエキサイティングな実況となるよう心がけてください。話すべきことがなければ <WAIT> を出力してください。")


    return user_prompt
def create_ds(folder):
    video_directory = "recordings"
    video_directory = os.path.join(folder, video_directory)

    commentary_directory = "transcriptions_whole_data_english"
    commentary_directory = os.path.join(folder, commentary_directory)
    hf_dataset = []

    all_game_path = [os.path.join(video_directory, name) for name in os.listdir(video_directory) if
                     os.path.isdir(os.path.join(video_directory, name))]

    count = 0
    for i, game_path in enumerate(all_game_path):
        transcription_file = get_commentary_path(commentary_directory, game_path)
        if transcription_file is not None:
            mp4_file = [os.path.join(game_path, file) for file in os.listdir(game_path) if
                        file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
        else:
            # print (f"kyakkan commentary not available for game: {game_path}")
            count += 1
            continue

        sample_name = os.path.dirname(mp4_file).split('/')[-1]
        dataset_item = {"sample_name": sample_name,
                        "video_path": mp4_file,
                        "srt_path": transcription_file, }
        hf_dataset.append(dataset_item)
        for i in range (50):
            hf_dataset.append(dataset_item)

    hf_dataset = Dataset.from_list(hf_dataset)
    dataset_processed = hf_dataset.shuffle(seed=42)
    print(f"kyakkan commentary not available for {count} samples.")
    #print(dataset_processed)
    hf_dataset = dataset_processed.train_test_split(test_size=0.25)
    dir = f"{os.path.basename(folder)}_HF" #"RaceCommentaryEn/"
    os.makedirs(dir, exist_ok=True)
    hf_dataset.save_to_disk(dir)
    return dir

def encode_frame(frame):
    if frame is None:
        raise ValueError("Received None as frame.")
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame is not a numpy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb)
    buf = BytesIO()
    im_pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_messages_openai(frames_b64, prompt="Describe what's happening in this video", ICL = False):
    messages = []
    if ICL:
        for i, ex in enumerate(ICL):
            content = [{"type": "text", "text": ex['prompt']}]
            for img_b64 in frames_b64[i]:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
            messages.append(
                {"role": "user", "content": content}
            )
            messages.append(
                {"role": "assistant", "content": ex['generation']}

            )

    content = [{"type": "text", "text": prompt}]
    for img_b64 in frames_b64[-1]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })
    messages.append(
            {"role": "user", "content": content}

    )
    return messages


def get_utterence_timing(ground_truth,metadata):
    utterence_timing = [False] * int(metadata.get("duration"))
    utterences = [""] * int(metadata.get("duration"))
    for gt in ground_truth:
        i = srt_time_to_seconds(gt.start)
        if i >= 0 and i < len(utterence_timing):
            utterence_timing[i] = True
            utterences[i] = gt.text
        #else:
            #print (f"i: {i}")


    return utterences, utterence_timing

def run_inference(model_name, model, processor, prompt, videos, ICL=False, context_window = 4096, split_word = "ASSISTANT:" ):
    if "gpt" in model_name:
        encoded_frames = [[encode_frame(f) for f in video] for video in videos]
        messages = get_messages_openai(encoded_frames, prompt=prompt, ICL=ICL)

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=max_new_tokens,
        )

        pred_utterence = response.choices[0].message.content.strip()



    else:
        messages = get_messages(prompt, ICL=ICL)
        if "qwen" in model_name:
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            #images, videos = process_vision_info(messages)
            inputs_video = processor(
                text=text,
                #images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, padding=True)
            inputs_video = processor(text=prompt, videos=videos, padding=True, return_tensors="pt",
                                 max_length=context_window).to(model.device)

        output = model.generate(**inputs_video, do_sample=False, max_new_tokens=50, no_repeat_ngram_size=4, temperature=1.0)
        pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
        pred_utterence = pred_utterence.split(split_word)[-1]
    pred_utterence = extract_until_last_complete_sentence(pred_utterence)
    #print (pred_utterence)
    return pred_utterence

def identify_dataset(transcription_file):
    if "transcriptions_whole_data_english" in transcription_file:
        return "" # race game in English
    elif "smabra_ja" in transcription_file:
        return "_smabra" # smash corpus
    else:
        return "_ja" # race game in Japanese


def baseline(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False, split_word = "ASSISTANT:", ):
    data_prefix = identify_dataset(transcription_file)
    user_prompt = get_user_prompt("baseline" + data_prefix)

    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]
    previous_generation = ""
    pred_utterences = []
    pred_utterences_step =[]
    pred_timing = []
    #print(transcription_file)


    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):


        video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        pred_utterence = run_inference(model_name, model, processor, user_prompt, [video],
                                       context_window=context_window, split_word=split_word)

        if "WAIT" in pred_utterence:
            pred_timing.append(False)
            pred_utterences.append("<WAIT>")
        else:
            pred_timing.append(True)
            pred_utterences.append(pred_utterence)
            previous_generation = pred_utterence
        pred_utterences_step.append(t)


        if t % 10 == 0 and verbose:
            print(f"{t}: {pred_utterence}")

    #pred_utterences = remove_repeatitions(pred_utterences)

    ref_timing = [ref_timing[ref] for ref in range(0,len(ref_timing),step)]
    ref_utterences = [ref_utterences[ref] for ref in range(0, len(ref_utterences), step)]
    pred_srt_file = write_logs(out_folder, pred_utterences, pred_utterences_step, mode="baseline",
                               talking_speed_sample=icl_transcription_file)
    eval_metrics = compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences,
                            pred_srt_file, transcription_file)
    print (f"Logs written at {out_folder}")
    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences, pred_utterences_step, eval_metrics, ref_utterences

def get_messages(user_prompt, ICL = False , proc = None):
    if proc:
        processor = proc
    conversation = []
    if ICL:
        for icl_number in range(len(ICL)):
            # add user text
            conversation.append(
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": ICL[icl_number]['prompt']},
                    {"type": "video"},
                ],
            })
            # add assistant text
            conversation.append(
                {

                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ICL[icl_number]['generation']},
                        #{"type": "video"},
                    ],
                })

    conversation.append(
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video"},
                ],
            }
        )
    #print (conversation)
    return conversation
def construct_icl_examples(example, t, k=2, step=1,num_frames_to_use = 5,skip_frames = 20,):
    icl_examples = []
    transcriptions = read_srt(example['transcription'])
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(transcriptions, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

    k_pair = int(k/2) if (k/2) >1 else 1
    for i in range(k_pair):

        # get positive and negative examples
        if t <= skip_frames:
            window = skip_frames -1
            start_window = 0
        else:
            window = video_metadata['duration'] -1
            start_window = skip_frames

        t1 = random.randint(start_window,window)
        while ref_timing[t1] != True:
            t1 = random.randint(start_window,window)
        t2 = random.randint(start_window,window)
        while ref_timing[t2] != False:
            t2 = random.randint(start_window,window)

        init_str = " ".join(ref_utterences[:skip_frames])
        output_buffer_str = " ".join(ref_utterences[:t1])
        data_prefix = identify_dataset(example["transcription"])
        user_prompt_t1 = get_user_prompt("feedback_loop" + data_prefix, context=init_str, step=step)
        user_prompt_t1 += output_buffer_str
        generation_t1 = ref_utterences[t1]
        video_t1 = sample_frames(mp4_file, num_frames_to_use, start_frame=(t1 - step + 1) * num_frames_per_second,
                              end_frame=(t1 + 1) * num_frames_per_second, format="video")

        generate_example = {"video": video_t1, "prompt": user_prompt_t1, "generation": generation_t1}

        video_t2 = sample_frames(mp4_file, num_frames_to_use, start_frame=(t2 - step + 1) * num_frames_per_second,
                                 end_frame=(t2 + 1) * num_frames_per_second, format="video")
        init_str = "".join(ref_utterences[:skip_frames])
        output_buffer_str = "".join(ref_utterences[:t2])
        data_prefix = identify_dataset(example['transcription'])
        user_prompt_t2 = get_user_prompt("feedback_loop" + data_prefix, context=init_str, step=step)
        user_prompt_t2 += output_buffer_str
        generation_t2 = "<WAIT>"

        wait_example = {"video": video_t2, "prompt": user_prompt_t2, "generation": generation_t2}
        icl_examples.append(generate_example)
        icl_examples.append(wait_example)


    return icl_examples


def realtime_feedback_loop(mp4_file, transcription_file, num_frames_to_use, processor,
                            model,  model_name, step=1, verbose=False, init_skip_frames=5, ICL=False,
                           split_word="ASSISTANT:", k=2,  context_window=4096,):

    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

    pred_timing = []
    pred_utterences = []
    pred_utterences_step = []
    output_buffer_str = ""
    wait_count = 0
    init_str = ""
    temp = 1.0

    start_time = time.time()
    prev_elapsed = 0
    init = True
    while True:
        current_time = time.time()
        t = int(current_time - start_time)

        # 動画の長さを超えたら終了
        if t >= video_metadata["duration"]:
            break

        # 初期スキップ処理
        # print(t)
        if t < init_skip_frames:
            if init:
                data_prefix = identify_dataset(transcription_file)
                user_prompt = get_user_prompt("feedback_loop_init" + data_prefix)
                max_new_tokens = 100
                do_sample = False
                init = False
            else:
                #pred_timing.append(False)
                #pred_utterences.append("<WAIT>")
                #pred_utterences_step.append(t)
                continue
        else:
            force_flag = wait_count >= int(20 / step)
            data_prefix = identify_dataset(transcription_file)
            user_prompt = get_user_prompt("feedback_loop" + data_prefix, context=init_str, step=step, force=force_flag)
            user_prompt += "\nPrevious generated commentary: \n" + output_buffer_str + "\n\nDescribe this scene as a single-sentence commentary for making audience immersed. Please avoid repeating earlier descriptions. Do not repeat the same commentary as before. Only generate new commentary if there is a clear change or you have something to say. If you have nothing to say, generate a <WAIT> token."
            max_new_tokens = 50
            do_sample = False
            temp = 1.0 if force_flag else 1.2

        video = sample_frames(mp4_file, num_frames_to_use,
                              start_frame=(prev_elapsed) * num_frames_per_second,
                              # start_frame=(t-5)*num_frames_per_second,
                              end_frame=t * num_frames_per_second,
                              format="video")

        # ICL例の取得
        if ICL:
            icl_examples = construct_icl_examples(ICL, k=k, step=step, t=t, num_frames_to_use=num_frames_to_use)
            videos = [icl_example['video'] for icl_example in icl_examples]
        else:
            videos = []
            icl_examples = False

        videos.append(video)


        # プロンプト生成と推論
        pred_utterance = run_inference(model_name, model, processor, user_prompt, videos, ICL=icl_examples,
                                       context_window=context_window, split_word=split_word)

        prev_elapsed = t

        if "wait" in pred_utterance.lower():

            pred_timing.append(False)
            pred_utterences.append("<WAIT>")
            pred_utterences_step.append(t)
            wait_count += 1
        else:
            pred_timing.append(True)
            pred_utterences.append(pred_utterance)
            pred_utterences_step.append(t)
            output_buffer_str += f"utterance generated at {str(t)} seconds from the start: " + pred_utterance + "\n"
            wait_count = 0
            if t < init_skip_frames:
                init_str = pred_utterance

            # 🗣 語単位で話すように出力
            simulate_speaking(pred_utterance, words_per_sec=4.0)

    # 書き出しと評価
    mode = "realtime_icl_feedback_loop" if ICL else "realtime_feedback_loop"
    ref_timing_s = [ref_timing[i] for i in pred_utterences_step]
    ref_utterences_s = [ref_utterences[i]  for i in pred_utterences_step]

    pred_srt_file = write_logs(out_folder, pred_utterences, pred_utterences_step, mode=mode,
                               talking_speed_sample=icl_transcription_file)
    eval_metrics = compute_metrics(ref_timing_s, pred_timing, pred_utterences,
                                   ref_utterences_s, pred_srt_file, transcription_file)

    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences, pred_utterences_step, eval_metrics, ref_utterences_s
def baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False,init_skip_frames=5,
                           ICL = False, split_word = "ASSISTANT:", k = 2, processor = None,
                           model = None, context_window = 4096, logs_dir = None, model_name = None):
    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

    icl_transcription_file = transcription_file


    pred_timing = []
    pred_utterences = []
    pred_utterences_step = []
    output_buffer_str = ""
    wait_count = 0
    init_str = ""
    temp = 0
    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):

        #print(f"Timestep: {t}")
        #print (f"Output Buffer: {output_buffer_str}")
        video = sample_frames(mp4_file, num_frames_to_use, start_frame=(t-step+1) * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        if t < init_skip_frames:
            if t == 0:
                data_prefix = identify_dataset(transcription_file)
                user_prompt = get_user_prompt("feedback_loop_init" + data_prefix)
                max_new_tokens = 150
                do_sample = False

            else:
                pred_timing.append(False)
                pred_utterences.append("<WAIT>")
                continue
        else:
            if wait_count >= int(20/step):
                data_prefix = identify_dataset(transcription_file)
                user_prompt = get_user_prompt("feedback_loop" + data_prefix, context=init_str, step=step, force=True)
                temp = 1
            else:
                data_prefix = identify_dataset(transcription_file)
                user_prompt = get_user_prompt("feedback_loop" + data_prefix, context=init_str, step=step)
                temp = 1.2
            user_prompt += output_buffer_str
            max_new_tokens = 50
            do_sample = False
        if ICL:
            icl_examples = construct_icl_examples(ICL, k=k, step=step, t=t, num_frames_to_use=num_frames_to_use)
            videos = [icl_examples[i]['video'] for i in range(len(icl_examples))]
        else:
            videos = []
            icl_examples = False

        videos.append(video)
        pred_utterence = run_inference(model_name, model, processor, user_prompt, videos, ICL=icl_examples,
                                       context_window=context_window, split_word=split_word)


        if "WAIT" in pred_utterence:
            pred_timing.append(False)
            wait_count +=1
            pred_utterences.append("<WAIT>")
        else:
            pred_timing.append(True)

            if wait_count >= int(20 / step):
                wait_count = 0
            #else:
            previous_generation = pred_utterence
            output_buffer_str += pred_utterence
            pred_utterences.append(pred_utterence)
                #if pred_utterence[:25].strip() == previous_generation[:25].strip():
            #    pass

        pred_utterences_step.append(t)
        if t ==0:
            init_str = pred_utterence

        if t % 10 == 0 and verbose:
            print(f"{t}: {pred_utterence}")

    #pred_utterences = remove_repeatitions(pred_utterences)
    if ICL is False:
        mode = "feedback_loop"
    else:
        mode = "icl_feedback_loop"

    ref_timing = [ref_timing[ref] for ref in range(0,len(ref_timing),step)]
    ref_utterences = [ref_utterences[ref] for ref in range(0, len(ref_utterences), step)]

    pred_srt_file = write_logs(logs_dir, pred_utterences, pred_utterences_step,  mode=mode, talking_speed_sample=icl_transcription_file)
    eval_metrics = compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences, pred_srt_file, transcription_file)
    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences, pred_utterences_step, eval_metrics, ref_utterences

def contains_japanese(text):
    return bool(re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text))

def simulate_speaking(pred_utterance, words_per_sec=4.0):
    if contains_japanese(pred_utterance):
        words = list(pred_utterance.strip()) # split by character
        words_per_sec = 6.5 # average speed of japanese speech
        delay = 1.0 / words_per_sec  # 1語あたりの表示時間（秒）
    else:
        words = pred_utterance.strip().split()
        delay = 1.0 / words_per_sec  # 1語あたりの表示時間（秒）

    for word in words:
        #print(word, end=' ', flush=True)
        time.sleep(delay)
    ##print()  # 行末で改行

def extract_until_last_complete_sentence(paragraph):
    # Find the position of the last period in the text
    last_period_pos = paragraph.rfind('.')

    # If no period is found, return the whole paragraph
    if last_period_pos == -1:
        return paragraph + ". "

    # Extract text till the last period
    return paragraph[:last_period_pos + 1]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

    parser = argparse.ArgumentParser(
        description="Generates commentary as per the defined settings"
    )
    parser.add_argument("--dir", required=False, type=str, help="Directory containing the videos "
                        "and respective commentary in recordings and transcriptions_whole_data_english folder")
    parser.add_argument("--n", required=False, type=int, default=-1, help="Number of samples to run")
    parser.add_argument("--hf_dataset", required=False, type=str,
                        help="The directory containing hf_Dataset")
    parser.add_argument("--icl", required=False, type=bool, default=False, help="If ICL should be used. Currently disabled")
    parser.add_argument("--k", required=False, type=int,default=2, help="number of examples for ICL")
    parser.add_argument("--step", required=False, type=int,default=1, help="Time Step for generation")
    parser.add_argument("--wb", required=False, type=str2bool, default=True, nargs='?', const=True,
                        help="Whether or not to push results to W&B (true/false)") # the use of action="store_true" is natural for me.
    parser.add_argument("--model_name", required=False, type=str, default="llava7b",
                        help="Name of the model to be used")

    parser.add_argument("--frames", required=False, type=int, default=-1, help="Number of frames to use per step of generation")
    parser.add_argument("--context_window", required=False, type=int, default=5120,
                        help="Context Window to be used by LLM")
    parser.add_argument("--max_new_tokens", required=False, type=int, default=50, help="number of examples for ICL")
    parser.add_argument("--skip_frames", required=False, type=int, default=10, help="number of examples for ICL")


    args = parser.parse_args()
    folder = args.dir
    n = args.n
    model_name = args.model_name
    step = args.step
    k = args.k
    num_frames_to_use = args.frames
    context_window = args.context_window
    icl = args.icl
    WB = args.wb
    hf_dataset_path = args.hf_dataset
    max_new_tokens = args.max_new_tokens
    skip_frames = args.skip_frames

    if hf_dataset_path is None and folder is None:
        print (f"Either provide path to the dataset folder through --dir or path to HF dataset through --hf_dataset")
        exit()

    if WB:
        wandb_setup()

    my_folder = os.path.join("logs", date_time)
    if hf_dataset_path is None:
        hf_dataset_path = create_ds(folder)

    ds = datasets.load_from_disk(hf_dataset_path)
    test_dataset = ds['test'].with_format("torch")

    if n == -1:
        n = len(test_dataset)
    test_dataset = test_dataset.select(range(n))


    if num_frames_to_use == -1:
        num_frames_to_use = step

    #define model
    model_name_dict = {"llava7b": "llava-hf/LLaVA-NeXT-Video-7B-hf",
                       "llava34b": "llava-hf/LLaVA-NeXT-Video-34B-hf",
                       "gpt-4.1":"gpt-4o-mini-2024-07-18",
                       "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct"}

    split_word_dict = {"llava7b": "ASSISTANT:",
                       "llava34b": "<|im_start|> assistant",
                       "gpt-4.1":"",
                       "qwen7b": "assistant",
                       }

    model_type_dict = {"llava7b": "hf",
                       "llava34b": "hf",
                       "qwen7b": "hf",
                       "gpt-4.1":"closed"}

    model_id = model_name_dict[model_name]
    split_word = split_word_dict[model_name]
    model_type = model_type_dict[model_name]

    if model_type == "hf":
        if "qwen" in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, load_in_4bit=True, low_cpu_mem_usage=True
            ).to(0)
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


        else:
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,load_in_4bit=True,).to(0)
            processor = LlavaNextVideoProcessor.from_pretrained(model_id, use_fast = True)
    else:
        model = None
        processor = None
    metrics_all_samples = []

    for i in tqdm(range(len(test_dataset))):
        # get sample
        mp4_file = test_dataset[i]["video_path"]
        transcription_file = test_dataset[i]["srt_path"]

        # create folder to store logs for each sample.
        if "smabra" in hf_dataset_path.lower():
            sample_name = (os.path.basename(mp4_file)).replace('.mp4', '')
        else:
            sample_name = os.path.dirname(mp4_file).split('/')[-1]
        out_folder = os.path.join(my_folder, os.path.basename(hf_dataset_path), model_id.replace('/', '_'), sample_name, f"step_{step}_frames-used_{num_frames_to_use}_k_{k}")
        print (out_folder)
        os.makedirs(out_folder, exist_ok=True)

        # define path for icl example
        icl_index = random.sample(range(0, len(test_dataset)), k = 1)[0]
        icl_example = test_dataset[icl_index]
        icl_mp4_file = icl_example["video_path"]
        icl_transcription_file = icl_example["srt_path"]
        icl_example_paths = {'mp4_file': icl_mp4_file,
                             'transcription': icl_transcription_file}
        run_name = f"{sample_name}_step_{step}_k_{k}_frames_{num_frames_to_use}"
        try:
        #if True:

            print ("Baseline")
            baseline_generation = baseline(mp4_file, transcription_file, num_frames_to_use, step=step, split_word = split_word)

            print ("Feedback")
            feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use,
                                                              init_skip_frames=skip_frames, step=step, ICL=False,
                                                              split_word = split_word, processor=processor, model=model,
                                                              context_window=context_window, model_name=model_name
                                                              , logs_dir=out_folder
                                                              )
            
            print ("Realtime")
            realtime_loop_generation = realtime_feedback_loop(mp4_file, transcription_file, num_frames_to_use,
                                                              init_skip_frames=skip_frames, step=step,
                                                              split_word=split_word, ICL=icl_example_paths, processor=processor,
                                                              model=model, context_window=context_window, model_name=model_name)
            print ("ICL Feedback")
            icl_feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use,
                                                                  init_skip_frames=skip_frames, step=step,
                                                                  ICL=icl_example_paths, split_word = split_word,
                                                                  k = 4 , processor=processor, model=model,
                                                                  context_window=context_window, logs_dir=out_folder,
                                                                  model_name=model_name)


            run_name = f"{sample_name}_step_{step}_k_{k}_frames_{num_frames_to_use}"
            config = {"model": model_id, "step": step, "# frame": num_frames_to_use, "sample_name": sample_name, "k": k,
                      "dataset": hf_dataset_path
                      }

            metrics_per_sample = write_to_wb(run_name=run_name, baseline_output = baseline_generation, feedback_output = feedback_loop_generation,
                        icl_output = icl_feedback_loop_generation, realtime_output=realtime_loop_generation, config=config, WB = WB,
                        )
            metrics_all_samples.append(metrics_per_sample)
            with open(f'{out_folder}/{run_name}_{str(date_time)}.json', 'w') as fp:
                json.dump(metrics_per_sample, fp)
        except Exception as e:
            print (f"Caught the following exception for the sample \n Video Path:{mp4_file} \n Transcription File: {transcription_file} \n Exception: {e}")

        # Writing per experiments logs every loop
        #print(means_dict)
    # Writing per experiments logs every loop
    df = pd.DataFrame(metrics_all_samples)
    means_dict = df.select_dtypes(include='number').mean().to_dict()
    means_dict["n"] = len(df)
    means_dict["model_name"] = model_id
    means_dict["# frame"] = num_frames_to_use
    means_dict["step"] = step
    means_dict["k"] = k
    means_dict["baseline_pearson_wo_nan"] = np.nanmean(np.array(df["baseline_pearson"]))
    means_dict["feedback_pearson_wo_nan"] = np.nanmean(np.array(df["feedback_pearson"]))
    means_dict["realtime_pearson_wo_nan"] = np.nanmean(np.array(df["realtime_pearson"]))
    means_dict["icl_pearson_wo_nan"] = np.nanmean(np.array(df["icl_pearson"]))
    run_name = f"step_{step}_k_{k}_frames_{num_frames_to_use}"
    json_file = f"{hf_dataset_path.replace('/','-')}_{model_id.replace('/', '_')}_{run_name}_{str(date_time)}.json"
    print (json_file)
    with open(json_file, 'w') as fp:
        json.dump(means_dict, fp)

    if WB:
        project_name = "CommGen"
        entity = "anum-afzal-technical-university-of-munich"
        wandb_setup()
        wandb_mode = "online"

        wandb.init(project=project_name, entity=entity, config=config, name=f"g_{run_name}",
               mode=wandb_mode, group="final")
        table = wandb.Table(columns=list(means_dict.keys()),data = [list(means_dict.values())] )
        wandb.log({"experiment_metrics": table}, commit=True)
        wandb.finish()








