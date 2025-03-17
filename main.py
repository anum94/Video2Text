import os.path
import random
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

from utils.data_utils import *
from utils.video_utils import *

def get_user_prompt(mode="baseline", context="", step = 1, force=False):
    if mode == "baseline":
        user_prompt = ("You are a professional commentator for car racing games.You will be provided with a"
                       " video interval extracted from the whole game and your task is generate brief Commentary."
                       "If there is no developments in the game, then generate <WAIT> instead of commentary."
                       "1) Identify the name of car diver through the legends provided and refer to cars by the name of the driver."
                       "2) Ignore the background information and refrain the describing the scenery."
                       "3) observe the game and briefly describe any developments.")

    elif mode == "feedback_loop_init":
        user_prompt = ("You are a professional commentator for car racing games. You will be provided with a video interval"
                       "which represents the start of a race. Your task is to generate one sentences of commentary. "
                       "1) You should identify the number of players and their names along with cars. "
                       "2) Ignore the background information and refrain the describing the scenery."
                       "3) Initial information about the game without being too verbose."
                       )

    elif mode == "feedback_loop":
        if force:
            user_prompt = ("You are a professional commentator for car racing games. You will be provided with few frames"
                           " from an on-going game and your task is generate brief Commentary."
            "1) Ignore the background information and refrain the describing the scenery."
            "2) Do not regenerate information that is already part of the Previous Commentary."
            "3) Identify new developments in the provided video as compared to previous commentary, then generate 1 sentence of commentary."
            "Previous Commentary: "
            )
        else:
            user_prompt = ("You are a professional commentator for car racing games.You will be provided with few frames"
                           "from an ongoing game and your task is generate brief Commentary for it."
                "1) Identify if the provided video has any new development as compared to the already provided commentary."
                "2) Ignore the background information and refrain the describing the scenery."
                "3) If the state of the game as compared to the provided commentary has not changed, then generate <WAIT>"
                "4) If there are new developments in the provided video, such as if a new player is in lead, or if one of the players did an "
                "impressive move then generate 1 line of commentary to describe the change."
                "Previous Commentary:"
            )

    return user_prompt


def get_utterence_timing(ground_truth,metadata):
    utterence_timing = [False] * int(metadata.get("duration"))
    utterences = [""] * int(metadata.get("duration"))
    for gt in ground_truth:
        i = srt_time_to_seconds(gt.start)
        utterence_timing[i] = True
        utterences[i] = gt.text
    return utterences, utterence_timing

def baseline(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False):

    user_prompt = get_user_prompt("baseline")
    prompt = get_messages(user_prompt, ICL=False)

    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]
    previous_generation = ""
    pred_utterences = []
    pred_utterences_step =[]
    pred_timing = []

    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):

        video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        inputs_video = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video,  do_sample=False, max_new_tokens=50)
        pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
        pred_utterence = pred_utterence.split("ASSISTANT:")[-1]
        pred_utterence = extract_until_last_complete_sentence(pred_utterence)
        if "<WAIT>" in pred_utterence:
            pred_timing.append(False)
        else:
            pred_timing.append(True)

        if pred_utterence.strip() == previous_generation.strip():
            pred_utterences.append("<WAIT>")
        else:
            pred_utterences.append(pred_utterence)
            pred_utterences_step.append(t)
            previous_generation = pred_utterence
        if t % 10 == 0 and verbose:
            print(f"{t}: {pred_utterence}")

    #pred_utterences = remove_repeatitions(pred_utterences)

    ref_timing = [ref_timing[ref] for ref in range(0,len(ref_timing),step)]
    eval_metrics = compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences)
    out_file = write_logs(out_folder, pred_utterences, pred_utterences_step, eval_metrics,  mode="baseline")

    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences

def get_messages(user_prompt, ICL = False,):
    if ICL:
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": ICL[0]['prompt']},
                    {"type": "video"},
                ],
            },
            {

                "role": "assistant",
                "content": [
                    {"type": "text", "text": ICL[0]['generation']},

                ],
            },
            {
            "role": "user",
        "content": [
            {"type": "text", "text": ICL[1]['prompt']},
            {"type": "video"},
        ],
        },
        {

            "role": "assistant",
            "content": [
                {"type": "text", "text": ICL[1]['generation']},
            ],
        },
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video"},
                ],
            },

        ]
    else:
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video"},
                ],
            },
        ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, padding=True)
    return prompt
def construct_icl_examples(example, t, k=2, step=1,num_frames_to_use = 5,skip_frames = 20,):

    transcriptions = read_srt(example['transcription'])
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(transcriptions, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

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
    user_prompt_t1 = get_user_prompt("feedback_loop", context=init_str, step=step)
    user_prompt_t1 += output_buffer_str
    generation_t1 = ref_utterences[t1]
    video_t1 = sample_frames(mp4_file, num_frames_to_use, start_frame=(t1 - step + 1) * num_frames_per_second,
                          end_frame=(t1 + 1) * num_frames_per_second, format="video")

    generate_example = {"video": video_t1, "prompt": user_prompt_t1, "generation": generation_t1}

    video_t2 = sample_frames(mp4_file, num_frames_to_use, start_frame=(t2 - step + 1) * num_frames_per_second,
                             end_frame=(t2 + 1) * num_frames_per_second, format="video")
    init_str = "".join(ref_utterences[:skip_frames])
    output_buffer_str = "".join(ref_utterences[:t2])
    user_prompt_t2 = get_user_prompt("feedback_loop", context=init_str, step=step)
    user_prompt_t2 += output_buffer_str
    generation_t2 = "<WAIT>"

    wait_example = {"video": video_t2, "prompt": user_prompt_t2, "generation": generation_t2}

    return (generate_example, wait_example)


def baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False,init_skip_frames=5, ICL = False):

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
    temp = 0
    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):

        print(f"Timestep: {t}")
        print (f"Output Buffer: {output_buffer_str}")
        video = sample_frames(mp4_file, num_frames_to_use, start_frame=(t-step+1) * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        if t < init_skip_frames:
            if t == 0:
                user_prompt = get_user_prompt("feedback_loop_init")
                max_new_tokens = 150
                do_sample = False

            else:
                pred_timing.append(False)
                pred_utterences.append("<WAIT>")
                continue
        else:
            if wait_count >= int(20/step):
                user_prompt = get_user_prompt("feedback_loop", context=init_str, step=step, force=True)
                temp = 1
            else:
                user_prompt = get_user_prompt("feedback_loop", context=init_str, step=step)
                temp = 1
            user_prompt += output_buffer_str
            max_new_tokens = 25
            do_sample = False
        if ICL:
            icl_examples = construct_icl_examples(ICL, k=2, step=step, t=t, num_frames_to_use=num_frames_to_use)
            videos = [icl_examples[0]['video'], icl_examples[1]['video'], video]
        else:
            icl_examples = False
            videos = [video]
        prompt = get_messages(user_prompt=user_prompt, ICL=icl_examples)
        inputs_video = processor(text=prompt, padding = True, videos=videos, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature = temp)
        pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
        pred_utterence = pred_utterence.split("ASSISTANT:")[-1]
        pred_utterence = extract_until_last_complete_sentence(pred_utterence)
        print(pred_utterence)


        if "<WAIT>" in pred_utterence:
            pred_timing.append(False)
            wait_count +=1
        else:
            pred_timing.append(True)

            if wait_count >= int(20 / step):
                wait_count = 0
            #else:
            previous_generation = pred_utterence
            output_buffer_str += pred_utterence
                #if pred_utterence[:25].strip() == previous_generation[:25].strip():
            #    pass
        pred_utterences.append(pred_utterence)
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

    eval_metrics = compute_metrics(ref_timing, pred_timing, pred_utterences, ref_utterences)
    out_file = write_logs(out_folder, pred_utterences, pred_utterences_step, eval_metrics,  mode=mode)

    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences


def extract_until_last_complete_sentence(paragraph):
    # Find the position of the last period in the text
    last_period_pos = paragraph.rfind('.')

    # If no period is found, return the whole paragraph
    if last_period_pos == -1:
        return paragraph + ". "

    # Extract text till the last period
    return paragraph[:last_period_pos + 1]
if __name__ == '__main__':
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

    folder = os.path.join("logs", date_time)
    if len(sys.argv) > 3:
        folder = sys.argv[1]
        n = int(sys.argv[2])
        step = int(sys.argv[3])
    elif len(sys.argv) > 2:
        folder = sys.argv[1]
        n = int(sys.argv[2])
        step = None
    elif len(sys.argv) > 1:
        folder = sys.argv[1]
        n = None
        step = None
    else:
        print("Usage: python main.py path/to/folder/containing/data")
        sys.exit(1)


video_directory = "recordings"
video_directory = os.path.join(folder,video_directory)

commentary_directory = "transcriptions_whole_data_english"
commentary_directory = os.path.join(folder,commentary_directory)


all_game_path = [os.path.join(video_directory,name) for name in os.listdir(video_directory) if os.path.isdir(os.path.join(video_directory, name))]
if n is None:
    n = len(all_game_path)
for i, game_path in enumerate(all_game_path[:n]):
    transcription_file = get_commentary_path(commentary_directory,game_path)
    if transcription_file is not None:
        mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
                     file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
    else:
        print (f"kyakkan commentary not available for game: {game_path}")
        continue
icl_path = os.path.join(video_directory,
                           "AC_120221-180622_R_ks_audi_r8_plus_ks_nurburgring_layout_sprint_a"
                           )
icl_mp4_file = mp4_file = [os.path.join(icl_path,file)
                           for file in os.listdir(icl_path) if
                     file.endswith('.mp4') and os.path.isfile(os.path.join(icl_path, file)) and "客観" in file][0]
icl_transcription_file = transcription_file = get_commentary_path(commentary_directory,icl_path)
icl_example_paths = {'mp4_file':icl_mp4_file,
               'transcription': icl_transcription_file}
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"


model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

#model = None
processor = LlavaNextVideoProcessor.from_pretrained(model_id)


# Baseline without feedback loop

num_frames_to_use = 6
max_new_tokens = 50
if step is None:
    step = 1
skip_frames = 20

sample_name = os.path.dirname(mp4_file).split('/')[-1]
out_folder = os.path.join(folder, sample_name, f"step_{step}")
os.makedirs(out_folder, exist_ok=True)

baseline_generation = baseline(mp4_file, transcription_file, num_frames_to_use, step=step)

feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, init_skip_frames=skip_frames, step=step, ICL=False)

icl_feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, init_skip_frames=skip_frames, step=step, ICL=icl_example_paths)





