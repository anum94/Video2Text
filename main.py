from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

from utils.data_utils import *
from utils.video_utils import *

def get_user_prompt(mode="baseline", context="", step = 1):
    if mode == "baseline":
        user_prompt = ("You are a professional commentator for car racing games. You will be provided with few seconds"
                       "interval video extracted from the whole game and your task is to either generate one sentence "
                       "regarding the current state of the game or generate a <WAIT> if there is no developments in the game. "
                       "You should focus on the following without being too verbose: "
                       "1)Identify the name of car diver through the legends provided and refer to cars by the name of the driver."
                       "2) observe the game as a professional commentator would and decribe any developments."
                       "3) Ignore the background information and refrain the describing the scenery. Just explain the game.")

    elif mode == "feedback_loop_init":
        user_prompt = ("You are a professional commentator for car racing games. You will be provided with a video interval"
                       "which represents the beginning of a race. Your task is to generate one to two sentences of commentary "
                       "which describe the game in terms of number of players, their names and their cars. You should: "
                       "1)Identify the name of car diver through the legends provided and refer to cars by the name of the driver."
                       "2) Ignore the background information and refrain the describing the scenery. Just provide some brief"
                       "initial information about the game without being too verbose. \nCommentary: ")

    elif mode == "feedback_loop":
        user_prompt = (
            f"You are a professional commentator for car racing games and you are currently generating commentary for the following game: {context}"
            f"You will be provided with a short video interval depicting the state of the game at a given interval of {step} seconds along with"
            "some text that summarizes the game before the provided time interval."
            "As you might know, sometimes commentators stay silent for a few seconds during the game."
            "Your task is to first decide if you should say something for the provided time interval or choose to wait for some developments in the games"
            "If you choose to stay quite then simply generate <WAIT>, otherwise generate one or two sentences "
            "of commentary without being too verbose." 
            "1) Identify if the provided video has any new development as compared to the already provided commentary."
            "2) Ignore the background information and refrain the describing the scenery."
            "3) If the state of the game as compared to the provided commentary has not changed, then generate <WAIT>"
            "4) If there are new developments in the provided video, then generate 1 - 2 lines of commentary."
            "Previous Commentary:"
            )

    return user_prompt


def get_utterence_timing(ground_truth,metadata):
    utterence_timing = [False] * int(metadata.get("duration"))
    utterences = []
    for gt in ground_truth:
        i = srt_time_to_seconds(gt.start)
        utterence_timing[i] = True
        utterences.append(gt.text)
    return utterences, utterence_timing

def baseline(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False):

    user_prompt = get_user_prompt("baseline")
    prompt = get_messages(user_prompt, ICL=False)

    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

    pred_utterences = []
    pred_timing = []

    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):

        video = sample_frames(mp4_file, num_frames_to_use, start_frame=t * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        inputs_video = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=False)
        pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
        pred_utterence = pred_utterence.split("ASSISTANT:")[-1]
        if pred_utterence.strip() == "<WAIT>" or pred_utterence.strip() == "<WAIT> The provided video interval shows no new developments compared to the already provided commentary. Therefore, I will stay quiet and not generate any commentary for this interval.":
            pred_timing.append(False)
        else:
            pred_timing.append(True)

        if pred_utterence[:20].strip() == pred_utterences[-1][:20].strip():
            pred_utterences.append("<WAIT>")
        else:
            pred_utterences.append(pred_utterence)
        if t % 10 == 0 and verbose:
            print(f"{t}: {pred_utterence}")

    #pred_utterences = remove_repeatitions(pred_utterences)
    out_file = write_logs(out_folder, pred_utterences)

    eval_metrics = compute_metrics(ref_timing, pred_timing)

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

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt
def baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, step = 1, verbose = False,init_skip_frames=5, ICL = False):

    ground_truth = read_srt(transcription_file)
    video_metadata = get_video_info(mp4_file)
    ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
    num_frames_per_second = video_metadata["frames_per_second"]

    pred_timing = []
    pred_utterences = []
    output_buffer_str = ""
    previous_generation = ""
    init_str = ""
    for t in tqdm(range(0,video_metadata["duration"],step), total=video_metadata["duration"]/step):

        print(f"Timestep: {t}")
        print (f"Output Buffer: {output_buffer_str}")
        video = sample_frames(mp4_file, num_frames_to_use, start_frame=(t-step+1) * num_frames_per_second,
                              end_frame=(t + 1) * num_frames_per_second, format="video")

        if t < init_skip_frames:
            if t == 0:
                user_prompt = get_user_prompt("feedback_loop_init")
                max_new_tokens = 200

            else:
                pred_timing.append(False)
                pred_utterences.append("<WAIT>")
                continue
        else:
            user_prompt = get_user_prompt("feedback_loop", context=init_str, step=step)
            user_prompt += output_buffer_str
            max_new_tokens = 50
        prompt = get_messages(user_prompt=user_prompt, ICL=ICL)


        inputs_video = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=False)
        pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
        print(pred_utterence)
        pred_utterence = pred_utterence.split("ASSISTANT:")[-1]

        if "<WAIT>" in pred_utterence:
            pred_timing.append(False)
        else:
            pred_timing.append(True)
            previous_generation = pred_utterence
            output_buffer_str += pred_utterence
            #if pred_utterence[:25].strip() == previous_generation[:25].strip():
            #    pass
        pred_utterences.append(pred_utterence)
        if t ==0:
            init_str = pred_utterence

        if t % 10 == 0 and verbose:
            print(f"{t}: {pred_utterence}")

    #pred_utterences = remove_repeatitions(pred_utterences)
    out_file = write_logs(out_folder, pred_utterences, mode = "feedback_loop")
    ref_timing = [ref for ref in range(0,len(ref_timing),step)]
    print(len(ref_timing), len(pred_timing))
    eval_metrics = compute_metrics(ref_timing, pred_timing)

    if verbose:
        print(eval_metrics)
        print(f"Complete Commentary: {pred_utterences}")

    return pred_utterences


if __name__ == '__main__':
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

    out_folder = os.path.join("logs", date_time)
    os.makedirs(out_folder, exist_ok=True)
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
for game_path in all_game_path[:n]:
    transcription_file = get_commentary_path(commentary_directory,game_path)
    if transcription_file is not None:
        mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
                     file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
    else:
        print (f"kyakkan commentary not available for game: {game_path}")
        continue

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"


model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
processor = LlavaNextVideoProcessor.from_pretrained(model_id)


# Baseline without feedback loop

num_frames_to_use = 3
max_new_tokens = 50
if step is None:
    step = 2


baseline_generation = baseline(mp4_file, transcription_file, num_frames_to_use, step=10)

#baseline_feedback_loop_generation = baseline_feedback_loop(mp4_file, transcription_file, num_frames_to_use, init_skip_frames=10, step=step, ICL=False)






