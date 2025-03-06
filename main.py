from transformers import LlavaProcessor, LlavaForConditionalGeneration
import os
import cv2
import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.data_utils import read_srt, srt_time_to_seconds
from datetime import datetime
from PIL import Image
import torch
import sys
import warnings
warnings.filterwarnings("ignore")
def get_utterence_timing(ground_truth,metadata):
    utterence_timing = [False] * int(metadata.get("duration"))
    utterences = []
    for gt in ground_truth:
        i = srt_time_to_seconds(gt.start)
        utterence_timing[i] = True
        utterences.append(gt.text)
    return utterences, utterence_timing

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

def get_video_info(path):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = int(total_frames/fps)
    return {"total_frames": total_frames, "frames_per_second": fps,
            "duration":duration }
def sample_frames(path, num_frames, start_frame = None, end_frame = None, format = "images"):

    video = cv2.VideoCapture(path)
    if start_frame is not None:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_frame is None:
        end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame - end_frame < num_frames:
        print ()
    interval = (end_frame - start_frame) // num_frames
    frames = []
    take_next_frame = False


    for i in range(end_frame - start_frame):
        try:
            ret, frame = video.read()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not ret:
                continue
            if i % interval == 0:
                frames.append(pil_img)
            if take_next_frame:
                frames.append(pil_img)
                take_next_frame = False
        except Exception as e:
            print (f"Failed to get frame number {i} for video: {path} due to Exception {e} \n Using the next frame instead.")
            if i % interval == 0:
                take_next_frame = True

    video.release()
    if format == "video":
        frames = [np.array(frame) for frame in frames]
        frames = np.stack(frames, axis=0)
    return frames
def get_commentary_path(game_path):
    game_path = os.path.basename(game_path)
    commentary_path = [os.path.join(commentary_directory, file) for file in os.listdir(commentary_directory) if
     file.endswith('.srt') and os.path.isfile(os.path.join(commentary_directory, file)) and "kyakkan" in file
         and game_path in file]
    if len(commentary_path) > 0:
        commentary_path = commentary_path[0]
    else:
        commentary_path = None
    return commentary_path

if __name__ == '__main__':
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

    out_folder = os.path.join("logs", date_time)
    os.makedirs(out_folder, exist_ok=True)
    if len(sys.argv) > 2:
        folder = sys.argv[1]
        n = int(sys.argv[2])
    elif len(sys.argv) > 1:
        folder = sys.argv[1]
        n = None
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
    transcription_file = get_commentary_path(game_path)
    if transcription_file is not None:
        mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
                     file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
    else:
        print (f"kyakkan commentary not available for game: {game_path}")
        continue


#transcription_file = "transcriptions_whole_data_english/AC_150221-130155_R_ks_porsche_macan_mugello__kyakkan.merged.mp4_translated.srt"
#transcription_file = os.path.join(folder, transcription_file)
#mp4_file = "AC_150221-130155_R_ks_porsche_macan_mugello_/AC_150221-130155_R_ks_porsche_macan_mugello_客観.mp4"
#mp4_file = os.path.join(video_directory, mp4_file)
ground_truth = read_srt(transcription_file)
video_metadata = get_video_info(mp4_file)
ref_utterences, ref_timing = get_utterence_timing(ground_truth, video_metadata)
num_frames_to_use = 3
num_frames_per_second = video_metadata["frames_per_second"]
pred_utterences = []
pred_timing = []
user_prompt = ("You are a professional commentator for car racing games. You will be provided with few seconds"
                   "interval video extracted from the whole game and your task is to either generate one sentence "
                   "regarding the current state of the game or generate a <WAIT> if there us no development in the state"
                   "of the game. Please observe the state in terms of the car shown and the associated players. Ignore the "
                   "background information and avoid from describing the scene. Just explain the game.")
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"


model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
processor = LlavaNextVideoProcessor.from_pretrained(model_id)

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
for t in tqdm(range(video_metadata["duration"]), total = video_metadata["duration"]):

    video = sample_frames(mp4_file, num_frames_to_use, start_frame=t*num_frames_per_second, end_frame=(t+1)*num_frames_per_second, format="video")

    inputs_video = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    pred_utterence = processor.decode(output[0][2:], skip_special_tokens=True)
    pred_utterence = pred_utterence.split("ASSISTANT:")[-1]
    if "<WAIT>" in pred_utterence:
        pred_timing.append(False)
    else:
        pred_timing.append(True)

    pred_utterences.append(pred_utterence)
    if t % 10 == 0:
        print(f"{t}: {pred_utterence}")

complete_commentary = ""
previous = ""
for pred_utterence in pred_utterences:
    if previous != pred_utterence:
        complete_commentary += pred_utterence
    previous = pred_utterence
print (f"Complete Commentary: {complete_commentary}")
out_file = os.path.join(out_folder, "logs.txt")
print (f"Generation stored at {out_file}")
with open(out_file, 'a') as the_file:
    for t, ut in enumerate(pred_utterences):
        the_file.write(f"{t}: {ut}\n")
correlations = [1 if a == b else 0 for a, b in zip(ref_timing, pred_timing)]
cm = confusion_matrix(ref_timing, pred_timing)
print(correlations.count(1))
print(cm)







