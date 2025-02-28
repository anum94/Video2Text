from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import os
import cv2
from PIL import Image

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

def get_video_info(path):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames
def sample_frames(path, num_frames):

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]
def get_commentary_path(game_path):
    game_path = os.path.basename(game_path)
    commentary_path = [os.path.join(commentary_directory, file) for file in os.listdir(commentary_directory) if
     file.endswith('.srt') and os.path.isfile(os.path.join(commentary_directory, file)) and "kyakkan" in file
         and game_path in file][0]
    return commentary_path
video_directory = "/Users/anumafzal/PycharmProjects/video2Text/RaceCommentary/recordings"
commentary_directory = "/Users/anumafzal/PycharmProjects/video2Text/RaceCommentary/transcriptions_whole_data_english"

#all_game_path = [os.path.join(video_directory,name) for name in os.listdir(video_directory) if os.path.isdir(os.path.join(video_directory, name))]
#for game_path in all_game_path:
#    mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
#                 file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
#    transcription_file = get_commentary_path(game_path)


transcription_file = "transcriptions_whole_data_english/AC_150221-130155_R_ks_porsche_macan_mugello__kyakkan.merged.mp4_translated.srt"
mp4_file = "recordings/AC_150221-130155_R_ks_porsche_macan_mugello_/AC_150221-130155_R_ks_porsche_macan_mugello_客観.mp4"
video = sample_frames(mp4_file, 6)

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda")

user_prompt = "Please generate a commentary in english for the provided video?"
toks = "<image>" * 12
prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
inputs = processor(text=prompt, images=video, return_tensors="pt").to(model.device, model.dtype)

output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:])

