from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import os
import cv2
from PIL import Image
import sys
from tqdm import tqdm
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
    take_next_frame = False
    for i in tqdm(range(total_frames), total=total_frames):
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
    return frames[:num_frames]
def get_commentary_path(game_path):
    game_path = os.path.basename(game_path)
    commentary_path = [os.path.join(commentary_directory, file) for file in os.listdir(commentary_directory) if
     file.endswith('.srt') and os.path.isfile(os.path.join(commentary_directory, file)) and "kyakkan" in file
         and game_path in file][0]
    return commentary_path



if __name__ == '__main__':
    if len(sys.argv) > 2:
        folder = sys.argv[1]
        n = int(sys.argv[1])
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
    mp4_file = [os.path.join(game_path,file) for file in os.listdir(game_path) if
                 file.endswith('.mp4') and os.path.isfile(os.path.join(game_path, file)) and "客観" in file][0]
    transcription_file = get_commentary_path(game_path)


#transcription_file = "transcriptions_whole_data_english/AC_150221-130155_R_ks_porsche_macan_mugello__kyakkan.merged.mp4_translated.srt"
#transcription_file = os.path.join(folder, transcription_file)
#mp4_file = "AC_150221-130155_R_ks_porsche_macan_mugello_/AC_150221-130155_R_ks_porsche_macan_mugello_客観.mp4"
#mp4_file = os.path.join(video_directory, mp4_file)
video = sample_frames(mp4_file, 100)

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

