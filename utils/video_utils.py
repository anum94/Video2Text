import cv2
import torch
import numpy as np
from PIL import Image
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

    if end_frame - start_frame < num_frames:
        start_frame = end_frame - num_frames
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

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

def process_video_cv2(video: cv2.VideoCapture, indices: np.array, length: int):
    index = 0
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if index in indices:
            # Channel 0:B 1:G 2:R
            height, width, channel = frame.shape
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= length:
            break

    video.release()
    return frames