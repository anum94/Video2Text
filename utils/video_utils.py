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
def write_video(video_array, path, fps):
    width = video_array.shape[2]
    height = video_array.shape[1]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'mp4v' for .mp4 files
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i, frame in enumerate(video_array):
        out.write(frame)

    out.release()
    return path
def process_video(video, num_frames, height=1080, width=1920, channels=3):
    frames = []
    for frame in video:
        # Resize to target size and ensure 3 channels
        frame_resized = cv2.resize(frame, (width, height))
        if frame_resized.shape[2] != channels:
            frame_resized = frame_resized[:,:,:channels]  # or convert as needed
        frames.append(frame_resized)
    video_fixed = np.stack(frames)  # (nf, h, w, c)
    # Truncate or pad to the fixed number of frames
    nf = video_fixed.shape[0]
    if nf < num_frames:
        pad = np.zeros((num_frames - nf, height, width, channels), dtype=video_fixed.dtype)
        video_fixed = np.concatenate([video_fixed, pad], axis=0)
    else:
        video_fixed = video_fixed[:num_frames]
    return video_fixed
def read_video(video_path):
    print (video_path)
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Optionally, convert BGR to RGB if you need it in standard format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Convert list of frames to a numpy array (N, H, W, C)
    video_np = np.array(frames)
    print (video_np.shape)


    #print("Shape of video numpy array:", video_np.shape)  # e.g., (num_frames, height, width, 3)
    return video_np


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