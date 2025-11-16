import cv2
import os
from tqdm import tqdm

# Input folders
REAL_VIDEOS = r"C:\Users\ashta\deepfake_detector\dataset\face++dataset\ffpp_real"
FAKE_VIDEOS = r"C:\Users\ashta\deepfake_detector\dataset\face++dataset\ffpp_fake"

# Output folder
OUTPUT_DIR = r"data\ffpp_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many frames per video to sample
FRAME_SAMPLE_COUNT = 16

def extract_frames_from_video(video_path, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return
    interval = max(frame_count // FRAME_SAMPLE_COUNT, 1)
    frame_idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0 and saved < FRAME_SAMPLE_COUNT:
            frame_path = os.path.join(output_subdir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        frame_idx += 1
    cap.release()

def process_all_videos(input_dir, label):
    for video_name in tqdm(os.listdir(input_dir), desc=f"Processing {label}"):
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        video_path = os.path.join(input_dir, video_name)
        output_subdir = os.path.join(OUTPUT_DIR, label, os.path.splitext(video_name)[0])
        extract_frames_from_video(video_path, output_subdir)

if __name__ == "__main__":
    process_all_videos(REAL_VIDEOS, "real")
    process_all_videos(FAKE_VIDEOS, "fake")
    print(" Frame extraction complete.")
