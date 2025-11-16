# File: models/video_model/infer_video.py
"""
Simple video-level inference using:
- image model (ResNet feature extractor saved at models/image_model/best_model.pth)
- video model (LSTM saved at models/video_model/best_video_model.pth)

Usage:
  python models/video_model/infer_video.py --video "C:\path\to\video.mp4"
"""
import argparse
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

# CONFIG — adjust paths if needed
IMAGE_MODEL_PATH = Path(r"C:\Users\ashta\deepfake_detector\models\models\image_model\best_model.pth")
VIDEO_MODEL_PATH = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\best_video_model.pth")
TEMP_FRAMES_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\data\temp_frames")
FRAME_SAMPLE_COUNT = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess same as training
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---- Helper utilities ----
def extract_frames(video_path: Path, out_dir: Path, sample_count=FRAME_SAMPLE_COUNT):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        raise RuntimeError("Couldn't read video or no frames.")
    step = max(total // sample_count, 1)
    saved = 0
    idx = 0
    while saved < sample_count and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            path = out_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        idx += 1
    cap.release()
    return sorted(out_dir.glob("frame_*.jpg"))

# ---- Models ----
def load_image_feature_extractor():
    # build resnet18 backbone and load weights; remove fc to get features
    model = models.resnet18()
    model.fc = nn.Identity()
    ckpt = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
    # ckpt could be state_dict or full dict
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    # try load (allow partial)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

class LSTMVideoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x, lengths):
        # x: (B, T, D)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]
        return self.fc(last)

def load_video_model():
    ckpt = torch.load(VIDEO_MODEL_PATH, map_location=DEVICE)
    input_dim = ckpt.get("input_dim", 512) if isinstance(ckpt, dict) else 512
    model = LSTMVideoClassifier(input_size=input_dim)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.to(DEVICE).eval()
    return model

# ---- Inference ----
def predict_video(video_path: Path, out_dir: Path = Path("results/video_infer")):
    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Image model not found: {IMAGE_MODEL_PATH}")
    if not VIDEO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Video model not found: {VIDEO_MODEL_PATH}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cleanup temp frames
    TEMP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in TEMP_FRAMES_DIR.glob("*.jpg"):
        try: f.unlink()
        except: pass

    frames = extract_frames(video_path, TEMP_FRAMES_DIR, FRAME_SAMPLE_COUNT)
    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")

    img_model = load_image_feature_extractor()
    vid_model = load_video_model()

    feats = []
    for p in frames:
        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = img_model(tensor)  # (1, D)
        feats.append(feat.cpu().numpy().squeeze(0))
    feats = np.stack(feats)  # (T, D)

    # convert to torch (batch size 1)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, D)
    lengths = torch.tensor([feats.shape[0]]).to(DEVICE)
    with torch.no_grad():
        logits = vid_model(x, lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100.0

    label = "REAL" if pred == 1 else "FAKE"
    print(f"Prediction: {label} ({conf:.2f}% confidence)")
    return {"label": label, "confidence": conf, "frames_used": len(frames)}

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--out", default="results/video_infer", help="Output directory")
    args = parser.parse_args()
    predict_video(Path(args.video), Path(args.out))
