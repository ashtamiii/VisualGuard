# models/video_model/extract_features.py
"""
Extract per-frame ResNet features for each video (real/fake) and save as .npy arrays.
Each .npy file will have shape (num_frames, 512).
Usage:
    python models/video_model/extract_features.py
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# ---------------- CONFIG ----------------
# Directories
FRAME_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\data\ffpp_frames")
OUTPUT_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\data\ffpp_features")
MODEL_PATH = Path(r"C:\Users\ashta\deepfake_detector\models\models\image_model\best_model.pth")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")

# Ensure output folder exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Transform (same as image training)
transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------------- MODEL LOADER ----------------
def load_feature_extractor():
    """Load pretrained image model (ResNet18 backbone without classifier)."""
    model = models.resnet18()
    model.fc = nn.Identity()  # remove classification layer
    weights = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------- FEATURE EXTRACTOR ----------------
def extract_video_features(model, video_dir: Path):
    """Extract ResNet features for each frame in a video directory."""
    frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not frame_files:
        return None

    features = []
    for frame_path in frame_files:
        img = Image.open(frame_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(tensor)  # (1, 512)
        features.append(feat.cpu().numpy().squeeze())  # -> (512,)
    return np.stack(features)  # -> (num_frames, 512)


def process_all_videos(label: str, model):
    """Process all videos under a given label (real/fake)."""
    label_dir = FRAME_DIR / label
    if not label_dir.exists():
        print(f" Directory not found: {label_dir}")
        return

    for video_folder in tqdm(sorted(label_dir.iterdir()), desc=f"Extracting {label} features"):
        if not video_folder.is_dir():
            continue
        out_name = f"{label}_{video_folder.name}.npy"
        out_path = OUTPUT_DIR / out_name

        # Skip if already done
        if out_path.exists():
            continue

        feats = extract_video_features(model, video_folder)
        if feats is None:
            print(f"Skipping {video_folder} (no frames found).")
            continue

        np.save(out_path, feats)
    print(f" {label.capitalize()} feature extraction complete.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    model = load_feature_extractor()

    print(" Extracting features from frames...")
    process_all_videos("real", model)
    process_all_videos("fake", model)

    print(f"\n Feature extraction complete. Features saved to: {OUTPUT_DIR}")

