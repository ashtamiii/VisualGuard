# models/video_model/infer_video_gradcam.py
"""
Video Grad-CAM explainability for deepfake videos.
Outputs:
  - Per-frame Grad-CAM overlays
  - Contact sheet
  - Final prediction (REAL/FAKE)

Usage:
  python infer_video_gradcam.py --video "path/to/video.mp4"
"""

from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import shutil

# ---------------- CONFIG ----------------
FRAME_SAMPLE_COUNT = 16
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_ROOT = Path(r"C:\Users\ashta\deepfake_detector\results")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_IMAGE_MODEL_PATHS = [
    PROJECT_ROOT / "models" / "image_model" / "best_model.pth",
    PROJECT_ROOT / "models" / "models" / "image_model" / "best_model.pth",
    Path(r"C:\Users\ashta\deepfake_detector\models\models\image_model\best_model.pth"),
]

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------- GRAD-CAM IMPLEMENTATION ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        # forward hook
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        # backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())

        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        # tensors
        grads = self.gradients[0]    # (C, H, W)
        acts = self.activations[0]   # (C, H, W)

        # channel-wise weight
        weights = grads.mean(dim=(1, 2))  # (C,)

        # weighted sum
        cam = torch.zeros_like(acts[0])
        for c, w in enumerate(weights):
            cam += w * acts[c]

        # ReLU
        cam = torch.relu(cam)

        # normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # convert to numpy
        cam = cam.cpu().numpy()

        # resize to image size
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))

        return cam


# ---------------- MODEL LOADING ----------------
def find_image_model():
    for p in CANDIDATE_IMAGE_MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("❌ Image model not found.\nChecked:\n" +
                            "\n".join(str(p) for p in CANDIDATE_IMAGE_MODEL_PATHS))


def load_image_classifier(ckpt_path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE).eval()

    return model


# ---------------- FRAME SAMPLING ----------------
def sample_frames(video_path, out_dir, count=FRAME_SAMPLE_COUNT):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Cannot read video frames.")

    step = max(total // count, 1)

    frames = []
    idx = 0
    saved = 0

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            fp = out_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(fp), frame)
            frames.append(fp)
            saved += 1

        idx += 1

    cap.release()
    return frames


# ---------------- CAM OVERLAY ----------------
def overlay_cam(pil_img, cam_mask):
    img = pil_img.resize(IMG_SIZE)
    img_np = np.array(img)

    heat = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    out = (0.6 * heat + 0.4 * img_np).astype(np.uint8)
    return Image.fromarray(out)


# ---------------- CONTACT SHEET ----------------
def make_contact_sheet(paths, out_path, cols=4):
    if len(paths) == 0:
        return None

    imgs = [Image.open(p).resize(IMG_SIZE) for p in paths]

    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB",
                      (cols * IMG_SIZE[0], rows * IMG_SIZE[1]),
                      color=(255, 255, 255))

    for i, im in enumerate(imgs):
        x = (i % cols) * IMG_SIZE[0]
        y = (i // cols) * IMG_SIZE[1]
        sheet.paste(im, (x, y))

    sheet.save(str(out_path))
    return out_path


# ---------------- MAIN EXPLAIN FUNCTION ----------------
def infer_and_explain(video_path, results_root, sample_count):
    video_path = Path(video_path)
    results_root = Path(results_root)

    # ---------------- LOAD MODEL ----------------
    ckpt = find_image_model()
    print("Using image checkpoint:", ckpt)

    model = load_image_classifier(ckpt)
    target_layer = model.layer4[-1]
    camgen = GradCAM(model, target_layer)

    # ---------------- OUTPUT FOLDERS ----------------
    out_dir = results_root / video_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True)

    # ---------------- SAMPLE FRAMES ----------------
    print("Sampling frames...")
    frames = sample_frames(video_path, frame_dir, sample_count)
    print(f"Extracted {len(frames)} frames → {frame_dir}")

    probs_list = []
    overlay_paths = []
    label_map = {0: "FAKE", 1: "REAL"}

    # ---------------- PROCESS FRAMES ----------------
    for i, fp in enumerate(frames):
        pil = Image.open(fp).convert("RGB")
        inp = transform(pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        probs_list.append(probs)

        cam = camgen.generate_cam(inp, pred_idx)

        overlay = overlay_cam(pil, cam)
        save_fp = out_dir / f"frame_{i:03d}_overlay.jpg"
        overlay.save(save_fp)
        overlay_paths.append(save_fp)

        print(f"Frame {i:02d}: {label_map[pred_idx]} ({probs[pred_idx]*100:.2f}%)")

    # ---------------- FINAL VIDEO PRED ----------------
    arr = np.stack(probs_list)
    avg_probs = arr.mean(axis=0)

    final_idx = int(np.argmax(avg_probs))
    final_label = label_map[final_idx]
    final_conf = avg_probs[final_idx] * 100

    print("\n======== FINAL RESULT ========")
    print(f"Video: {video_path.name}")
    print(f"Prediction: {final_label} ({final_conf:.2f}%)")
    print("================================\n")

    # ---------------- CONTACT SHEET ----------------
    sheet_path = results_root / f"{video_path.stem}_contact.jpg"
    make_contact_sheet(overlay_paths, sheet_path)

    print("Overlays saved to:", out_dir)
    print("Contact sheet saved:", sheet_path)


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--samples", type=int, default=FRAME_SAMPLE_COUNT)
    parser.add_argument("--out", default=str(RESULTS_ROOT))

    args = parser.parse_args()

    infer_and_explain(args.video, args.out, args.samples)
