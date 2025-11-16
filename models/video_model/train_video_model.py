# models/video_model/train_video_model.py
"""
Train an LSTM video-level classifier on pre-extracted frame features (.npy).

Usage:
    python models/video_model/train_video_model.py
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\data\ffpp_features")
OUT_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_video_model.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Training on device: {DEVICE}")


# ============================================================
# SEED BEHAVIOR
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class VideoFeatureDataset(Dataset):
    """Each item is .npy file containing features: shape (T,512) or (T,1,512)."""

    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        arr = np.load(p).astype(np.float32)

        # Fix shape if (T,1,512) → (T,512)
        if arr.ndim == 3:
            arr = arr.squeeze(1)

        label = 1 if "real" in p.name.lower() else 0
        return torch.tensor(arr), label, p.name


def collate_fn(batch):
    """Pads variable-length sequences."""

    sequences, labels, names = zip(*batch)
    lengths = [s.shape[0] for s in sequences]

    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, feat_dim)
    for i, s in enumerate(sequences):
        padded[i, :s.shape[0]] = s

    return padded.float(), torch.tensor(lengths), torch.tensor(labels), names


# ============================================================
# MODEL
# ============================================================

class LSTMVideoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        final_hidden = h_n[-1]
        logits = self.fc(final_hidden)
        return logits


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model():
    files = sorted([p for p in FEATURE_DIR.iterdir() if p.suffix == ".npy"])
    if len(files) == 0:
        raise RuntimeError(" No .npy feature files found. Run extract_features.py first.")

    print(f" Found {len(files)} feature files.")

    # Determine input dimension
    sample = np.load(files[0])
    if sample.ndim == 3:
        sample = sample.squeeze(1)

    input_dim = sample.shape[1]
    print(f" Feature dimension detected: {input_dim}")

    # Split
    labels = [1 if "real" in p.name.lower() else 0 for p in files]
    train_files, val_files = train_test_split(
        files, test_size=0.15, stratify=labels, random_state=SEED
    )

    # DataLoaders
    train_loader = DataLoader(
        VideoFeatureDataset(train_files),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        VideoFeatureDataset(val_files),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model, loss, optimizer
    model = LSTMVideoClassifier(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # ============================================================
    # EPOCH LOOP
    # ============================================================

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for padded, lengths, labels, _names in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(padded, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * padded.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += padded.size(0)

        avg_loss = train_loss / total
        train_acc = correct / total

        # ---------------- Validate ----------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for padded, lengths, labels, _ in val_loader:
                padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
                preds = model(padded, lengths).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += padded.size(0)

        val_acc = val_correct / val_total

        print(f" Epoch {epoch}: Train Loss={avg_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        # ---------------- Save best model ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "epoch": epoch,
                "val_acc": val_acc
            }, BEST_MODEL_PATH)
            print(f" Saved new BEST model → {BEST_MODEL_PATH} (Val Acc={val_acc:.4f})")

    print(f"\n Training complete! Best Validation Accuracy: {best_val_acc:.4f}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()
