# models/video_model/train_video_model.py
"""
Train an LSTM video-level classifier on pre-extracted frame features (.npy).
Usage (from project root):
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
import os

# ---------------- CONFIG ----------------
FEATURE_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model\data\ffpp_features")
OUT_DIR = Path(r"C:\Users\ashta\deepfake_detector\models\video_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "best_video_model.pth"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 12
LR = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Training device: {DEVICE}")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ---------------- DATASET ----------------
class VideoFeatureDataset(Dataset):
    """Dataset of pre-extracted .npy video features"""
    def __init__(self, feature_paths):
        self.paths = feature_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        arr = np.load(path).astype(np.float32)  # (T, D)
        if arr.ndim == 3:  # (T, 1, D) → squeeze middle dim
            arr = arr.squeeze(1)
        label = 1 if "real" in path.name.lower() else 0
        return torch.from_numpy(arr), label, path.name


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    seqs, labels, names = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0], :] = s
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, lengths, labels, names


# ---------------- MODEL ----------------
class LSTMVideoClassifier(nn.Module):
    """LSTM model for video-level deepfake classification"""
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits


# ---------------- HELPERS ----------------
def discover_feature_files(feature_dir: Path):
    files = sorted([p for p in feature_dir.iterdir() if p.suffix.lower() == ".npy"])
    return files


def prepare_dataloaders(all_files):
    labels = [1 if "real" in p.name.lower() else 0 for p in all_files]
    train_files, val_files = train_test_split(all_files, test_size=0.15,
                                              stratify=labels, random_state=SEED)
    train_ds = VideoFeatureDataset(train_files)
    val_ds = VideoFeatureDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    return train_loader, val_loader


# ---------------- TRAINING ----------------
def train():
    files = discover_feature_files(FEATURE_DIR)
    if len(files) == 0:
        raise RuntimeError(f"No .npy feature files found in {FEATURE_DIR}. Run feature extraction first.")
    print(f"📁 Found {len(files)} feature files.")

    # Infer feature dimension
    sample = np.load(files[0])
    if sample.ndim == 3:
        sample = sample.squeeze(1)
    input_dim = sample.shape[1]
    print(f"🧩 Feature dimension: {input_dim}")

    # Loaders
    train_loader, val_loader = prepare_dataloaders(files)

    # Model setup
    model = LSTMVideoClassifier(input_size=input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for padded, lengths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(padded, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * padded.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += padded.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for padded, lengths, labels, _ in val_loader:
                padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
                outputs = model(padded, lengths)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += padded.size(0)
        val_acc = val_correct / max(1, val_total)

        print(f"📊 Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "epoch": epoch,
                "val_acc": val_acc
            }, BEST_PATH)
            print(f"💾 Saved best model to {BEST_PATH} (Val Acc={val_acc:.4f})")

    print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.4f}")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    train()
