"""
╔══════════════════════════════════════════════════════════════╗
║   CUSTOMER EMOTION TRACKER — PYTORCH DEEP LEARNING          ║
║   Text Classification with Neural Network + Epochs          ║
║   U24IT401 – AI & ML  |  Meenakshi Sundararajan Engg College ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, re, json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE     = torch.device("cpu")   # Intel Iris Xe → CPU
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 0.001
MAX_FEATS  = 10000

EMOTION_MAP = {
    "joy":      "Happy",
    "anger":    "Angry",
    "sadness":  "Sad",
    "fear":     "Frustrated",
    "surprise": "Excited",
    "love":     "Happy",
}

# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]|#\w+|@\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return " ".join(text.split())

# ═══════════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load real emotions dataset from txt files."""
    print("\n" + "═"*60)
    print("  LOADING DATASET")
    print("═"*60)

    files = {
        "train": os.path.join(DATASET_DIR, "train.txt"),
        "val":   os.path.join(DATASET_DIR, "val.txt"),
        "test":  os.path.join(DATASET_DIR, "test.txt"),
    }

    all_samples = []
    for split, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ";" not in line:
                    continue
                parts = line.rsplit(";", 1)
                if len(parts) != 2:
                    continue
                text, label = parts[0].strip(), parts[1].strip().lower()
                emotion = EMOTION_MAP.get(label)
                if emotion:
                    all_samples.append({"text": preprocess(text), "emotion": emotion})
                    count += 1
        print(f"  {split}.txt → {count} samples")

    df = pd.DataFrame(all_samples)
    counts = Counter(df["emotion"])
    print(f"\n  Total samples : {len(df)}")
    print(f"  Emotion distribution:")
    for e, c in sorted(counts.items()):
        print(f"    {e:<12} {c:>5}  {'█' * (c // 300)}")
    print("═"*60)
    return df

# ═══════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        # X is a dense numpy array from TF-IDF
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class EmotionNet(nn.Module):
    """
    3-layer fully connected neural network for emotion classification.
    
    Architecture:
      Input (TF-IDF features: 10,000)
        → Dense Layer 1 (512 neurons) + ReLU + Dropout
        → Dense Layer 2 (256 neurons) + ReLU + Dropout
        → Dense Layer 3 (128 neurons) + ReLU + Dropout
        → Output Layer (6 emotions) + Softmax
    """
    def __init__(self, input_dim, num_classes):
        super(EmotionNet, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)

# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += X_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model on a dataloader. Returns loss, accuracy, predictions."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += X_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. Load data ──
    df = load_data()

    # ── 2. TF-IDF Vectorization ──
    print("\n  Vectorizing text with TF-IDF …")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATS,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["text"]).toarray()   # dense array for PyTorch
    print(f"  Feature matrix shape : {X.shape}")

    # ── 3. Encode labels ──
    le = LabelEncoder()
    y  = le.fit_transform(df["emotion"])
    print(f"  Classes              : {list(le.classes_)}")

    # ── 4. Train / Val / Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # ── 5. DataLoaders ──
    train_ds  = EmotionDataset(X_train, y_train)
    val_ds    = EmotionDataset(X_val,   y_val)
    test_ds   = EmotionDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── 6. Model, Loss, Optimizer ──
    model     = EmotionNet(input_dim=MAX_FEATS, num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n  Model Architecture:")
    print(f"  {model}")
    print(f"\n  Device    : {DEVICE}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Optimizer : Adam (lr={LR})")

    # ── 7. Training Loop with Epochs ──
    print("\n" + "═"*60)
    print("  TRAINING — EPOCH BY EPOCH")
    print("═"*60)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<12} {'LR'}")
    print("  " + "─"*70)

    history = []
    best_val_acc  = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch       = epoch
            flag = " ← best"
        else:
            flag = ""

        print(f"  {epoch:<8} {train_loss:<14.4f} {train_acc*100:<14.2f} "
              f"{val_loss:<14.4f} {val_acc*100:<12.2f} {current_lr:.5f}{flag}")

    # ── 8. Final Evaluation on Test Set ──
    print("\n" + "═"*60)
    print("  FINAL TEST EVALUATION (Best Model)")
    print("═"*60)

    model.load_state_dict(best_model_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Epoch  : {best_epoch}")
    print(f"  Test Loss   : {test_loss:.4f}")
    print(f"  Test Acc    : {test_acc*100:.2f}%")
    print(f"  F1 Score    : {f1_score(labels, preds, average='weighted'):.4f}")

    print("\n  Classification Report:")
    print(classification_report(labels, preds, target_names=le.classes_))

    # ── 9. Save Model ──
    torch.save({
        "model_state":   best_model_state,
        "vectorizer":    vectorizer,
        "label_encoder": le,
        "input_dim":     MAX_FEATS,
        "num_classes":   len(le.classes_),
        "best_epoch":    best_epoch,
        "test_accuracy": round(test_acc, 4),
        "history":       history,
    }, os.path.join(MODEL_DIR, "pytorch_emotion_model.pt"))

    print(f"\n  ✅ PyTorch model saved → models/pytorch_emotion_model.pt")

    # ── 10. Epoch History Summary ──
    print("\n" + "═"*60)
    print("  EPOCH HISTORY SUMMARY")
    print("═"*60)
    print(f"  {'Epoch':<8} {'Train Acc %':<16} {'Val Acc %':<16} {'Train Loss'}")
    print("  " + "─"*50)
    for h in history:
        marker = " ★" if h["epoch"] == best_epoch else ""
        print(f"  {h['epoch']:<8} {h['train_acc']*100:<16.2f} "
              f"{h['val_acc']*100:<16.2f} {h['train_loss']}{marker}")

    print("\n" + "═"*60)
    print("  PYTORCH TRAINING COMPLETE")
    print("═"*60)
    print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%  (Epoch {best_epoch})")
    print(f"  Final Test Accuracy      : {test_acc*100:.2f}%")
    print(f"  Model saved to           : models/pytorch_emotion_model.pt")
    print("═"*60)

    # ── 11. Quick Inference Test ──
    print("\n  QUICK INFERENCE TEST (Unseen Sentences)")
    print("  " + "─"*50)
    test_sentences = [
        ("I am so happy with this purchase!", "Happy"),
        ("This is a complete waste of money!", "Angry"),
        ("The app keeps crashing on my phone.", "Frustrated"),
        ("I wish things were like before.", "Sad"),
        ("Oh wow this is absolutely incredible!!", "Excited"),
        ("The product arrived on time.", "Neutral"),
    ]

    model.eval()
    correct = 0
    for text, expected in test_sentences:
        clean   = preprocess(text)
        vec     = vectorizer.transform([clean]).toarray()
        tensor  = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out  = model(tensor)
            pred = le.classes_[out.argmax(dim=1).item()]
        match = "✅" if pred == expected else "❌"
        correct += (pred == expected)
        print(f"  {match} '{text[:45]}'")
        print(f"       Expected: {expected:<12} Predicted: {pred}")

    print(f"\n  Unseen accuracy: {correct}/{len(test_sentences)} = "
          f"{correct/len(test_sentences)*100:.1f}%")
    print("\n  Run `python app.py` to launch the GUI.\n")


if __name__ == "__main__":
    main()