"""
lstm_model.py — Phase 1
2-layer bidirectional LSTM for DDoS temporal sequence detection.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PacketSequenceDataset(Dataset):
    """
    Converts flat feature matrix into overlapping windows.
    Each sample = WINDOW_SIZE consecutive packets.
    Label = label of the LAST packet in the window.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int = 10):
        self.window_size = window_size
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.window_size]
        label = self.y[idx + self.window_size - 1]
        return x_seq, label


class DDoSLSTM(nn.Module):
    """
    2-layer bidirectional LSTM.
    Bidirectional: catches both the ramp-up AND tail of an attack burst.
    """

    def __init__(self, input_size: int, num_classes: int = 2,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step   = lstm_out[:, -1, :]
        return self.classifier(last_step)


class LSTMTrainer:

    def __init__(self, model: DDoSLSTM, device: str = None,
                 lr: float = 1e-3, checkpoint_path: str = 'lstm_best.pt'):
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model      = model.to(self.device)
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5)
        self.criterion  = nn.CrossEntropyLoss()
        self.checkpoint = checkpoint_path
        print(f"[LSTM] Device: {self.device}")

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits  = self.model(X_batch)
                loss    = self.criterion(logits, y_batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += loss.item() * len(y_batch)
                correct    += (logits.argmax(1) == y_batch).sum().item()
                total      += len(y_batch)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs: int = 20):
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(vl_loss)
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)
            print(f"  Epoch {epoch:02d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(self.model.state_dict(), self.checkpoint)
                print(f"  ✓ Checkpoint saved")

        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location=self.device,
                       weights_only=True))
        print(f"\n[LSTM] Best val_loss: {best_val_loss:.4f}")
        return history

    def predict_proba(self, X: np.ndarray, window_size: int = 10,
                      batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        dummy_y  = np.zeros(len(X), dtype=int)
        dataset  = PacketSequenceDataset(X, dummy_y, window_size)
        loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                logits = self.model(X_batch.to(self.device))
                probs  = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())
        window_probs = np.concatenate(all_probs)
        pad = np.full(window_size - 1, window_probs[0])
        return np.concatenate([pad, window_probs])

    def predict(self, X: np.ndarray, window_size: int = 10,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X, window_size) >= threshold).astype(int)
