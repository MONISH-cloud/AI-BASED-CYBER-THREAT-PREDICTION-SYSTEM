"""
autoencoder_model.py — Phase 1
Trained on BENIGN traffic only.
High reconstruction error = anomalous / never-seen-before attack pattern.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class TrafficAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder. Bottleneck = 16 dims.
    Trained on benign only → attack packets have high reconstruction error.
    """

    def __init__(self, input_size: int, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),  nn.ReLU(),
            nn.Linear(32, 64),          nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=1)


class AutoencoderTrainer:

    def __init__(self, model: TrafficAutoencoder, device: str = None,
                 lr: float = 1e-3, checkpoint_path: str = 'autoencoder_best.pt'):
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model      = model.to(self.device)
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5)
        self.criterion  = nn.MSELoss()
        self.checkpoint = checkpoint_path
        self.threshold  = None
        print(f"[Autoencoder] Device: {self.device}")

    def fit(self, X_benign: np.ndarray, epochs: int = 30,
            batch_size: int = 512, val_split: float = 0.1):
        n_val  = int(len(X_benign) * val_split)
        X_tr   = X_benign[n_val:]
        X_vl   = X_benign[:n_val]

        def make_loader(arr, shuffle):
            t = torch.tensor(arr, dtype=torch.float32)
            return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

        tr_loader = make_loader(X_tr, shuffle=True)
        vl_loader = make_loader(X_vl, shuffle=False)
        best_val  = float('inf')

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for (xb,) in tr_loader:
                xb   = xb.to(self.device)
                loss = self.criterion(self.model(xb), xb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(X_tr)

            self.model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for (xb,) in vl_loader:
                    xb = xb.to(self.device)
                    vl_loss += self.criterion(self.model(xb), xb).item() * len(xb)
            vl_loss /= len(X_vl)

            self.scheduler.step(vl_loss)
            print(f"  Epoch {epoch:02d}/{epochs} | train {tr_loss:.6f} | val {vl_loss:.6f}")

            if vl_loss < best_val:
                best_val = vl_loss
                torch.save(self.model.state_dict(), self.checkpoint)
                print(f"  ✓ Checkpoint saved")

        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location=self.device,
                       weights_only=True))
        self.threshold = float(np.percentile(self.reconstruction_errors(X_tr), 95))
        print(f"\n[Autoencoder] Done. Anomaly threshold (p95): {self.threshold:.6f}")
        return self

    def reconstruction_errors(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch_size)
        errors = []
        with torch.no_grad():
            for (xb,) in loader:
                errors.append(
                    self.model.reconstruction_error(xb.to(self.device)).cpu().numpy())
        return np.concatenate(errors)

    def predict_anomaly(self, X: np.ndarray,
                        custom_threshold: float = None) -> tuple:
        assert self.threshold is not None, "Call fit() first"
        threshold = custom_threshold or self.threshold
        errors    = self.reconstruction_errors(X)
        scores    = np.clip(errors / (threshold * 2), 0, 1)
        flags     = errors > threshold
        return flags, scores
