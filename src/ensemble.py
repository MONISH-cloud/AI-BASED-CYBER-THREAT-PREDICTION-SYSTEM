"""
ensemble.py — Phase 1 (fixed)
Always reloads LSTM from checkpoint file to ensure correct weights.
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch

from lstm_model        import DDoSLSTM, LSTMTrainer
from autoencoder_model import TrafficAutoencoder, AutoencoderTrainer

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = '/Users/a91959/Downloads/deeplearning project'
LSTM_CKPT  = os.path.join(BASE_DIR, 'files', 'lstm_best.pt')
AE_CKPT    = os.path.join(BASE_DIR, 'files', 'autoencoder_best.pt')

# ─────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────
W_LSTM        = 0.90
W_AUTOENCODER = 0.10

SEVERITY_TIERS = {
    'LOW':      (0.00, 0.30),
    'MEDIUM':   (0.30, 0.60),
    'HIGH':     (0.60, 0.85),
    'CRITICAL': (0.85, 1.01),
}


def get_severity(risk: float) -> str:
    for label, (lo, hi) in SEVERITY_TIERS.items():
        if lo <= risk < hi:
            return label
    return 'CRITICAL'


def _align_features(input_df: pd.DataFrame, expected_count: int,
                    scaler=None) -> np.ndarray:
    df       = input_df.copy()
    drop_cols = ['Label', 'Label_encoded', 'label', 'label_name']
    X_raw    = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_raw    = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_aligned = pd.DataFrame(
        0.0, index=np.arange(len(df)),
        columns=[f"f{i}" for i in range(expected_count)]
    )
    for i in range(min(X_raw.shape[1], expected_count)):
        X_aligned.iloc[:, i] = X_raw.iloc[:, i]

    X_final = X_aligned.astype(np.float64).values
    if scaler is not None:
        try:
            X_final = scaler.transform(X_aligned)
        except Exception:
            pass
    return X_final.astype(np.float32)


def _load_lstm_from_checkpoint(n_features: int) -> LSTMTrainer:
    """Always reload from .pt file — never trust pickled model state."""
    model = DDoSLSTM(input_size=n_features, num_classes=2,
                     hidden_size=128, num_layers=2)
    model.load_state_dict(
        torch.load(LSTM_CKPT, map_location='cpu', weights_only=True))
    model.eval()
    trainer        = LSTMTrainer.__new__(LSTMTrainer)
    trainer.model  = model
    trainer.device = 'cpu'
    return trainer


def _load_ae_from_checkpoint(n_features: int,
                              threshold: float) -> AutoencoderTrainer:
    """Always reload autoencoder from .pt file."""
    model = TrafficAutoencoder(input_size=n_features, bottleneck=16)
    model.load_state_dict(
        torch.load(AE_CKPT, map_location='cpu', weights_only=True))
    model.eval()
    trainer           = AutoencoderTrainer.__new__(AutoencoderTrainer)
    trainer.model     = model
    trainer.device    = 'cpu'
    trainer.threshold = threshold
    return trainer


# Cache so we only load once per app session
_lstm_cache = None
_ae_cache   = None


def _get_lstm(pkg: dict) -> LSTMTrainer:
    global _lstm_cache
    if _lstm_cache is None:
        _lstm_cache = _load_lstm_from_checkpoint(pkg['n_features'])
    return _lstm_cache


def _get_ae(pkg: dict) -> AutoencoderTrainer:
    global _ae_cache
    if _ae_cache is None:
        ae_trainer = pkg.get('ae_trainer')
        threshold  = getattr(ae_trainer, 'threshold', 0.09) \
                     if ae_trainer else 0.09
        _ae_cache  = _load_ae_from_checkpoint(pkg['n_features'], threshold)
    return _ae_cache


def get_predictions(input_df: pd.DataFrame, package: dict) -> tuple:
    """
    Drop-in replacement for detect_logic.get_predictions().
    Returns: (labels, risk_scores)
    """
    n_feat  = package['n_features']
    scaler  = package['scaler']
    encoder = package['label_encoder']

    X = _align_features(input_df, n_feat, scaler)

    # ── LSTM ──────────────────────────────────────────────────
    lstm_probs = np.zeros(len(X))
    try:
        lstm         = _get_lstm(package)
        window       = package.get('window_size', 10)
        lstm_probs   = lstm.predict_proba(X, window_size=window)
    except Exception as e:
        print(f"[Ensemble] LSTM error: {e}")

    # ── Autoencoder ───────────────────────────────────────────
    ae_scores = np.zeros(len(X))
    try:
        ae           = _get_ae(package)
        _, ae_scores = ae.predict_anomaly(X)
    except Exception as e:
        print(f"[Ensemble] AE error: {e}")

    # ── Weighted risk ─────────────────────────────────────────
    risk_scores = np.clip(
        W_LSTM * lstm_probs + W_AUTOENCODER * ae_scores, 0, 1
    )

    preds_numeric = (risk_scores >= 0.5).astype(int)
    try:
        labels = encoder.inverse_transform(preds_numeric)
    except Exception:
        labels = np.where(preds_numeric == 1, 'DDoS', 'BENIGN')

    return labels, risk_scores