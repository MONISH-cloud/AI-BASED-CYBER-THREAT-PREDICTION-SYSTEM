"""
train_phase1_pro.py — The "Logic-First" Ensemble Trainer
Fixes: PyArrow Indexing, Flow-Based Splits, and Dynamic AE Scaling.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

# Internal Model Imports (Ensure these files are in your folder)
from lstm_model        import DDoSLSTM, LSTMTrainer, PacketSequenceDataset
from autoencoder_model import TrafficAutoencoder, AutoencoderTrainer

# ─────────────────────────────────────────────
# PATHS & HYPERPARAMETERS
# ─────────────────────────────────────────────
BASE_DIR   = '/Users/a91959/Downloads/deeplearning project'
DATA_PATH  = os.path.join(BASE_DIR, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX copy.csv')
XGB_PKL    = os.path.join(BASE_DIR, 'ddos_detector_clean.pkl')
OUTPUT_PKL = os.path.join(BASE_DIR, 'files', 'ddos_ensemble_phase1_pro.pkl')

WINDOW_SIZE   = 10
LSTM_EPOCHS   = 20
AE_EPOCHS     = 25
BATCH_SIZE    = 512
MAX_TRAIN_SEQ = 150_000

def save_pkg(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def load_pkg(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    print("=" * 60)
    print("  PHASE 1 — UPGRADED ENSEMBLE (PRO LOGIC EDITION)")
    print("=" * 60)

    # 1. Load and Clean
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. Robust Flow-Based Grouping
    print("[2/6] Implementing Flow-Based Sequence Integrity...")
    
    # Header Detective: Look for IP columns
    src_col = next((c for c in df.columns if 'Source' in c and 'IP' in c), None)
    dst_col = next((c for c in df.columns if 'Destination' in c and 'IP' in c), None)

    if src_col and dst_col:
        print(f"  ✓ Found IP Columns: '{src_col}' and '{dst_col}'")
        df['flow_id'] = df[src_col].astype(str) + "_" + df[dst_col].astype(str)
    else:
        print("  ! IP columns not found. Using Row-Block grouping.")
        print(f"  Available columns: {list(df.columns[:10])}...")
        df['flow_id'] = (np.arange(len(df)) // 100).astype(str)

    # Sort to ensure packets stay in sequence order
    df = df.sort_values(['flow_id'])
    
    # 3. Feature Alignment
    print("\n[3/6] Aligning features with XGBoost requirements...")
    package = load_pkg(XGB_PKL)
    label_encoder = package['label_encoder']
    features_used = package.get('features_used', None)
    
    if not features_used:
        # Exclude metadata and non-numeric labels
        exclude = ['Label', 'Label_encoded', 'flow_id', 'Timestamp', src_col, dst_col]
        features_used = [c for c in df.columns if c not in exclude]
    
    # Ensure all listed features exist in the CSV
    features_used = [c for c in features_used if c in df.columns]

    df['Label_encoded'] = label_encoder.transform(df['Label'])
    X_raw = df[features_used].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Label_encoded'].values

    # 4. Flow-Based Split (FIXED for PyArrow Indexing)
    print(f"\n[4/6] Splitting {len(df['flow_id'].unique()):,} Unique Flows...")
    
    # Convert to standard Python list to avoid Arrow indexing TypeErrors
    unique_flows = df['flow_id'].unique().tolist()
    train_f, test_f = train_test_split(unique_flows, test_size=0.3, random_state=42)
    
    # Using sets for O(1) lookup speed
    train_f_set, test_f_set = set(train_f), set(test_f)
    train_mask = df['flow_id'].isin(train_f_set)
    test_mask  = df['flow_id'].isin(test_f_set)
    
    scaler = StandardScaler()
    # Passing .values to ensure NumPy arrays reach the scaler
    X_train = scaler.fit_transform(X_raw[train_mask].values).astype(np.float32)
    X_test  = scaler.transform(X_raw[test_mask].values).astype(np.float32)
    y_train = y[train_mask.values]
    y_test  = y[test_mask.values]

    # 5. Train LSTM
    print(f"\n[5/6] Training LSTM (Input Dim: {X_raw.shape[1]})...")
    lstm_model = DDoSLSTM(input_size=X_raw.shape[1], num_classes=len(label_encoder.classes_))
    
    train_ds = PacketSequenceDataset(X_train, y_train, WINDOW_SIZE)
    test_ds  = PacketSequenceDataset(X_test,  y_test,  WINDOW_SIZE)

    if len(train_ds) > MAX_TRAIN_SEQ:
        idx = np.random.choice(len(train_ds), MAX_TRAIN_SEQ, replace=False)
        train_ds = Subset(train_ds, idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    lstm_trainer = LSTMTrainer(lstm_model, checkpoint_path=os.path.join(BASE_DIR, 'files/lstm_pro.pt'))
    lstm_trainer.fit(train_loader, val_loader, epochs=LSTM_EPOCHS)

    # 6. Train Autoencoder & Calculate Dynamic Threshold
    print(f"\n[6/6] Training Autoencoder (Unsupervised Baseline)...")
    benign_idx = list(label_encoder.classes_).index('BENIGN')
    X_benign_train = X_train[y_train == benign_idx]

    ae_model = TrafficAutoencoder(input_size=X_raw.shape[1])
    ae_trainer = AutoencoderTrainer(ae_model, checkpoint_path=os.path.join(BASE_DIR, 'files/ae_pro.pt'))
    ae_trainer.fit(X_benign_train, epochs=AE_EPOCHS, batch_size=BATCH_SIZE)

    # Thresholding logic: Find the error limit for 95% of normal traffic
    print("  Generating Anomaly Threshold (Percentile 95)...")
    _, benign_errors = ae_trainer.predict_anomaly(X_benign_train)
    ae_threshold = np.percentile(benign_errors, 95)
    print(f"  ✓ Dynamic Threshold: {ae_threshold:.6f}")

    # 7. Evaluate Ensemble
    print("\n--- FINAL PERFORMANCE REPORT ---")
    lstm_probs = lstm_trainer.predict_proba(X_test, window_size=WINDOW_SIZE)
    _, ae_errors = ae_trainer.predict_anomaly(X_test)
    
    # Scale AE risk: 0 (Normal) to 1 (Beyond Threshold)
    ae_risk = np.clip(ae_errors / ae_threshold, 0, 1)

    # Ensemble: LSTM provides high-accuracy classification, AE catches outliers
    ensemble_risk = (0.90 * lstm_probs) + (0.10 * ae_risk)
    ensemble_preds = (ensemble_risk >= 0.5).astype(int)

    print(classification_report(y_test, ensemble_preds, target_names=label_encoder.classes_))

    # Save
    final_package = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features_used': features_used,
        'lstm_trainer': lstm_trainer,
        'ae_trainer': ae_trainer,
        'ae_threshold': ae_threshold,
        'window_size': WINDOW_SIZE,
        'n_features': X_raw.shape[1]
    }
    save_pkg(final_package, OUTPUT_PKL)
    print(f"\nSuccess! Pro-Level Ensemble saved to: {OUTPUT_PKL}")

if __name__ == "__main__":
    main()