# 🚀 AI-Driven Cyber Threat Prediction System

## 📌 Overview

This project implements an **AI-Driven Cyber Threat Prediction System** for real-time detection and forecasting of network security threats, with a primary focus on **Distributed Denial-of-Service (DDoS) attacks**.

The system integrates a **hybrid multi-engine architecture** combining:

* Bidirectional LSTM (Bi-LSTM) for temporal traffic analysis
* Autoencoder for anomaly detection
* Weighted ensemble model for unified risk scoring

Unlike traditional intrusion detection systems, this framework provides **predictive capabilities**, enabling early detection of cyber threats before they escalate.

---

## 🎯 Key Contributions

* 🔹 Hybrid detection framework combining supervised and unsupervised learning
* 🔹 Predictive threat forecasting (short-term & long-term horizons)
* 🔹 Real-time risk scoring and alert generation
* 🔹 Detection of both known and zero-day attacks
* 🔹 Streamlit-based interactive monitoring system

---

## 🧠 System Architecture

The system consists of the following modules:

1. **Data Preprocessing**

   * Cleaning, normalization, and feature extraction
   * Sequence generation for temporal modeling

2. **Temporal Modeling (Bi-LSTM)**

   * Captures sequential patterns in network traffic
   * Learns evolving attack behavior

3. **Anomaly Detection (Autoencoder)**

   * Learns normal traffic patterns
   * Detects deviations (zero-day attacks)

4. **Ensemble Risk Fusion**

   * Combines outputs:

     * 90% Bi-LSTM
     * 10% Autoencoder

5. **Risk Prediction Module**

   * Forecasts future attack trends
   * Enables proactive defense

6. **Decision Engine**

   * Generates severity-based alerts:

     * LOW (0–30%)
     * MEDIUM (30–60%)
     * HIGH (60–85%)
     * CRITICAL (85–100%)

---

## 📊 Results & Performance

* ✅ **Accuracy:** 98.6%
* ✅ **AUC Score:** 0.981
* ✅ Low false positive rate
* ✅ Strong generalization to unseen traffic

### 📈 Model Comparison

| Model                     | Accuracy  |
| ------------------------- | --------- |
| Random Forest             | 96.8%     |
| XGBoost                   | 97.5%     |
| Bi-LSTM                   | 97.9%     |
| **Proposed Hybrid Model** | **98.6%** |

📌 As shown in the confusion matrix and ROC curve (paper Figures), the model achieves high true positives and low misclassification. 

---

## 📂 Project Structure

```id="projstruct"
AI-BASED-CYBER-THREAT-PREDICTION-SYSTEM/
│
├── README.md
├── requirements.txt
├── report.pdf
│
├── data/              # Dataset (CICIDS-based)
│
├── src/               # Source code
│   ├── Predictor.py
│   ├── detect_logic.py
│   ├── ensemble.py
│   ├── lstm_model.py
│   ├── autoencoder_model.py
│   ├── train2.py
│   └── app2.py
│
├── models/            # Trained models
│   ├── lstm_best3.pt
│   ├── autoencoder_best3.pt
│   ├── ddos_ensemble_phase1.pkl
│
├── results/           # Output results
│   └── ddos_model_comparison.png
```

---

## ⚙️ Installation

```bash
git clone https://github.com/MONISH-cloud/AI-BASED-CYBER-THREAT-PREDICTION-SYSTEM.git
cd AI-BASED-CYBER-THREAT-PREDICTION-SYSTEM
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the Model

```bash
python src/train2.py
```

### Run Prediction

```bash
python src/Predictor.py
```

### Launch Dashboard

```bash
python src/app2.py
```

---

## 🧪 Dataset

* Based on **CICIDS-2017 dataset**
* Includes benign and DDoS traffic samples
* Preprocessed into sequential data for LSTM

---

## 🔍 Risk Scoring Formula

The system computes a unified risk score:

```id="risk"
R = 0.90 × P_LSTM + 0.10 × E_AE
```

Where:

* **P_LSTM** = classification probability
* **E_AE** = autoencoder reconstruction error

This enables **continuous risk assessment** instead of binary classification.

---

## 💡 Key Features

✔ Hybrid AI architecture
✔ Real-time intrusion detection
✔ Predictive threat forecasting
✔ Low false positives
✔ Scalable and deployable system

---

## 🔮 Future Work

* Extend to multiple attack types (APT, ransomware)
* Integrate attention mechanisms into Bi-LSTM
* Use federated learning for distributed security
* Evaluate on datasets like UNSW-NB15, CIC-IDS-2018

---

## 👨‍💻 Authors

* Monish R
* Omkar Suresh Naik
* Pradeep M Doddakaragi
* Nihal Saukar K

(RV University, Bengaluru)

---


