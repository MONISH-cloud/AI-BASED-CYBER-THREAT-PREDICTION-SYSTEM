# AI-BASED-CYBER-THREAT-PREDICTION-SYSTEMT

---

# AI-Driven Cyber Threat Prediction System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.6%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 📌 Project Overview
The **AI-Driven Cyber Threat Prediction System** is an intelligent, multi-engine framework designed for real-time detection and proactive forecasting of network security threats. By integrating temporal sequence modeling with unsupervised anomaly detection, the system provides a robust defense against both known patterns (e.g., DDoS) and previously unseen "zero-day" vulnerabilities.

### Key Features
* **Hybrid Detection Engine:** Combines **Bi-LSTM** for temporal patterns and **Deep Autoencoders** for anomaly detection.
* **Predictive Forecasting:** Anticipates attack evolution across short-term (5-packet) and extended (30-packet) horizons.
* **Ensemble Risk Fusion:** A weighted mechanism ($0.90$ Bi-LSTM + $0.10$ Autoencoder) to produce a unified risk score.
* **Real-time Dashboard:** Integrated Streamlit-based interface for live monitoring and severity-based alerting.

---

## 🏗️ System Architecture & File Flow

The framework follows a structured pipeline from raw traffic capture to predictive alerting.


### 1. Preprocessing & Sequence Formation
* **Flow Transformation:** Raw network traffic is converted into structured flow-based feature representations.
* **Temporal Windows:** Data is grouped into fixed-length sequential windows to capture gradual attack buildup.

### 2. Dual-Engine Modeling
* **Temporal Engine (Bi-LSTM):** Captures dependencies by analyzing both past and future context within traffic sequences.
* **Anomaly Engine (Autoencoder):** Learned on benign traffic; it identifies threats by measuring reconstruction loss (deviations from "normal").

### 3. Risk Fusion & Prediction
* **Weighted Ensemble:** Merges the classification probability and anomaly score into a single continuous risk metric.
* **Temporal Forecasting:** Analyzes the rate of change in risk scores to predict future threat trajectories.

---

## 📊 Performance
The system was benchmarked using the **CICIDS-2017 dataset**, achieving state-of-the-art results:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 98.6% |
| **Precision** | 0.98 |
| **AUC** | 0.981 |
| **False Positive Rate (FPR)** | 0.019 |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow / Keras
* Streamlit
* Pandas & Scikit-learn

### Installation
```bash
git clone https://github.com/MONISH-cloud/AI-BASED-CYBER-THREAT-PREDICTION-SYSTEM.git
cd AI-BASED-CYBER-THREAT-PREDICTION-SYSTEM
pip install -r requirements.txt
```

### Usage
To launch the real-time monitoring dashboard:
```bash
streamlit run livepro.py
```

---

## 👥 Contributors
* **Monish R** - *RV University*
* **Pradeep M Doddakaragi** - *RV University*
* **Omkar Suresh Naik** - *RV University*
* **Nihal Saukar K** - *RV University*

---

