---

# Offline Phishing Detection Model for Websites Using Recurrent and Convolutional Neural Networks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frangelbarrera/phishing-detection-rnn-cnn/blob/thesis-original/URL_Phishing_Detection.ipynb)


##  Description  
This project implements an **offline** phishing website detection model, capable of operating without an internet connection by leveraging the knowledge acquired during its training phase.  

The architecture combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, leveraging the strengths of both paradigms. CNN layers act as feature extractors, scanning URL components to detect local spatial patterns such as character groupings, token distributions, and structural anomalies. In parallel, the LSTM layer captures sequential dependencies across the URL string, retaining contextual information about recurring patterns that may indicate phishing behavior. This hybrid design allows the model to simultaneously identify fine-grained lexical cues and long-range dependencies, resulting in a more robust and accurate classification of web addresses.  

The system analyzes multiple lexical, structural, and heuristic attributes of web addresses, classifying them as **legitimate** or **potentially malicious (phishing)** with high accuracy.

---

##  Key Features  
- **Hybrid CNN + LSTM architecture** for superior detection performance  
- **Offline execution**: no internet connection required for URL analysis  
- **Advanced feature extraction**: length, subdomains, special characters, suspicious patterns, and more  
- **Interactive analysis interface** for user-provided URLs  
- **Warning messages and recommendations** to enhance user security  
- **Feature standardization** with a persisted `StandardScaler` so inference reproduces training preprocessing
- **Early stopping and learning-rate scheduling** to avoid overfitting

---

##  Model Results  
| Metric | Value |
| --- | --- |
| Test accuracy | **88.67 %** |
| AUC-ROC | **0.946** |
| Average precision | 0.949 |
| F1 (at tuned threshold) | 0.883 |
| Brier score | 0.087 |
| Decision threshold | 0.504 |

Confusion matrix on the held-out 20 % test split (rows = true, cols = predicted):

```
              legit   phishing
legit          1046       97
phishing        162      981
```

All metrics are computed on predicted probabilities (not argmax) and are
fully reproducible from `train.py` with seed `42`. See `training_metadata.json`
for the full training configuration and `metrics.json` for the ROC / PR curves.

---

##  Repository Structure  
```
/phishing-detection-rnn-cnn
│
├── dataset_phishing.csv         # Dataset used for training (11,430 rows)
├── URL_Phishing_Detection.ipynb # Main notebook with the complete workflow
├── train.py                     # Reproducible training script
├── my_model.keras               # Trained CNN+LSTM model
├── scaler.pkl                   # Fitted StandardScaler
├── feature_columns.txt          # Canonical 55-feature order
├── metrics.json                 # Full evaluation report (ROC, PR, etc.)
├── history.json                 # Training history
├── training_metadata.json       # Versions, seed, architecture, metrics digest
├── SHA256SUMS                   # Integrity checksums for all artifacts
├── X_train.npy / X_test.npy     # Preprocessed datasets (legacy, superseded by train.py)
├── Y_train.npy / Y_test.npy
└── README.md
```

---

##  Branches

This repository contains two branches:

- **`thesis-original`** (this branch) — The original CNN+LSTM hybrid architecture
  from the academic thesis defended in July 2025, with reproducibility fixes
  applied post-hoc (corrected input shape, softmax output head, feature
  standardization, early stopping, AUC computed on probabilities). The
  architecture is preserved exactly as defended: Conv1D(64) → MaxPool →
  LSTM(100) → Flatten → Dense(64) → Dense(512) → Dense(64) → Dense(2).

- **`main`** — A post-thesis engineering evolution (July 2026) that replaces
  the CNN+LSTM with a simpler two-hidden-layer MLP trained on the same 55
  features. Includes a Flask web UI, CLI, and reproducible artifacts.

To switch between branches:
```bash
git checkout main              # post-thesis MLP evolution
git checkout thesis-original   # defended CNN+LSTM thesis (this branch)
```

---

##  Running the Project with Google Drive and Colab  
To run this project seamlessly in **Google Colab** while keeping all files organized and persistent:  

1. In your Google Drive, **create a new folder** named:  
   ```
   phishing-detection-rnn-cnn
   ```
2. Upload **all project files** into this folder (`.ipynb`, `.csv`, `.keras`, `scaler.pkl`, etc.).  
3. Open `URL_Phishing_Detection.ipynb` in Colab — the notebook auto-detects
   Colab and mounts Drive. Run all cells top to bottom.
4. After training, the new `my_model.keras`, `scaler.pkl`, `metrics.json`,
   `history.json`, `training_metadata.json`, and `SHA256SUMS` will be
   written to the project folder.

To retrain locally instead:
```bash
python train.py
```

---

##  Requirements  
- Python 3.10+  
- TensorFlow ≥ 2.15
- Pandas, NumPy, Matplotlib, Seaborn  
- scikit-learn ≥ 1.3

Quick installation:  
```bash
pip install -r requirements.txt
```

---

##  Usage  
1. Clone the repository and check out the thesis branch:  
```bash
git clone https://github.com/frangelbarrera/phishing-detection-rnn-cnn.git
cd phishing-detection-rnn-cnn
git checkout thesis-original
```
2. Load the trained model and scaler:  
```python
import pickle
import tensorflow as tf
model = tf.keras.models.load_model("my_model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))
```
3. Run the interactive analysis from the notebook, or retrain from scratch:
```bash
python train.py
```

---

##  Notes and Warnings  
- Features are purely lexical/structural. The model does **not** inspect
  SSL certificates, DNS records, or page content, so it cannot detect
  phishing pages hosted on otherwise-legitimate domains.
- Five features from the original dataset (`random_domain`,
  `domain_in_brand`, `brand_in_subdomain`, `brand_in_path`,
  `nb_external_redirection`) require curated brand lists or live network
  access and are hardcoded to 0 in `extract_features` for offline use.
- For greater accuracy, it is recommended to supplement its use with other
  security tools.

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
