<<<<<<< HEAD
# VibraMind: Advanced Industrial Motor Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**VibraMind** is a state-of-the-art predictive maintenance system designed for industrial motor bearings. It utilizes a hybrid Deep Learning architecture combining **Multi-Scale CNNs (MSCAN)**, **Temporal Convolutional Networks (TCN)**, and **Transformers** to predict Remaining Useful Life (RUL) with high precision and uncertainty quantification.

## 🚀 Key Features

- **Hybrid Deep Architecture**: Combines local feature extraction (CNN), temporal dynamics (TCN), and long-range dependencies (Transformer).
- **Uncertainty-Aware RUL**: Uses **Quantile Regression** to provide not just a single RUL value, but a confidence interval (10th, 50th, and 90th percentiles).
- **Multi-Task Learning**: Jointly predicts:
    - **RUL**: Remaining Useful Life percentage.
    - **Health Indicator (HI)**: An ensemble metric (RMS, Kurtosis, Crest Factor, Spectral Energy).
    - **Operating Conditions**: Real-time Motor Speed and Voltage Load.
- **Domain Adaptation (DANN)**: Implements a Gradient Reversal Layer to ensure the model generalizes across different operating conditions (Condition 1, 2, and 3).
- **Memory-Efficient Pipeline**: On-the-fly data augmentation and lightweight processing to handle large vibration datasets without crashing system memory.
- **Solar UI Dashboard**: A premium, high-contrast light-mode dashboard (Cream, Orange, Brown palette) for real-time monitoring and manual file analysis.

## 🛠️ Tech Stack

- **Core**: Python 3.10
- **Deep Learning**: PyTorch (CUDA supported)
- **Data Science**: NumPy, Pandas, Scikit-Learn (RobustScaler, QuantileTransformer)
- **Web Interface**: Flask, Jinja2, Bootstrap 5
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```text
├── dashboard/              # Flask Application & Web Assets
│   ├── static/             # CSS (Solar Theme), JS (Inference Logic)
│   ├── templates/          # HTML5 UI
│   └── app.py              # Backend API & Model Inference
├── models/                 # Saved Model Weights & Preprocessing Scalers
│   ├── best_mscan_model.pth
│   ├── input_scaler.pkl
│   └── quantile_transformer.pkl
├── scripts/                # Utility Scripts
│   ├── eda_analysis.py     # exploratory data analysis
│   └── processing.py       # Data cleaning and merging
├── samples/                # Sample CSVs for Dashboard Testing
├── output/                 # Training plots and evaluation logs
├── main.py                 # Core Training & Validation Pipeline
└── requirements.txt        # Minimal dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/VibraMind-AI.git
   cd VibraMind-AI
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/Scripts/activate  # Windows: .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚄 Usage

### 1. Training the Model
To train the hybrid model from scratch using the FEMTO/PRONOSTIA vibration datasets:
```bash
python main.py
```
*Note: This will automatically generate scalers in `models/` and performance plots in `output/`.*

### 2. Launching the Dashboard
To start the web interface for real-time analysis:
```bash
python dashboard/app.py
```
Then visit `http://127.0.0.1:5000` in your browser.

## 🧠 Model Architecture

The core model, `MSCAN_Hybrid`, features:
- **MSCAN Encoder**: Three parallel branches with 3x3, 5x5, and 7x7 kernels for multi-scale vibration feature extraction.
- **TCN Blocks**: Dilated convolutions to capture temporal patterns across the 5120-sample windows.
- **Transformer Encoder**: Lightweight attention mechanism to weight the most critical degradation patterns.
- **Quantile Heads**: Outputs 3 values for RUL to calculate predictive intervals.

## 📊 Results
The model achieves high accuracy in Health Indicator reconstruction (R² > 0.90) and provides robust RUL estimations even on unseen operating conditions through its Domain Adaptation layer.

---
Developed by **PREC MACHINE AI** | Powered by DeepMind Advanced Agentic Coding
=======
# VibraMind
VibraMind is a state-of-the-art predictive maintenance system designed for industrial motor bearings. It utilizes a hybrid Deep Learning architecture combining Multi-Scale CNNs (MSCAN), Temporal Convolutional Networks (TCN), and Transformers to predict Remaining Useful Life (RUL) with high precision and uncertainty quantification.
>>>>>>> e6978587f45cca3bff4a2889162394e6f86a8f2b
