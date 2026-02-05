
# VibraMind: Advanced Industrial Motor Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg)
![Plotly](https://img.shields.io/badge/Plotly-6.5-636efa.svg)

**VibraMind** is a state-of-the-art predictive maintenance system designed for industrial motor bearings. It utilizes a hybrid Deep Learning architecture combining **Multi-Scale CNNs (MSCAN)**, **Temporal Convolutional Networks (TCN)**, and **Transformers** to predict Remaining Useful Life (RUL) with high precision and uncertainty quantification.

## 🚀 Key Features

- **Hybrid Deep Architecture**: Combines local feature extraction (CNN), temporal dynamics (TCN), and long-range dependencies (Transformer).
- **Uncertainty-Aware RUL**: Uses **Quantile Regression** to provide not just a single RUL value, but a confidence interval (10th, 50th, and 90th percentiles).
- **Multi-Task Learning**: Jointly predicts:
    - **RUL**: Remaining Useful Life percentage.
    - **Health Indicator (HI)**: An ensemble metric (RMS, Kurtosis, Crest Factor, Spectral Energy).
    - **Operating Conditions**: Real-time Motor Speed and Voltage Load.
- **Interactive Visualizations**: 
    - **Live Signal Chart**: Real-time Plotly charts in the dashboard for Vib H, Vib V, and Temperature.
    - **Full Interactive Report**: Comprehensive training analysis via a standalone HTML report.
- **Automated EDA**: Generates signal comparisons, frequency domain analysis (PSD), and correlation heatmaps during training.
- **Memory-Efficient Pipeline**: On-the-fly data augmentation and lightweight processing to handle large vibration datasets.
- **Modern Dashboard**: A premium, high-contrast dark-styled dashboard for real-time monitoring and file analysis.

## 🛠️ Tech Stack

- **Core**: Python 3.10
- **Deep Learning**: PyTorch (CUDA supported)
- **Data Science**: NumPy, Pandas, Scikit-Learn (RobustScaler), SciPy (Signal)
- **Web Interface**: Flask, Jinja2, Bootstrap 5
- **Visualization**: Plotly.js, Matplotlib, Seaborn

## 📁 Project Structure

```text
├── dashboard/              # Flask Application & Web Assets
│   ├── static/             # CSS (Solar Theme), JS (Inference & Plotly logic)
│   ├── templates/          # HTML5 UI
│   └── app.py              # Backend API & Model Inference
├── models/                 # Saved Model Weights & Preprocessing Scalers
│   ├── best_mscan_model.pth
│   └── input_scaler.pkl
├── scripts/                # Utility Scripts
│   └── generate_samples.py # Test data generator (Healthy, Degraded, Critical)
├── samples/                # Sample CSVs for Dashboard Testing
├── output/                 # Training plots, EDA, and Interactive Reports
│   ├── eda_01_sample_signals.png
│   ├── interactive_results.html
│   └── multi_task_summary.txt
├── main.py                 # Core Training & EDA Pipeline
└── requirements.txt        # Dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SudhirMahaaraja/VibraMind.git
   cd VibraMind
   ```

2. **Create and Activate Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install flask torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn joblib plotly scipy
   ```

## 🚄 Usage

### 1. Generating Test Data
To create sample CSV files for testing the dashboard:
```bash
python scripts/generate_samples.py
```
This creates `test_healthy.csv`, `test_degraded.csv`, and `test_critical.csv` in the `samples/` folder.

### 2. Training the Model
To perform EDA and train the hybrid model on the FEMTO dataset:
```bash
python main.py
```
*Note: This generates the interactive HTML report in `output/` and fits the `RobustScaler`.*

### 3. Launching the Dashboard
To start the web interface:
```bash
python dashboard/app.py
```
Then visit `http://127.0.0.1:5000` in your browser.

## 🧠 Model Architecture

The core model, `MSCAN_Hybrid`, features:
- **MSCAN Encoder**: Four parallel branches with 1x1, 3x3, 5x5, and 7x7 kernels for multi-scale vibration feature extraction.
- **TCN Blocks**: Dilated convolutions (dilation 1, 2, 4) to capture temporal patterns.
- **Lightweight Transformer**: Positional encoding and attention mechanism for long-range dependency modeling.
- **Quantile Regression**: Outputs 10th, 50th, and 90th percentiles for robust uncertainty quantification.

## 📊 Results & EDA
The system automatically generates an Exploratory Data Analysis (EDA) suite including:
- **Time Domain**: Comparisons between healthy and degraded bearings.
- **Frequency Domain**: PSD analysis showing energy shifts at failure.
- **Visual Report**: Interactive Plotly-based training history and prediction metrics.
