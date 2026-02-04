import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, kurtosis
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
import gc
import time
import math

# ============================================================================
# PART 1: DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("="*80)
print("PART 1: DATA LOADING & EXPLORATORY DATA ANALYSIS")
print("="*80)

def load_and_parse_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()

def get_sorted_csv_files(bearing_path):
    if not os.path.exists(bearing_path):
        return []
    files = [f for f in os.listdir(bearing_path) if f.endswith('.csv')]
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()
    return files

def process_motor_bearing(bearing_path, feature_cols):
    print(f"Processing bearing: {bearing_path}")
    files = get_sorted_csv_files(bearing_path)
    files = files[::5]
    
    if not files:
        print("  No CSV files found.")
        return None, None
        
    bearing_data = []
    req_cols = ['Horizontal_vibration_signals', 'Vertical_vibration_signals', 'motor_speed', 'u_q', 'u_d']
    
    for csv_file in files:
        file_path = os.path.join(bearing_path, csv_file)
        try:
            df = load_and_parse_csv(file_path)
            if df.empty: continue
            if all(col in df.columns for col in req_cols):
                data_chunk = df[req_cols].values
                bearing_data.append(data_chunk)
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
            continue
            
    if not bearing_data:
        return None, None
        
    full_data = np.concatenate(bearing_data, axis=0)
    return full_data, None

# Dataset Config
DATASET_ROOT = "datasets"
CONDITIONS = ["Condition 1", "Condition 2", "Condition 3"]
WINDOW_SIZE = 5120
STEP_SIZE = 2560
PREDICTION_HORIZON = 50

# ============================================================================
# PART 2: ENHANCED FEATURE EXTRACTION & HEALTH INDICATOR
# ============================================================================

def compute_rms(signal, window_size=1024):
    rms_values = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values)

def compute_kurtosis(signal, window_size=1024):
    kurt_values = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        kurt_values.append(kurtosis(window))
    return np.array(kurt_values)

def compute_crest_factor(signal, window_size=1024):
    crest_values = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        peak = np.max(np.abs(window))
        rms = np.sqrt(np.mean(window**2)) + 1e-8
        crest_values.append(peak / rms)
    return np.array(crest_values)

def compute_spectral_energy(signal, window_size=1024, fs=25600):
    spec_energy = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        freqs, psd = welch(window, fs=fs, nperseg=min(256, len(window)))
        spec_energy.append(np.sum(psd))
    return np.array(spec_energy)

def compute_ensemble_hi(signal_h, signal_v, window_size=1024):
    """Ensemble Health Indicator: RMS + Kurtosis + Crest Factor + Spectral Energy"""
    rms_h = compute_rms(signal_h, window_size)
    rms_v = compute_rms(signal_v, window_size)
    kurt_h = compute_kurtosis(signal_h, window_size)
    kurt_v = compute_kurtosis(signal_v, window_size)
    crest_h = compute_crest_factor(signal_h, window_size)
    crest_v = compute_crest_factor(signal_v, window_size)
    spec_h = compute_spectral_energy(signal_h, window_size)
    spec_v = compute_spectral_energy(signal_v, window_size)
    
    min_len = min(len(rms_h), len(kurt_h), len(crest_h), len(spec_h))
    features = np.stack([
        rms_h[:min_len], rms_v[:min_len],
        np.clip(kurt_h[:min_len], -10, 50), np.clip(kurt_v[:min_len], -10, 50),
        np.clip(crest_h[:min_len], 0, 20), np.clip(crest_v[:min_len], 0, 20),
        spec_h[:min_len], spec_v[:min_len]
    ], axis=1)
    
    # Normalize each feature
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    
    # PCA to single HI
    pca = PCA(n_components=1)
    hi = pca.fit_transform(features_norm).flatten()
    hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-8)
    return hi

# ============================================================================
# PART 3: ADVANCED DATA AUGMENTATION
# ============================================================================

def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1, sigma, (x.shape[0], 1, 1))
    return x * factor

def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[2])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2))
    warp_steps = np.linspace(0, x.shape[2]-1, num=knot+2)
    
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        time_warp_factor = np.interp(orig_steps, warp_steps, random_warps[i])
        warped_time = np.cumsum(time_warp_factor)
        warped_time = (warped_time - warped_time.min()) / (warped_time.max() - warped_time.min()) * (x.shape[2]-1)
        for j in range(x.shape[1]):
            ret[i, j] = np.interp(orig_steps, warped_time, x[i, j])
    return ret

def window_slice(x, reduce_ratio=0.9):
    target_len = int(x.shape[2] * reduce_ratio)
    if target_len < 100:
        return x
    starts = np.random.randint(0, x.shape[2] - target_len, x.shape[0])
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        sliced = x[i, :, starts[i]:starts[i]+target_len]
        # Resize back to original length
        for j in range(x.shape[1]):
            ret[i, j] = np.interp(np.linspace(0, 1, x.shape[2]), np.linspace(0, 1, target_len), sliced[j])
    return ret

def spike_injection(x, prob=0.05, magnitude=3.0):
    ret = x.copy()
    mask = np.random.random(x.shape) < prob
    spikes = np.random.choice([-1, 1], x.shape) * magnitude * np.random.random(x.shape)
    ret = ret + mask * spikes
    return ret

def synthetic_degradation(x, intensity=0.1):
    """Add synthetic degradation patterns: drift, harmonics, amplitude increase"""
    ret = x.copy()
    for i in range(x.shape[0]):
        t = np.linspace(0, 1, x.shape[2])
        drift = intensity * t * np.random.uniform(0.5, 1.5)
        harmonic = intensity * 0.1 * np.sin(2 * np.pi * np.random.randint(2, 10) * t)
        for j in range(x.shape[1]):
            ret[i, j] = ret[i, j] * (1 + intensity * np.random.uniform(0, 0.5)) + drift + harmonic
    return ret

def augment_data_light(X, y):
    """Apply LIGHTWEIGHT augmentation - single pass with random transforms per sample"""
    # Only apply jitter + scaling (minimal memory overhead)
    X_aug = X.copy()
    
    # Random jitter
    X_aug = X_aug + np.random.normal(0, 0.02, X_aug.shape).astype(np.float32)
    
    # Random per-sample scaling
    scale = np.random.uniform(0.9, 1.1, (X_aug.shape[0], 1, 1)).astype(np.float32)
    X_aug = X_aug * scale
    
    return X_aug, y

class AugmentedDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset with ON-THE-FLY augmentation"""
    def __init__(self, X, y, d, augment=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.d = torch.LongTensor(d)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        d = self.d[idx]
        
        if self.augment and np.random.random() > 0.5:
            # On-the-fly augmentation (no memory overhead)
            x = x + torch.randn_like(x) * 0.02  # Jitter
            x = x * torch.FloatTensor([np.random.uniform(0.9, 1.1)]).view(1, 1)  # Scale
        
        return x, y, d

def rolling_zscore(signal, window=512):
    """Per-sensor rolling z-score normalization"""
    result = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window)
        window_data = signal[start:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data) + 1e-8
        result[i] = (signal[i] - mean) / std
    return result

def per_run_normalize(data):
    """Normalize per-run to reduce stationary shifts"""
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + 1e-8
    return (data - mean) / std

def balance_dataset(X, y, threshold=0.25, factor=3):
    rul = y[:, 0]
    low_rul_idx = np.where(rul < threshold)[0]
    if len(low_rul_idx) == 0: return X, y
    X_low = X[low_rul_idx]
    y_low = y[low_rul_idx]
    X_bal = np.concatenate([X] + [X_low] * (factor - 1), axis=0)
    y_bal = np.concatenate([y] + [y_low] * (factor - 1), axis=0)
    return X_bal, y_bal

# ============================================================================
# PART 4: DATA PROCESSING
# ============================================================================

all_windows_train = []
all_labels_train = []
all_windows_test = []
all_labels_test = []
domain_labels_train = []
domain_labels_test = []

print(f"\nStage 1: Fitting Scalers...")
input_scaler = RobustScaler()
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
speed_scaler = MinMaxScaler()
voltage_scaler = MinMaxScaler()

train_inputs = []
train_speeds = []
train_volts = []

for condition in ["Condition 1", "Condition 2"]:
    cond_path = os.path.join(DATASET_ROOT, condition)
    if not os.path.exists(cond_path): continue
    bearings = [d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))]
    for bearing in bearings:
        raw_data, _ = process_motor_bearing(os.path.join(cond_path, bearing), [])
        if raw_data is not None:
            train_inputs.append(raw_data[:, :2])
            s = raw_data[:, 2]
            u_q = raw_data[:, 3]
            u_d = raw_data[:, 4]
            v = np.sqrt(u_q**2 + u_d**2)
            train_speeds.append(s.reshape(-1, 1))
            train_volts.append(v.reshape(-1, 1))

if not train_inputs:
    raise ValueError("No training data found to fit scaler!")

print("Fitting Input Scaler (RobustScaler + Quantile)...")
all_train_inputs = np.concatenate(train_inputs, axis=0)
input_scaler.fit(all_train_inputs)
quantile_transformer.fit(input_scaler.transform(all_train_inputs))

print("Fitting Target Scalers (Speed, Voltage)...")
speed_scaler.fit(np.concatenate(train_speeds, axis=0))
voltage_scaler.fit(np.concatenate(train_volts, axis=0))
print("✓ Scalers fitted.")

# Save scalers for dashboard inference
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(input_scaler, os.path.join('models', 'input_scaler.pkl'))
joblib.dump(quantile_transformer, os.path.join('models', 'quantile_transformer.pkl'))
print("✓ Scalers saved (models/input_scaler.pkl, models/quantile_transformer.pkl)")

print(f"\nStage 2: Processing datasets and generating windows...")

domain_id_map = {"Condition 1": 0, "Condition 2": 1, "Condition 3": 2}

for condition in CONDITIONS:
    cond_path = os.path.join(DATASET_ROOT, condition)
    if not os.path.exists(cond_path): continue
    bearings = [d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))]
    domain_id = domain_id_map[condition]
    
    for bearing in bearings:
        bearing_path = os.path.join(cond_path, bearing)
        raw_data, _ = process_motor_bearing(bearing_path, [])
        if raw_data is None: continue
        
        # Per-run normalization
        raw_vib = per_run_normalize(raw_data[:, :2])
        inputs_robust = input_scaler.transform(raw_vib)
        inputs_scaled = quantile_transformer.transform(inputs_robust)
        
        # Ensemble HI
        hi = compute_ensemble_hi(inputs_scaled[:, 0], inputs_scaled[:, 1])
        if len(hi) == 0: continue
        
        raw_speed = raw_data[:, 2].reshape(-1, 1)
        scaled_speed = speed_scaler.transform(raw_speed).flatten()
        raw_vq = raw_data[:, 3]
        raw_vd = raw_data[:, 4]
        raw_voltage = np.sqrt(raw_vq**2 + raw_vd**2).reshape(-1, 1)
        scaled_voltage = voltage_scaler.transform(raw_voltage).flatten()
        
        num_samples = len(inputs_scaled)
        rms_window = 1024
        
        for i in range(0, num_samples - WINDOW_SIZE - PREDICTION_HORIZON, STEP_SIZE):
            window = inputs_scaled[i:i+WINDOW_SIZE]
            future_idx = min((i + WINDOW_SIZE + PREDICTION_HORIZON) // rms_window, len(hi) - 1)
            rul_val = 1.0 - hi[future_idx]
            hi_val = hi[future_idx]
            speed_val = np.mean(scaled_speed[i:i+WINDOW_SIZE])
            volt_val = np.mean(scaled_voltage[i:i+WINDOW_SIZE])
            
            # Labels: [RUL, HI, Speed, Voltage]
            label = [rul_val, hi_val, speed_val, volt_val]
            
            if condition in ["Condition 1", "Condition 2"]:
                all_windows_train.append(window.T)
                all_labels_train.append(label)
                domain_labels_train.append(domain_id)
            else:
                all_windows_test.append(window.T)
                all_labels_test.append(label)
                domain_labels_test.append(domain_id)

    gc.collect()

X_train_val = np.array(all_windows_train)
y_train_val = np.array(all_labels_train)
d_train_val = np.array(domain_labels_train)

X_test = np.array(all_windows_test)
y_test = np.array(all_labels_test)
d_test = np.array(domain_labels_test)

X_train_raw, X_val, y_train_raw, y_val, d_train_raw, d_val = train_test_split(
    X_train_val, y_train_val, d_train_val, test_size=0.15, random_state=42
)

print(f"Balancing training dataset...")
X_train_bal, y_train_bal = balance_dataset(X_train_raw, y_train_raw)
d_train_bal = np.concatenate([d_train_raw] * 3)[:len(X_train_bal)]

print(f"Applying lightweight augmentation (memory-efficient)...")
# Use single-pass lightweight augmentation instead of 6x heavy augmentation
X_train, y_train = augment_data_light(X_train_bal, y_train_bal)
d_train = d_train_bal  # No tile needed - same size

# Clear intermediate arrays to free memory
del X_train_bal, y_train_bal, d_train_bal, all_windows_train, all_labels_train
gc.collect()

print(f"\nFinal Dataset Summary:")
print(f"  Original Train Windows: {len(X_train_raw)}")
print(f"  After Balancing + Augmentation: {X_train.shape}")
print(f"  Validation: {X_val.shape}")
print(f"  Test Pool (Cond 3): {X_test.shape}")

# ============================================================================
# PART 5: HYBRID MSCAN + TRANSFORMER MODEL WITH DOMAIN ADAPTATION
# ============================================================================

print("\n" + "="*80)
print("PART 5: HYBRID MSCAN + TRANSFORMER MODEL")
print("="*80)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc(avg_out)
        out = self.sigmoid(out).view(x.size(0), x.size(1), 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return x * self.sigmoid(out)

class TemporalConvBlock(nn.Module):
    """TCN Block for temporal modeling"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(out + residual)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LightweightTransformer(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer(x)

class FeatureAdapter(nn.Module):
    """Lightweight adapter for few-shot fine-tuning"""
    def __init__(self, dim, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))

class MSCAN_Hybrid(nn.Module):
    def __init__(self, input_channels=2, num_filters=16, dropout_rate=0.3, num_quantiles=3):
        super().__init__()
        
        # Multi-scale CNN
        self.conv1x1 = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.conv3x3 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(input_channels, num_filters, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn4 = nn.BatchNorm1d(num_filters)
        
        ms_channels = num_filters * 4
        
        # Attention
        self.channel_attention = ChannelAttention(ms_channels, reduction=4)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        
        # TCN for temporal dependencies
        self.tcn1 = TemporalConvBlock(ms_channels, ms_channels, dilation=1)
        self.tcn2 = TemporalConvBlock(ms_channels, ms_channels, dilation=2)
        self.tcn3 = TemporalConvBlock(ms_channels, ms_channels, dilation=4)
        
        # Downsample for transformer
        self.pool_for_transformer = nn.AdaptiveAvgPool1d(64)
        
        # Lightweight Transformer
        self.transformer = LightweightTransformer(d_model=ms_channels, nhead=4, num_layers=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature adapters
        self.adapter1 = FeatureAdapter(ms_channels, bottleneck=8)
        
        # FC layers
        self.fc1 = nn.Linear(ms_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Multi-task heads
        self.num_quantiles = num_quantiles
        self.rul_head = nn.Linear(64, num_quantiles)  # Quantile regression
        self.hi_head = nn.Linear(64, 1)  # HI reconstruction
        self.speed_head = nn.Linear(64, 1)
        self.voltage_head = nn.Linear(64, 1)
        
        # Domain discriminator (DANN)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(ms_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)  # 3 domains
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, return_domain=False):
        # Multi-scale feature extraction
        x1 = self.relu(self.bn1(self.conv1x1(x)))
        x2 = self.relu(self.bn2(self.conv3x3(x)))
        x3 = self.relu(self.bn3(self.conv5x5(x)))
        x4 = self.relu(self.bn4(self.conv7x7(x)))
        
        x_multi = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Attention
        x_ca = self.channel_attention(x_multi)
        x_sa = self.spatial_attention(x_ca)
        
        # TCN
        x_tcn = self.tcn1(x_sa)
        x_tcn = self.tcn2(x_tcn)
        x_tcn = self.tcn3(x_tcn)
        
        # Transformer branch
        x_pooled = self.pool_for_transformer(x_tcn)
        x_trans = x_pooled.permute(0, 2, 1)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.mean(dim=1)
        
        # Apply adapter
        x_adapted = self.adapter1(x_trans)
        
        # FC layers
        x_fc = self.relu(self.fc1(x_adapted))
        x_fc = self.dropout(x_fc)
        x_fc = self.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        
        # Multi-task outputs
        rul_quantiles = self.sigmoid(self.rul_head(x_fc))
        hi_pred = self.sigmoid(self.hi_head(x_fc))
        speed_pred = self.sigmoid(self.speed_head(x_fc))
        voltage_pred = self.sigmoid(self.voltage_head(x_fc))
        
        out = torch.cat([rul_quantiles, hi_pred, speed_pred, voltage_pred], dim=1)
        
        if return_domain:
            x_global = self.global_pool(x_tcn).view(x_tcn.size(0), -1)
            domain_out = self.domain_classifier(self.grl(x_global))
            return out, domain_out
        
        return out

# ============================================================================
# PART 6: LOSS FUNCTIONS
# ============================================================================

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)

class MultiTaskLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # Learnable task weights
        self.log_vars = nn.Parameter(torch.zeros(4))
    
    def forward(self, preds, targets, domain_preds=None, domain_targets=None):
        num_q = 3
        rul_pred = preds[:, :num_q]
        hi_pred = preds[:, num_q:num_q+1]
        speed_pred = preds[:, num_q+1:num_q+2]
        voltage_pred = preds[:, num_q+2:num_q+3]
        
        rul_target = targets[:, 0:1]
        hi_target = targets[:, 1:2]
        speed_target = targets[:, 2:3]
        voltage_target = targets[:, 3:4]
        
        # Losses with uncertainty weighting
        loss_rul = self.quantile_loss(rul_pred, rul_target)
        loss_hi = self.mse(hi_pred, hi_target)
        loss_speed = self.mse(speed_pred, speed_target)
        loss_voltage = self.mse(voltage_pred, voltage_target)
        
        precision0 = torch.exp(-self.log_vars[0])
        precision1 = torch.exp(-self.log_vars[1])
        precision2 = torch.exp(-self.log_vars[2])
        precision3 = torch.exp(-self.log_vars[3])
        
        total_loss = (precision0 * loss_rul + self.log_vars[0] +
                     precision1 * loss_hi + self.log_vars[1] +
                     precision2 * loss_speed + self.log_vars[2] +
                     precision3 * loss_voltage + self.log_vars[3])
        
        # Domain adaptation loss
        if domain_preds is not None and domain_targets is not None:
            domain_loss = self.ce(domain_preds, domain_targets)
            total_loss = total_loss + 0.1 * domain_loss
        
        return total_loss, {
            'rul': loss_rul.item(),
            'hi': loss_hi.item(),
            'speed': loss_speed.item(),
            'voltage': loss_voltage.item()
        }

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if device.type == 'cuda':
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

input_channels = X_train.shape[1]
print(f"Detected input channels: {input_channels}")

model = MSCAN_Hybrid(input_channels=input_channels, num_filters=16, dropout_rate=0.3).to(device)
print(f"\nModel created successfully and moved to {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# PART 7: TRAINING CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("PART 7: TRAINING CONFIGURATION")
print("="*80)

batch_size = 32
pin_memory = device.type == 'cuda'

# Use memory-efficient AugmentedDataset with on-the-fly augmentation
print("Creating memory-efficient datasets with on-the-fly augmentation...")
train_dataset = AugmentedDataset(X_train, y_train, d_train, augment=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0)

# Delete numpy arrays to free memory
del X_train, y_train, d_train
gc.collect()

val_dataset = AugmentedDataset(X_val, y_val, d_val, augment=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

test_dataset = AugmentedDataset(X_test, y_test, d_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

print(f"\nTraining Configuration:")
print(f"  Batch Size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

criterion = MultiTaskLoss(quantiles=[0.1, 0.5, 0.9])
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print(f"\nOptimizer: AdamW (lr=0.001, weight_decay=1e-4)")
print(f"Scheduler: CosineAnnealingWarmRestarts")

# ============================================================================
# PART 8: MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("PART 8: MODEL TRAINING")
print("="*80)

NUM_EPOCHS = 30
best_val_loss = float('inf')
patience = 15
patience_counter = 0

history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': [], 'rul_loss': [], 'hi_loss': []}

print(f"\nTraining for {NUM_EPOCHS} epochs...\n")

total_start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y, batch_d in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_d = batch_d.to(device)
        
        optimizer.zero_grad()
        outputs, domain_out = model(batch_x, return_domain=True)
        loss, loss_dict = criterion(outputs, batch_y, domain_out, batch_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    history['train_loss'].append(train_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_d in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_d = batch_d.to(device)
            outputs = model(batch_x)
            loss, _ = criterion(outputs, batch_y)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(outputs[:, 1] - batch_y[:, 0])).item()
    
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), os.path.join('models', 'best_mscan_model.pth'))
    else:
        patience_counter += 1
    
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
          f"Time: {epoch_duration:.2f}s | "
          f"Train: {train_loss:.6f} | "
          f"Val: {val_loss:.6f} | "
          f"MAE: {val_mae:.6f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if patience_counter >= patience:
        print(f"\n→ Early stopping at epoch {epoch+1}")
        break

total_training_time = time.time() - total_start_time

model.load_state_dict(torch.load('best_mscan_model.pth', map_location=device))
print(f"\n✓ Best model loaded from checkpoint to {device}")

# ============================================================================
# PART 9: TESTING & EVALUATION
# ============================================================================

print("\n" + "="*80)
print("PART 9: MODEL TESTING & EVALUATION")
print("="*80)

model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch_x, batch_y, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        test_predictions.append(outputs.cpu().numpy())
        test_targets.append(batch_y.numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

val_predictions = []
val_targets = []

with torch.no_grad():
    for batch_x, batch_y, _ in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        val_predictions.append(outputs.cpu().numpy())
        val_targets.append(batch_y.numpy())

val_predictions = np.concatenate(val_predictions, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

def calculate_metrics(y_true, y_pred, name="Metric"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Extract median quantile for RUL (index 1)
target_names = ["RUL (Median)", "HI", "SPEED", "VOLTAGE"]
pred_indices = [1, 3, 4, 5]  # Median RUL, HI, Speed, Voltage

print(f"\n{'VALIDATION METRICS':-^60}")
for i, (name, idx) in enumerate(zip(target_names, pred_indices)):
    mae, rmse, r2 = calculate_metrics(val_targets[:, i], val_predictions[:, idx])
    print(f"{name:15s} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

print(f"\n{'TEST METRICS':-^60}")
for i, (name, idx) in enumerate(zip(target_names, pred_indices)):
    mae, rmse, r2 = calculate_metrics(test_targets[:, i], test_predictions[:, idx])
    print(f"{name:15s} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

# ============================================================================
# PART 10: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PART 10: RESULTS VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# RUL with uncertainty bounds
axes[0, 0].fill_between(range(len(test_targets[:500])),
                        test_predictions[:500, 0],
                        test_predictions[:500, 2],
                        alpha=0.3, label='10-90% Interval')
axes[0, 0].plot(test_targets[:500, 0], 'b-', label='Actual', alpha=0.7)
axes[0, 0].plot(test_predictions[:500, 1], 'r--', label='Predicted (Median)')
axes[0, 0].set_title('RUL Prediction with Uncertainty')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Sample')
axes[0, 0].set_ylabel('RUL')

# Training history
axes[0, 1].plot(history['train_loss'], label='Train Loss')
axes[0, 1].plot(history['val_loss'], label='Val Loss')
axes[0, 1].set_title('Training History')
axes[0, 1].legend()

# HI prediction
axes[1, 0].scatter(test_targets[:, 1], test_predictions[:, 3], alpha=0.3, s=10)
axes[1, 0].plot([0, 1], [0, 1], 'r--')
axes[1, 0].set_title('Health Indicator Reconstruction')
axes[1, 0].set_xlabel('Actual HI')
axes[1, 0].set_ylabel('Predicted HI')

# Speed
axes[1, 1].scatter(test_targets[:, 2], test_predictions[:, 4], alpha=0.3, s=10, color='green')
axes[1, 1].plot([0, 1], [0, 1], 'r--')
axes[1, 1].set_title('Motor Speed Prediction')
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')

# Voltage
axes[2, 0].scatter(test_targets[:, 3], test_predictions[:, 5], alpha=0.3, s=10, color='orange')
axes[2, 0].plot([0, 1], [0, 1], 'r--')
axes[2, 0].set_title('Voltage Prediction')
axes[2, 0].set_xlabel('Actual')
axes[2, 0].set_ylabel('Predicted')

# Residuals
residuals = test_targets[:, 0] - test_predictions[:, 1]
axes[2, 1].hist(residuals, bins=50, color='purple', alpha=0.7)
axes[2, 1].set_title('RUL Residuals Distribution')

plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig(os.path.join('output', '03_multi_output_results.png'), dpi=150)
print("✓ Saved: output/03_multi_output_results.png")
plt.close()

# ============================================================================
# PART 11: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PART 11: FINAL SUMMARY")
print("="*80)

mae_rul, _, r2_rul = calculate_metrics(test_targets[:, 0], test_predictions[:, 1])
mae_hi, _, r2_hi = calculate_metrics(test_targets[:, 1], test_predictions[:, 3])
mae_spd, _, r2_spd = calculate_metrics(test_targets[:, 2], test_predictions[:, 4])
mae_vol, _, r2_vol = calculate_metrics(test_targets[:, 3], test_predictions[:, 5])

summary_text = f"""
MSCAN HYBRID MULTI-TASK MODEL WITH UNCERTAINTY
{'='*70}

1. ARCHITECTURE FEATURES
   - Multi-Scale CNN (1x1, 3x3, 5x5, 7x7 kernels)
   - Temporal Convolutional Network (TCN) with dilations
   - Lightweight Transformer for long-range dependencies
   - Domain Adaptation (DANN with Gradient Reversal)
   - Feature Adapters for few-shot fine-tuning

2. ADVANCED PREPROCESSING
   - Per-run normalization + RobustScaler + QuantileTransformer
   - Ensemble Health Indicator (RMS + Kurtosis + Crest + Spectral)
   - Advanced augmentations: Jitter, Scaling, TimeWarp, Spikes, Synthetic Degradation

3. MULTI-TASK OUTPUTS
   - RUL: Quantile Regression (10th, 50th, 90th percentiles)
   - Health Indicator Reconstruction
   - Motor Speed
   - Voltage Load

4. MODEL PERFORMANCE (TEST SET)
   a) RUL Prediction: MAE={mae_rul:.6f}, R²={r2_rul:.6f}
   b) HI Reconstruction: MAE={mae_hi:.6f}, R²={r2_hi:.6f}
   c) Speed: MAE={mae_spd:.6f}, R²={r2_spd:.6f}
   d) Voltage: MAE={mae_vol:.6f}, R²={r2_vol:.6f}

5. TRAINING DETAILS
   - Epochs: {len(history['train_loss'])}
   - Training Time: {total_training_time:.2f}s
   - Loss: Multi-task with learned uncertainty weighting

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary_text)

with open(os.path.join('output', 'multi_task_summary.txt'), 'w') as f:
    f.write(summary_text)

print("\n✓ Saved: output/multi_task_summary.txt")
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

gc.collect()
torch.cuda.empty_cache()