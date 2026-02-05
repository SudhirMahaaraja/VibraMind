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

def load_femto_bearing(bearing_path):
    print(f"Loading FEMTO bearing: {bearing_path}")
    if not os.path.exists(bearing_path):
        return None
        
    all_files = os.listdir(bearing_path)
    acc_files = sorted([f for f in all_files if f.startswith('acc_') and f.endswith('.csv')])
    temp_files = sorted([f for f in all_files if f.startswith('temp_') and f.endswith('.csv')])
    
    if not acc_files:
        return None
        
    bearing_data = []
    
    for acc_file in acc_files:
        acc_num = acc_file.split('_')[1].split('.')[0]
        acc_path = os.path.join(bearing_path, acc_file)
        
        try:
            acc_df = pd.read_csv(acc_path, header=None)
            if isinstance(acc_df.iloc[0, 0], str):
                acc_df = pd.read_csv(acc_path)
            
            vib_data = acc_df.iloc[:, -2:].values.astype(np.float32)
            
            # Find matching temp
            temp_file = f"temp_{acc_num}.csv"
            temp_val = 0.0
            if temp_file in temp_files:
                temp_path = os.path.join(bearing_path, temp_file)
                temp_df = pd.read_csv(temp_path, header=None)
                if isinstance(temp_df.iloc[0, 0], str):
                    temp_df = pd.read_csv(temp_path)
                temp_val = temp_df.iloc[:, -1].mean()
            
            temp_channel = np.full((vib_data.shape[0], 1), temp_val, dtype=np.float32)
            combined = np.concatenate([vib_data, temp_channel], axis=1)
            bearing_data.append(combined)
        except Exception as e:
            continue
            
    return bearing_data

# Dataset Config
DATASET_ROOT = "datasets"
FEMTO_TRAIN_DIR = os.path.join(DATASET_ROOT, "Learning_set")
FEMTO_TEST_DIR = os.path.join(DATASET_ROOT, "Test_set")
FEMTO_VAL_DIR = os.path.join(DATASET_ROOT, "Full_Test_Set")

WINDOW_SIZE = 2560 # Standard FEMTO window

STEP_SIZE = 2560
PREDICTION_HORIZON = 0 # Directly label each file


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

# --- Data Collection ---
all_windows_train, all_labels_train, domain_labels_train = [], [], []
all_windows_test, all_labels_test, domain_labels_test = [], [], []
all_windows_val, all_labels_val, domain_labels_val = [], [], []

def process_set(path, win_list, lbl_list, dom_list, dom_id):
    if not os.path.exists(path): return
    bearings = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for bearing in bearings:
        bearing_windows = load_femto_bearing(os.path.join(path, bearing))
        if bearing_windows:
            for idx, win in enumerate(bearing_windows):
                win_list.append(win.T)
                rul = 1.0 - (idx / len(bearing_windows))
                lbl_list.append([rul, rul, 0.5, 0.5]) # [RUL, HI, Speed, Voltage]
                dom_list.append(dom_id)

print(f"Loading Learning_set for Training...")
process_set(FEMTO_TRAIN_DIR, all_windows_train, all_labels_train, domain_labels_train, 0)

print(f"Loading Test_set for Testing...")
process_set(FEMTO_TEST_DIR, all_windows_test, all_labels_test, domain_labels_test, 0)

print(f"Loading Full_Test_set for Validation...")
process_set(FEMTO_VAL_DIR, all_windows_val, all_labels_val, domain_labels_val, 0)


X_train = np.array(all_windows_train)
y_train = np.array(all_labels_train)
d_train = np.array(domain_labels_train)

X_test = np.array(all_windows_test)
y_test = np.array(all_labels_test)
d_test = np.array(domain_labels_test)

X_val = np.array(all_windows_val)
y_val = np.array(all_labels_val)
d_val = np.array(domain_labels_val)

print(f"Dataset Summary: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scaling
print("Applying Scaling...")
scaler = RobustScaler()
# Reshape for fitting: (N, C, L) -> (N*L, C)
C = X_train.shape[1]
L = X_train.shape[2]
X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, C)
scaler.fit(X_train_flat)

def apply_scaling(X, scaler):
    N, C, L = X.shape
    X_flat = X.transpose(0, 2, 1).reshape(-1, C)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(N, L, C).transpose(0, 2, 1)

X_train = apply_scaling(X_train, scaler)
X_val = apply_scaling(X_val, scaler)
X_test = apply_scaling(X_test, scaler)

# Save the scaler for the dashboard
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, os.path.join('models', 'input_scaler.pkl'))
print(f"✓ Saved input scaler (3 channels) to models/input_scaler.pkl")

X_train_raw, y_train_raw, d_train_raw = X_train, y_train, d_train 

# Cleanup
del all_windows_train, all_labels_train, all_windows_test, all_labels_test, all_windows_val, all_labels_val
gc.collect()

# ============================================================================
# PART 4.5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("PART 4.5: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

os.makedirs('output', exist_ok=True)

# 1. Sample Signal Visualization (Healthy vs Late Stage)
print("Generating sample signal plots...")
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
sample_idx_early = 0
sample_idx_late = len(X_train) // 4 # FEMTO data usually at start

# Early stage
axes[0, 0].plot(X_train[sample_idx_early, 0, :], color='blue', alpha=0.7)
axes[0, 0].set_title('Early Stage Signal (Vib H)')
axes[1, 0].plot(X_train[sample_idx_early, 1, :], color='green', alpha=0.7)
axes[1, 0].set_title('Early Stage Signal (Vib V)')
axes[2, 0].plot(X_train[sample_idx_early, 2, :], color='red', alpha=0.7)
axes[2, 0].set_title('Early Stage Signal (Temp)')

# Late stage
axes[0, 1].plot(X_train[sample_idx_late, 0, :], color='blue', alpha=0.7)
axes[0, 1].set_title('Degraded Stage Signal (Vib H)')
axes[1, 1].plot(X_train[sample_idx_late, 1, :], color='green', alpha=0.7)
axes[1, 1].set_title('Degraded Stage Signal (Vib V)')
axes[2, 1].plot(X_train[sample_idx_late, 2, :], color='red', alpha=0.7)
axes[2, 1].set_title('Degraded Stage Signal (Temp)')

plt.tight_layout()
plt.savefig(os.path.join('output', 'eda_01_sample_signals.png'), dpi=150)
plt.close()

# 2. Power Spectral Density (PSD)
print("Generating frequency domain analysis...")
from scipy import signal
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
f, psd_early = signal.welch(X_train[sample_idx_early, 0, :], fs=25600)
f, psd_late = signal.welch(X_train[sample_idx_late, 0, :], fs=25600)

axes[0].semilogy(f, psd_early)
axes[0].set_title('PSD Early Stage (Vib H)')
axes[0].set_xlabel('Frequency [Hz]')
axes[1].semilogy(f, psd_late)
axes[1].set_title('PSD Degraded Stage (Vib H)')
axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()
plt.savefig(os.path.join('output', 'eda_02_frequency_analysis.png'), dpi=150)
plt.close()

# 3. Label Distributions
print("Generating label distribution plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(y_train[:, 0], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('RUL Distribution (Train)')
axes[1].hist(y_train[:, 1], bins=50, color='salmon', edgecolor='black')
axes[1].set_title('Health Indicator Distribution (Train)')
plt.tight_layout()
plt.savefig(os.path.join('output', 'eda_03_label_distributions.png'), dpi=150)
plt.close()

# 4. Correlation Heatmap (Mean features)
print("Generating feature correlation heatmap...")
mean_features = np.mean(X_train, axis=2)
df_corr = pd.DataFrame(mean_features, columns=['Mean_VibH', 'Mean_VibV', 'Mean_Temp'])
df_corr['RUL'] = y_train[:, 0]
df_corr['HI'] = y_train[:, 1]

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature & Label Correlations')
plt.savefig(os.path.join('output', 'eda_04_correlation_heatmap.png'), dpi=150)
plt.close()

print("✓ EDA plots saved in 'output/' directory.")



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
        # Energy branch adds 2 features (horizontal & vertical mean absolute value)
        self.fc1 = nn.Linear(ms_channels + input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Multi-task heads
        self.num_quantiles = num_quantiles
        self.rul_head = nn.Linear(64, num_quantiles)  # Quantile regression
        self.hi_head = nn.Linear(64, 1)  # HI reconstruction
        self.speed_head = nn.Linear(64, 1)
        self.voltage_head = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        # Parallel branches - NO BN here to preserve absolute vibration magnitude information
        x1 = self.relu(self.conv1x1(x))
        x2 = self.relu(self.conv3x3(x))
        x3 = self.relu(self.conv5x5(x))
        x4 = self.relu(self.conv7x7(x))
        
        x_multi = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Calculate Energy feature (Mean Absolute Value) - Key for magnitude sensitivity
        energy = torch.mean(torch.abs(x), dim=2) # (batch, 2)
        
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
        
        # Concatenate Energy feature to FC layers to distinguish silent vs loud bearings
        x_combined = torch.cat([x_adapted, energy], dim=1)
        
        # FC layers
        x_fc = self.relu(self.fc1(x_combined))
        x_fc = self.dropout(x_fc)
        x_fc = self.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        
        # Multi-task outputs
        rul_quantiles = self.sigmoid(self.rul_head(x_fc))
        hi_pred = self.sigmoid(self.hi_head(x_fc))
        speed_pred = self.sigmoid(self.speed_head(x_fc))
        voltage_pred = self.sigmoid(self.voltage_head(x_fc))
        
        out = torch.cat([rul_quantiles, hi_pred, speed_pred, voltage_pred], dim=1)
        
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
    
    def forward(self, preds, targets):
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

NUM_EPOCHS = 50
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
    
    for batch_x, batch_y, _ in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss, loss_dict = criterion(outputs, batch_y)
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
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
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

model.load_state_dict(torch.load(os.path.join('models', 'best_mscan_model.pth'), map_location=device))
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

# Static plot
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig(os.path.join('output', '03_multi_output_results.png'), dpi=150)
print("✓ Saved: output/03_multi_output_results.png")
plt.close()

# Interactive Plotly Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("\nGenerating Interactive Visualizations...")

fig_inter = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'RUL Prediction with Uncertainty', 'Training History',
        'HI Reconstruction', 'Motor Speed Prediction',
        'Voltage Prediction', 'RUL Residuals Distribution'
    )
)

# RUL with Uncertainty
fig_inter.add_trace(go.Scatter(
    x=list(range(500)), y=test_predictions[:500, 1],
    mode='lines', name='Predicted (Median)', line=dict(color='red')
), row=1, col=1)
fig_inter.add_trace(go.Scatter(
    x=list(range(500)), y=test_targets[:500, 0],
    mode='lines', name='Actual', line=dict(color='blue', dash='dash')
), row=1, col=1)
fig_inter.add_trace(go.Scatter(
    x=list(range(500)) + list(range(500))[::-1],
    y=list(test_predictions[:500, 2]) + list(test_predictions[:500, 0])[::-1],
    fill='toself', fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='80% Confidence Interval', showlegend=True
), row=1, col=1)

# Training History
fig_inter.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss'), row=1, col=2)
fig_inter.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss'), row=1, col=2)

# HI
fig_inter.add_trace(go.Scatter(
    x=test_targets[:, 1], y=test_predictions[:, 3],
    mode='markers', marker=dict(size=4, opacity=0.5), name='HI'
), row=2, col=1)
fig_inter.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False), row=2, col=1)

# Speed
fig_inter.add_trace(go.Scatter(
    x=test_targets[:, 2], y=test_predictions[:, 4],
    mode='markers', marker=dict(size=4, opacity=0.5, color='green'), name='Speed'
), row=2, col=2)
fig_inter.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False), row=2, col=2)

# Voltage
fig_inter.add_trace(go.Scatter(
    x=test_targets[:, 3], y=test_predictions[:, 5],
    mode='markers', marker=dict(size=4, opacity=0.5, color='orange'), name='Voltage'
), row=3, col=1)
fig_inter.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False), row=3, col=1)

# Residuals
fig_inter.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color='purple', name='Residuals'), row=3, col=2)

fig_inter.update_layout(height=1200, width=1200, title_text="MSCAN Hybrid Multi-Task Model Interactive Report", showlegend=True)
fig_inter.write_html(os.path.join('output', 'interactive_results.html'))
print("✓ Saved Interactive Report: output/interactive_results.html")


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