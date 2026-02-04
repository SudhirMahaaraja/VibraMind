import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import io
import math
import joblib

app = Flask(__name__)

# ============================================================================
# MODEL DEFINITION (Must match saved model - Hybrid MSCAN + Transformer)
# ============================================================================

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
        
        self.conv1x1 = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.conv3x3 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(input_channels, num_filters, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn4 = nn.BatchNorm1d(num_filters)
        
        ms_channels = num_filters * 4
        
        self.channel_attention = ChannelAttention(ms_channels, reduction=4)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        
        self.tcn1 = TemporalConvBlock(ms_channels, ms_channels, dilation=1)
        self.tcn2 = TemporalConvBlock(ms_channels, ms_channels, dilation=2)
        self.tcn3 = TemporalConvBlock(ms_channels, ms_channels, dilation=4)
        
        self.pool_for_transformer = nn.AdaptiveAvgPool1d(64)
        self.transformer = LightweightTransformer(d_model=ms_channels, nhead=4, num_layers=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.adapter1 = FeatureAdapter(ms_channels, bottleneck=8)
        
        # FC layers
        # Energy branch adds 2 features (horizontal & vertical mean absolute value)
        self.fc1 = nn.Linear(ms_channels + input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.num_quantiles = num_quantiles
        self.rul_head = nn.Linear(64, num_quantiles)
        self.hi_head = nn.Linear(64, 1)
        self.speed_head = nn.Linear(64, 1)
        self.voltage_head = nn.Linear(64, 1)
        
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(ms_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, return_domain=False):
        # Parallel branches - NO BN here to preserve absolute vibration magnitude information
        x1 = self.relu(self.conv1x1(x))
        x2 = self.relu(self.conv3x3(x))
        x3 = self.relu(self.conv5x5(x))
        x4 = self.relu(self.conv7x7(x))
        
        x_multi = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Calculate Energy feature (Mean Absolute Value) - Key for magnitude sensitivity
        energy = torch.mean(torch.abs(x), dim=2) # (batch, 2)
        
        x_ca = self.channel_attention(x_multi)
        x_sa = self.spatial_attention(x_ca)
        
        x_tcn = self.tcn1(x_sa)
        x_tcn = self.tcn2(x_tcn)
        x_tcn = self.tcn3(x_tcn)
        
        x_pooled = self.pool_for_transformer(x_tcn)
        x_trans = x_pooled.permute(0, 2, 1)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.mean(dim=1)
        
        x_adapted = self.adapter1(x_trans)
        
        # Concatenate Energy feature to FC layers to distinguish silent vs loud bearings
        x_combined = torch.cat([x_adapted, energy], dim=1)
        
        x_fc = self.relu(self.fc1(x_combined))
        x_fc = self.dropout(x_fc)
        x_fc = self.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        
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

# Legacy model for backward compatibility
class MSCAN_RUL_Legacy(nn.Module):
    def __init__(self, input_channels=2, num_filters=16, dropout_rate=0.3):
        super().__init__()
        self.conv1x1 = nn.Conv1d(input_channels, num_filters, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(input_channels, num_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.channel_attention = ChannelAttention(num_filters * 3, reduction=4)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1x1(x)))
        x2 = self.relu(self.bn2(self.conv3x3(x)))
        x3 = self.relu(self.bn3(self.conv5x5(x)))
        x_multi = torch.cat([x1, x2, x3], dim=1)
        x_ca = self.channel_attention(x_multi)
        x_sa = self.spatial_attention(x_ca)
        x_pool = self.global_pool(x_sa).view(x_sa.size(0), -1)
        x_fc1 = self.relu(self.fc1(x_pool))
        x_fc1 = self.dropout(x_fc1)
        x_fc2 = self.relu(self.fc2(x_fc1))
        x_fc2 = self.dropout(x_fc2)
        out = self.sigmoid(self.fc3(x_fc2))
        return out

# ============================================================================
# INITIALIZATION
# ============================================================================
MODEL_PATH = r"d:\prec machine\models\best_mscan_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 5120

print(f"Loading model on {DEVICE}...")
model = None
model_type = "hybrid"  # 'hybrid', 'legacy', 'untrained'

try:
    # Try Hybrid model first
    model = MSCAN_Hybrid(input_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_type = "hybrid"
    print("✓ Loaded HYBRID model (MSCAN + Transformer + Quantile RUL).")
except Exception as e:
    print(f"! Hybrid model load failed: {e}")
    try:
        # Fallback to legacy model
        model = MSCAN_RUL_Legacy(input_channels=2).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model_type = "legacy"
        print("✓ Loaded LEGACY model (Original MSCAN).")
    except Exception as e2:
        print(f"! Legacy model load failed: {e2}")
        print("WARNING: Using UNTRAINED Hybrid Model for demonstration.")
        model = MSCAN_Hybrid(input_channels=2).to(DEVICE)
        model_type = "untrained"

model.eval()

# Load Tabular Model
TABULAR_MODEL_PATH = r"d:\prec machine\models\tabular_model.pkl"
tabular_model = None
try:
    if os.path.exists(TABULAR_MODEL_PATH):
        tabular_model = joblib.load(TABULAR_MODEL_PATH)
        print("✓ Loaded Tabular Model for Manual Diagnostics.")
except Exception as e:
    print(f"Error loading tabular model: {e}")

# Load Scalers (for consistent preprocessing)
INPUT_SCALER_PATH = r"d:\prec machine\models\input_scaler.pkl"
QUANTILE_TRANSFORMER_PATH = r"d:\prec machine\models\quantile_transformer.pkl"
input_scaler = None
quantile_transformer = None

try:
    if os.path.exists(INPUT_SCALER_PATH) and os.path.exists(QUANTILE_TRANSFORMER_PATH):
        input_scaler = joblib.load(INPUT_SCALER_PATH)
        quantile_transformer = joblib.load(QUANTILE_TRANSFORMER_PATH)
        print("✓ Loaded Input Scalers (RobustScaler + QuantileTransformer).")
    else:
        print("! Scalers not found - will use simple z-score normalization.")
except Exception as e:
    print(f"Error loading scalers: {e}")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'demo' in request.form:
            # Generate REALISTIC demo signal with random health state
            health_state = np.random.choice(['healthy', 'degraded', 'critical'])
            t = np.linspace(0, 1, WINDOW_SIZE)
            
            # Base vibration (sinusoidal components at bearing frequencies)
            base_freq = np.random.uniform(50, 150)  # Random base frequency
            signal_h = 0.3 * np.sin(2 * np.pi * base_freq * t)
            signal_v = 0.3 * np.sin(2 * np.pi * base_freq * t + np.pi/4)
            
            # Add harmonics
            signal_h += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            signal_v += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            
            # Add health-dependent degradation
            if health_state == 'degraded':
                # Increased amplitude + some impulses
                signal_h *= np.random.uniform(1.5, 2.5)
                signal_v *= np.random.uniform(1.5, 2.5)
                # Add random impulses
                impulse_locs = np.random.randint(0, WINDOW_SIZE, 20)
                signal_h[impulse_locs] += np.random.uniform(1, 3, 20)
                signal_v[impulse_locs] += np.random.uniform(1, 3, 20)
            elif health_state == 'critical':
                # High amplitude + many impulses + drift
                signal_h *= np.random.uniform(3, 5)
                signal_v *= np.random.uniform(3, 5)
                signal_h += t * np.random.uniform(0.5, 2)  # Drift
                signal_v += t * np.random.uniform(0.5, 2)
                # Many impulses
                impulse_locs = np.random.randint(0, WINDOW_SIZE, 50)
                signal_h[impulse_locs] += np.random.uniform(2, 5, 50)
                signal_v[impulse_locs] += np.random.uniform(2, 5, 50)
            
            # Add realistic noise
            signal_h += np.random.normal(0, 0.1, WINDOW_SIZE)
            signal_v += np.random.normal(0, 0.1, WINDOW_SIZE)
            
            # Normalize (similar to training preprocessing)
            signal_h = (signal_h - np.mean(signal_h)) / (np.std(signal_h) + 1e-8)
            signal_v = (signal_v - np.mean(signal_v)) / (np.std(signal_v) + 1e-8)
            
            signal = np.stack([signal_h, signal_v], axis=0).astype(np.float32)
            
            with torch.no_grad():
                tensor_x = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)
                output = model(tensor_x).cpu().numpy().flatten()
            
            if model_type == "hybrid":
                rul_low = float(output[0])
                rul_median = float(output[1])
                rul_high = float(output[2])
                hi = float(output[3])
                speed = float(output[4])
                voltage = float(output[5])
            elif model_type == "legacy":
                rul_low = rul_median = rul_high = float(output[0])
                hi = rul_median
                speed = float(output[1]) if len(output) > 1 else 0.5
                voltage = float(output[2]) if len(output) > 2 else 0.5
            else:
                rul_low = rul_median = rul_high = 0.5
                hi = 0.5
                speed = voltage = 0.5
            
            return jsonify({
                'rul': rul_median,
                'rul_low': rul_low,
                'rul_high': rul_high,
                'hi': hi,
                'speed': speed,
                'voltage': voltage,
                'model_type': model_type,
                'demo_state': health_state,
                'message': f'Demo prediction ({health_state} bearing simulation)'
            })

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            df = pd.read_csv(file)
            cols = df.columns.tolist()
            h_col = next((c for c in cols if 'horiz' in c.lower()), cols[0])
            v_col = next((c for c in cols if 'vert' in c.lower()), cols[1] if len(cols)>1 else cols[0])
            
            sig_h = df[h_col].values.astype(np.float32)
            sig_v = df[v_col].values.astype(np.float32)
            
            current_len = len(sig_h)
            if current_len < WINDOW_SIZE:
                pad_len = WINDOW_SIZE - current_len
                sig_h = np.pad(sig_h, (0, pad_len), mode='edge')
                sig_v = np.pad(sig_v, (0, pad_len), mode='edge')
            elif current_len > WINDOW_SIZE:
                sig_h = sig_h[-WINDOW_SIZE:]
                sig_v = sig_v[-WINDOW_SIZE:]
            
            # Apply trained scalers (same as training preprocessing)
            if input_scaler is not None and quantile_transformer is not None:
                # Stack and reshape for scaler (samples, features)
                raw_signal = np.stack([sig_h, sig_v], axis=1)  # (WINDOW_SIZE, 2)
                scaled = input_scaler.transform(raw_signal)
                normalized = quantile_transformer.transform(scaled)
                sig_h = normalized[:, 0]
                sig_v = normalized[:, 1]
            else:
                # Fallback: simple z-score
                sig_h = (sig_h - np.mean(sig_h)) / (np.std(sig_h) + 1e-8)
                sig_v = (sig_v - np.mean(sig_v)) / (np.std(sig_v) + 1e-8)
            
            signal = np.stack([sig_h, sig_v], axis=0).astype(np.float32)
            
            with torch.no_grad():
                tensor_x = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)
                output = model(tensor_x).cpu().numpy().flatten()
            
            if model_type == "hybrid":
                rul_low = float(output[0])
                rul_median = float(output[1])
                rul_high = float(output[2])
                hi = float(output[3])
                speed = float(output[4])
                voltage = float(output[5])
            elif model_type == "legacy":
                rul_low = rul_median = rul_high = float(output[0])
                hi = rul_median
                speed = float(output[1]) if len(output) > 1 else 0.5
                voltage = float(output[2]) if len(output) > 2 else 0.5
            else:
                rul_low = rul_median = rul_high = 0.5
                hi = 0.5
                speed = voltage = 0.5
            
            # Health Indicator Inversion & Safeguard
            # In training, HI=1 means damage, so for UI we want 1 - hi
            health_remaining = 1.0 - hi
            
            # Safeguard: If normalized signal energy (std) is extremely low,
            # it indicates a healthy/new bearing with vibration below the training noise floor.
            sig_intensity = np.std(sig_h) + np.std(sig_v)
            if sig_intensity < 0.2: # Very low normalized vibration relative to dataset
                health_remaining = max(health_remaining, 0.98)
                rul_median = max(rul_median, 0.95)
                rul_low = max(rul_low, 0.85)
                rul_high = max(rul_high, 0.99)
            
            return jsonify({
                'rul': float(rul_median),
                'rul_low': float(rul_low),
                'rul_high': float(rul_high),
                'hi': float(health_remaining),
                'speed': float(speed),
                'voltage': float(voltage),
                'model_type': model_type
            })
            
        except Exception as e:
            return jsonify({'error': f"Processing error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json
        
        temp_coolant = float(data.get('coolant', 50))
        temp_stator = float(data.get('stator_winding', 60))
        torque = float(data.get('torque', 20))
        u_q = float(data.get('u_q', 100))
        i_d = float(data.get('i_d', -80))
        
        rul_pred = 0.5
        rul_low = 0.4
        rul_high = 0.6
        model_used = "Heuristic"
        
        if tabular_model:
            input_vector = np.array([[temp_coolant, temp_stator, torque, u_q, i_d]])
            rul_pred = float(tabular_model.predict(input_vector)[0])
            model_used = "Random Forest Model"
            rul_pred = max(0.01, min(1.0, rul_pred))
            rul_low = max(0.01, rul_pred - 0.1)
            rul_high = min(1.0, rul_pred + 0.1)
        else:
            rul_pred = 1.0
            avg_temp = (temp_coolant + temp_stator) / 2
            if avg_temp > 90: rul_pred -= 0.4
            elif avg_temp > 70: rul_pred -= 0.15
            if abs(torque) > 55: rul_pred -= 0.3
            elif abs(torque) > 40: rul_pred -= 0.1
            if abs(i_d) > 150: rul_pred -= 0.2
            rul_pred = max(0.05, min(1.0, rul_pred))
            rul_low = max(0.01, rul_pred - 0.15)
            rul_high = min(1.0, rul_pred + 0.15)

        impacts = {'thermal': 'Normal', 'load': 'Normal', 'elec': 'Normal'}
        
        avg_temp = (temp_coolant + temp_stator) / 2
        if avg_temp > 90: impacts['thermal'] = 'CRITICAL'
        elif avg_temp > 75: impacts['thermal'] = 'High'
        
        if abs(torque) > 50: impacts['load'] = 'CRITICAL'
        elif abs(torque) > 35: impacts['load'] = 'High'
        
        if abs(i_d) > 120: impacts['elec'] = 'Abnormal'

        return jsonify({
            'rul': rul_pred,
            'rul_low': rul_low,
            'rul_high': rul_high,
            'impact': impacts,
            'method': model_used
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    return jsonify({
        'model_type': model_type,
        'device': str(DEVICE),
        'features': [
            'Multi-Scale CNN (1x1, 3x3, 5x5, 7x7)',
            'Temporal Convolutional Network',
            'Lightweight Transformer',
            'Domain Adaptation (DANN)',
            'Quantile Regression (Uncertainty)',
            'Multi-Task Learning'
        ] if model_type == "hybrid" else ['Legacy MSCAN']
    })

@app.route('/results/<image_name>')
def get_image(image_name):
    image_map = {
        'training_graphs': '03_multi_output_results.png',
        'feature_importance': '01_feature_importance_analysis.png'
    }
    
    filename = image_map.get(image_name)
    if not filename:
        return "Image not found", 404
    
    filepath = os.path.join(r"d:\prec machine\output", filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return "Graph not generated yet", 404

if __name__ == '__main__':
    print("Starting Dashboard Server...")
    app.run(debug=True, port=5000)
