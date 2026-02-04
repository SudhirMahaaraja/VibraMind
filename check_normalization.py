import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add current dir to path to import classes
sys.path.append(os.getcwd())

# Need the class definitions from main.py or app.py
# Assuming they are available or I can mock them enough to load

# Manual load of MSCAN_Hybrid (from app.py or main.py)
class MSCAN_Hybrid(nn.Module):
    def __init__(self, input_channels=2, num_filters=16, dropout_rate=0.3, num_quantiles=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, 6) # RUL(3) + HI(1) + Speed(1) + Volt(1)
    def forward(self, x):
        # Mock forward that just returns dummy if structure differs
        return torch.zeros(x.size(0), 6)

# Actually I'll just use the one from app.py by importing it?
# No, let's just check the scalers and data first.

def check_data(file_path):
    df = pd.read_csv(file_path)
    cols = df.columns
    h = df.iloc[:, 0].values
    v = df.iloc[:, 1].values
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"  Raw H: mean={h.mean():.4f}, std={h.std():.4f}, max={h.max():.4f}, min={h.min():.4f}")
    
    # Load scalers
    input_scaler = joblib.load('models/input_scaler.pkl')
    qt = joblib.load('models/quantile_transformer.pkl')
    
    raw = np.stack([h, v], axis=1)[:5120]
    scaled = input_scaler.transform(raw)
    norm = qt.transform(scaled)
    
    print(f"  Norm H: mean={norm[:, 0].mean():.4f}, std={norm[:, 0].std():.4f}, max={norm[:, 0].max():.4f}, min={norm[:, 0].min():.4f}")

print("--- DATA CHECK ---")
check_data('samples/sample_healthy_100_rul.csv')
print("")
check_data('samples/sample_critical_low_rul.csv')
