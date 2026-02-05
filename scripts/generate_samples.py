import pandas as pd
import numpy as np
import os

def create_sample_csv(filename, health='healthy'):
    os.makedirs('samples', exist_ok=True)
    num_samples = 2560
    t = np.linspace(0, 1, num_samples)
    
    # Base frequencies
    f1 = 50.0
    f2 = 120.0
    
    if health == 'healthy':
        vib_h = 0.2 * np.sin(2 * np.pi * f1 * t) + 0.1 * np.random.normal(0, 1, num_samples)
        vib_v = 0.2 * np.cos(2 * np.pi * f1 * t) + 0.1 * np.random.normal(0, 1, num_samples)
        temp = 40.0 + np.random.normal(0, 0.5, num_samples)
    elif health == 'degraded':
        vib_h = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.random.normal(0, 1, num_samples)
        vib_v = 0.5 * np.cos(2 * np.pi * f1 * t) + 0.3 * np.cos(2 * np.pi * f2 * t) + 0.2 * np.random.normal(0, 1, num_samples)
        # Add some spikes
        spikes = np.random.choice(num_samples, 20)
        vib_h[spikes] += 2.0
        vib_v[spikes] += 2.0
        temp = 60.0 + np.random.normal(0, 1, num_samples)
    else: # critical
        vib_h = 1.2 * np.sin(2 * np.pi * f1 * t) + 0.8 * np.sin(2 * np.pi * f2 * t) + 0.5 * np.random.normal(0, 1, num_samples)
        vib_v = 1.2 * np.cos(2 * np.pi * f1 * t) + 0.8 * np.cos(2 * np.pi * f2 * t) + 0.5 * np.random.normal(0, 1, num_samples)
        # Add many spikes
        spikes = np.random.choice(num_samples, 100)
        vib_h[spikes] += 5.0
        vib_v[spikes] += 5.0
        temp = 85.0 + np.random.normal(0, 2, num_samples)

    df = pd.DataFrame({
        'Horizontal_vibration': vib_h,
        'Vertical_vibration': vib_v,
        'Temperature': temp
    })
    
    path = os.path.join('samples', filename)
    df.to_csv(path, index=False)
    print(f"Created {path}")

if __name__ == "__main__":
    create_sample_csv('test_healthy.csv', 'healthy')
    create_sample_csv('test_degraded.csv', 'degraded')
    create_sample_csv('test_critical.csv', 'critical')
