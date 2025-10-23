"""
Configuration file for IoT Sensor Anomaly Detection Project
"""

import os
import sys

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add src to path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data generation parameters
RANDOM_SEED = 42
NUM_SAMPLES = 10000
SAMPLING_RATE = '1min'  # 1 minute intervals

# Sensor parameters
SENSORS = {
    'temperature': {'mean': 75, 'std': 5, 'unit': '°F'},
    'vibration': {'mean': 0.5, 'std': 0.1, 'unit': 'mm/s'},
    'pressure': {'mean': 100, 'std': 10, 'unit': 'PSI'},
    'rotation_speed': {'mean': 1500, 'std': 50, 'unit': 'RPM'}
}

# Anomaly injection parameters
ANOMALY_RATIO = 0.05  # 5% of data points will be anomalies
ANOMALY_TYPES = ['spike', 'drift', 'level_shift', 'noise']

# Feature engineering parameters
ROLLING_WINDOWS = [5, 10, 30, 60]  # in minutes
LAG_FEATURES = [1, 5, 10]

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.05,
    'random_state': RANDOM_SEED,
    'max_samples': 'auto',
    'n_jobs': -1
}

LOF_PARAMS = {
    'n_neighbors': 20,
    'contamination': 0.05,
    'n_jobs': -1
}

# LSTM Autoencoder parameters
LSTM_PARAMS = {
    'sequence_length': 60,
    'encoding_dim': 32,
    'lstm_units': [64, 32],
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001
}

# Evaluation parameters
TEST_SIZE = 0.3
THRESHOLD_PERCENTILE = 95  # For reconstruction error threshold

