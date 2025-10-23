"""
Synthetic IoT Sensor Data Generator with Embedded Anomalies
This module generates realistic time series data for IoT sensors with various types of anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SensorDataGenerator:
    """
    Generates synthetic sensor data with realistic patterns and embedded anomalies.
    """
    
    def __init__(self, num_samples: int = config.NUM_SAMPLES, 
                 random_seed: int = config.RANDOM_SEED):
        """
        Initialize the data generator.
        
        Args:
            num_samples: Number of time series samples to generate
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"Initialized SensorDataGenerator with {num_samples} samples")
        
    def generate_base_signal(self, mean: float, std: float, 
                            add_trend: bool = True, 
                            add_seasonality: bool = True) -> np.ndarray:
        """
        Generate base signal with trend and seasonality.
        
        Args:
            mean: Mean value of the signal
            std: Standard deviation of the signal
            add_trend: Whether to add a linear trend
            add_seasonality: Whether to add seasonal patterns
            
        Returns:
            Base signal array
        """
        # Base random noise
        signal = np.random.normal(mean, std, self.num_samples)
        
        # Add trend (slight upward drift over time)
        if add_trend:
            trend = np.linspace(0, mean * 0.1, self.num_samples)
            signal += trend
            
        # Add daily seasonality (assuming 1-minute sampling)
        if add_seasonality:
            # Daily cycle (1440 minutes in a day)
            daily_cycle = 0.1 * mean * np.sin(2 * np.pi * np.arange(self.num_samples) / 1440)
            # Hourly cycle (60 minutes in an hour)
            hourly_cycle = 0.05 * mean * np.sin(2 * np.pi * np.arange(self.num_samples) / 60)
            signal += daily_cycle + hourly_cycle
            
        return signal
    
    def inject_spike_anomaly(self, signal: np.ndarray, indices: List[int], 
                            magnitude: float = 3.0) -> np.ndarray:
        """
        Inject spike anomalies (sudden jumps).
        
        Args:
            signal: Original signal
            indices: Indices where anomalies should be injected
            magnitude: Magnitude of the spike (in standard deviations)
            
        Returns:
            Signal with spike anomalies
        """
        signal_copy = signal.copy()
        std = np.std(signal)
        for idx in indices:
            signal_copy[idx] += np.random.choice([-1, 1]) * magnitude * std
        return signal_copy
    
    def inject_drift_anomaly(self, signal: np.ndarray, start_idx: int, 
                            duration: int, rate: float = 0.01) -> np.ndarray:
        """
        Inject drift anomaly (gradual increase/decrease).
        
        Args:
            signal: Original signal
            start_idx: Starting index of the drift
            duration: Duration of the drift
            rate: Rate of drift per sample
            
        Returns:
            Signal with drift anomaly
        """
        signal_copy = signal.copy()
        end_idx = min(start_idx + duration, len(signal))
        drift = np.linspace(0, rate * duration * np.std(signal), end_idx - start_idx)
        signal_copy[start_idx:end_idx] += drift * np.random.choice([-1, 1])
        return signal_copy
    
    def inject_level_shift_anomaly(self, signal: np.ndarray, start_idx: int, 
                                   duration: int, shift: float = 2.0) -> np.ndarray:
        """
        Inject level shift anomaly (sustained change in mean).
        
        Args:
            signal: Original signal
            start_idx: Starting index of the level shift
            duration: Duration of the level shift
            shift: Magnitude of shift (in standard deviations)
            
        Returns:
            Signal with level shift anomaly
        """
        signal_copy = signal.copy()
        end_idx = min(start_idx + duration, len(signal))
        signal_copy[start_idx:end_idx] += shift * np.std(signal) * np.random.choice([-1, 1])
        return signal_copy
    
    def inject_noise_anomaly(self, signal: np.ndarray, indices: List[int], 
                            noise_factor: float = 5.0) -> np.ndarray:
        """
        Inject noise anomaly (increased variance).
        
        Args:
            signal: Original signal
            indices: Indices where anomalies should be injected
            noise_factor: Factor to multiply the noise by
            
        Returns:
            Signal with noise anomalies
        """
        signal_copy = signal.copy()
        std = np.std(signal)
        for idx in indices:
            signal_copy[idx] += np.random.normal(0, noise_factor * std)
        return signal_copy
    
    def generate_anomalies(self, signal: np.ndarray, 
                          anomaly_ratio: float = config.ANOMALY_RATIO) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate various types of anomalies in the signal.
        
        Args:
            signal: Original signal
            anomaly_ratio: Ratio of anomalous points
            
        Returns:
            Tuple of (signal with anomalies, binary labels)
        """
        labels = np.zeros(len(signal), dtype=int)
        signal_with_anomalies = signal.copy()
        
        num_anomalies = int(len(signal) * anomaly_ratio)
        
        # Distribute anomalies across different types
        num_spikes = num_anomalies // 4
        num_drifts = 3  # Number of drift events
        num_level_shifts = 3  # Number of level shift events
        num_noise = num_anomalies // 4
        
        # Spike anomalies
        spike_indices = np.random.choice(len(signal), num_spikes, replace=False)
        signal_with_anomalies = self.inject_spike_anomaly(signal_with_anomalies, spike_indices)
        labels[spike_indices] = 1
        
        # Drift anomalies
        if len(signal) > 200:
            for _ in range(num_drifts):
                start_idx = np.random.randint(0, len(signal) - 200)
                duration = np.random.randint(50, 200)
                signal_with_anomalies = self.inject_drift_anomaly(
                    signal_with_anomalies, start_idx, duration
                )
                labels[start_idx:start_idx + duration] = 1
        
        # Level shift anomalies
        if len(signal) > 150:
            for _ in range(num_level_shifts):
                start_idx = np.random.randint(0, len(signal) - 150)
                duration = np.random.randint(30, 150)
                signal_with_anomalies = self.inject_level_shift_anomaly(
                    signal_with_anomalies, start_idx, duration
                )
                labels[start_idx:start_idx + duration] = 1
        
        # Noise anomalies
        available_indices = np.where(labels == 0)[0]
        noise_indices = np.random.choice(available_indices, 
                                        min(num_noise, len(available_indices)), 
                                        replace=False)
        signal_with_anomalies = self.inject_noise_anomaly(signal_with_anomalies, noise_indices)
        labels[noise_indices] = 1
        
        logger.info(f"Injected {np.sum(labels)} anomalies ({np.sum(labels)/len(labels)*100:.2f}%)")
        
        return signal_with_anomalies, labels
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete multi-sensor dataset with anomalies.
        
        Returns:
            DataFrame with sensor readings and anomaly labels
        """
        logger.info("Generating synthetic sensor dataset...")
        
        # Create timestamp index
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(self.num_samples)]
        
        # Initialize DataFrame
        df = pd.DataFrame({'timestamp': timestamps})
        df.set_index('timestamp', inplace=True)
        
        # Generate data for each sensor
        all_labels = []
        
        for sensor_name, params in config.SENSORS.items():
            logger.info(f"Generating {sensor_name} data...")
            
            # Generate base signal
            base_signal = self.generate_base_signal(
                params['mean'], 
                params['std'],
                add_trend=True,
                add_seasonality=True
            )
            
            # Inject anomalies
            signal_with_anomalies, labels = self.generate_anomalies(base_signal)
            
            # Add to DataFrame
            df[sensor_name] = signal_with_anomalies
            all_labels.append(labels)
        
        # Combine labels (if any sensor has anomaly, mark as anomaly)
        combined_labels = np.any(np.array(all_labels), axis=0).astype(int)
        df['anomaly'] = combined_labels
        
        # Add some missing values randomly (2% of data)
        for sensor_name in config.SENSORS.keys():
            missing_indices = np.random.choice(
                len(df), 
                int(len(df) * 0.02), 
                replace=False
            )
            df.loc[df.index[missing_indices], sensor_name] = np.nan
        
        logger.info(f"Dataset generated with shape: {df.shape}")
        logger.info(f"Total anomalies: {np.sum(combined_labels)} ({np.sum(combined_labels)/len(df)*100:.2f}%)")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'sensor_data.csv'):
        """
        Save dataset to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = f"{config.DATA_DIR}/{filename}"
        df.to_csv(filepath)
        logger.info(f"Dataset saved to {filepath}")


if __name__ == "__main__":
    # Generate and save dataset
    generator = SensorDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    # Display basic statistics
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nAnomaly distribution:")
    print(df['anomaly'].value_counts())

