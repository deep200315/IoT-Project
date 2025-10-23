"""
Data Preprocessing and Feature Engineering Module
Handles data cleaning, missing value imputation, and feature creation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import logging
from typing import Tuple, List, Dict
import joblib

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing including missing value imputation and outlier handling.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = None
        self.feature_columns = None
        logger.info("Initialized DataPreprocessor")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward fill and mean imputation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Handling missing values...")
        df_copy = df.copy()
        
        # Get sensor columns (exclude timestamp and anomaly label)
        sensor_cols = [col for col in df_copy.columns if col != 'anomaly']
        
        # First, try forward fill (useful for time series)
        df_copy[sensor_cols] = df_copy[sensor_cols].fillna(method='ffill')
        
        # Then backward fill for any remaining NaNs at the start
        df_copy[sensor_cols] = df_copy[sensor_cols].fillna(method='bfill')
        
        # Finally, use mean imputation for any remaining NaNs
        if df_copy[sensor_cols].isnull().any().any():
            df_copy[sensor_cols] = pd.DataFrame(
                self.imputer.fit_transform(df_copy[sensor_cols]),
                columns=sensor_cols,
                index=df_copy.index
            )
        
        logger.info(f"Missing values after imputation: {df_copy.isnull().sum().sum()}")
        return df_copy
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str], 
                           threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: IQR threshold multiplier
            
        Returns:
            DataFrame with outlier indicators
        """
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outlier_mask
    
    def get_preprocessing_report(self, df_original: pd.DataFrame, 
                                df_processed: pd.DataFrame) -> Dict:
        """
        Generate preprocessing report.
        
        Args:
            df_original: Original DataFrame
            df_processed: Processed DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        report = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'missing_values_original': df_original.isnull().sum().to_dict(),
            'missing_values_processed': df_processed.isnull().sum().to_dict(),
        }
        return report


class FeatureEngineer:
    """
    Creates features from raw time series data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_names = []
        logger.info("Initialized FeatureEngineer")
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str],
                               windows: List[int] = config.ROLLING_WINDOWS) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            columns: Columns to create features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling window features...")
        df_features = df.copy()
        
        for col in columns:
            for window in windows:
                # Rolling mean
                df_features[f'{col}_rolling_mean_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation
                df_features[f'{col}_rolling_std_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min and max
                df_features[f'{col}_rolling_min_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df_features[f'{col}_rolling_max_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )
        
        logger.info(f"Created rolling features. New shape: {df_features.shape}")
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lags: List[int] = config.LAG_FEATURES) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create features for
            lags: List of lag values
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        df_features = df.copy()
        
        for col in columns:
            for lag in lags:
                df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Fill NaN values created by lagging
        df_features = df_features.fillna(method='bfill')
        
        logger.info(f"Created lag features. New shape: {df_features.shape}")
        return df_features
    
    def create_difference_features(self, df: pd.DataFrame, 
                                   columns: List[str]) -> pd.DataFrame:
        """
        Create difference and rate of change features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create features for
            
        Returns:
            DataFrame with difference features
        """
        logger.info("Creating difference features...")
        df_features = df.copy()
        
        for col in columns:
            # First difference
            df_features[f'{col}_diff'] = df[col].diff()
            
            # Second difference
            df_features[f'{col}_diff2'] = df[col].diff().diff()
            
            # Rate of change
            df_features[f'{col}_roc'] = df[col].pct_change()
        
        # Fill NaN values
        df_features = df_features.fillna(method='bfill')
        
        logger.info(f"Created difference features. New shape: {df_features.shape}")
        return df_features
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                    columns: List[str],
                                    window: int = 30) -> pd.DataFrame:
        """
        Create statistical features over a rolling window.
        
        Args:
            df: Input DataFrame
            columns: Columns to create features for
            window: Window size for statistics
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Creating statistical features...")
        df_features = df.copy()
        
        for col in columns:
            # Skewness
            df_features[f'{col}_skew_{window}'] = (
                df[col].rolling(window=window, min_periods=1).skew()
            )
            
            # Kurtosis
            df_features[f'{col}_kurt_{window}'] = (
                df[col].rolling(window=window, min_periods=1).kurt()
            )
            
            # Quantiles
            df_features[f'{col}_q25_{window}'] = (
                df[col].rolling(window=window, min_periods=1).quantile(0.25)
            )
            df_features[f'{col}_q75_{window}'] = (
                df[col].rolling(window=window, min_periods=1).quantile(0.75)
            )
        
        # Fill NaN values
        df_features = df_features.fillna(method='bfill')
        
        logger.info(f"Created statistical features. New shape: {df_features.shape}")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                    columns: List[str]) -> pd.DataFrame:
        """
        Create interaction features between sensors.
        
        Args:
            df: Input DataFrame
            columns: Columns to create features for
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        df_features = df.copy()
        
        # Create ratios between sensors
        if 'temperature' in columns and 'vibration' in columns:
            df_features['temp_vibration_ratio'] = df['temperature'] / (df['vibration'] + 1e-6)
        
        if 'pressure' in columns and 'rotation_speed' in columns:
            df_features['pressure_speed_ratio'] = df['pressure'] / (df['rotation_speed'] + 1e-6)
        
        # Create products
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                df_features[f'{col1}_{col2}_product'] = df[col1] * df[col2]
        
        logger.info(f"Created interaction features. New shape: {df_features.shape}")
        return df_features
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns if col in config.SENSORS.keys()]
        
        # Start with original data
        df_features = df.copy()
        
        # Create rolling features
        df_features = self.create_rolling_features(df_features, sensor_cols)
        
        # Create lag features
        df_features = self.create_lag_features(df_features, sensor_cols)
        
        # Create difference features
        df_features = self.create_difference_features(df_features, sensor_cols)
        
        # Create statistical features
        df_features = self.create_statistical_features(df_features, sensor_cols)
        
        # Create interaction features
        df_features = self.create_interaction_features(df_features, sensor_cols)
        
        # Fill any remaining NaN values
        df_features = df_features.fillna(method='bfill').fillna(method='ffill')
        
        # Store feature names (excluding anomaly label)
        self.feature_names = [col for col in df_features.columns if col != 'anomaly']
        
        logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
        logger.info(f"Total features created: {len(self.feature_names)}")
        
        return df_features
    
    def scale_features(self, df: pd.DataFrame, 
                      fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Scale features using RobustScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (scaled features array, feature names)
        """
        logger.info("Scaling features...")
        
        # Get feature columns (exclude anomaly label)
        feature_cols = [col for col in df.columns if col != 'anomaly']
        
        if fit:
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            logger.info("Scaler fitted and transformed")
        else:
            scaled_features = self.scaler.transform(df[feature_cols])
            logger.info("Scaler transformed")
        
        return scaled_features, feature_cols
    
    def save_scaler(self, filename: str = 'feature_scaler.pkl'):
        """
        Save the fitted scaler.
        
        Args:
            filename: Output filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filename: str = 'feature_scaler.pkl'):
        """
        Load a fitted scaler.
        
        Args:
            filename: Input filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")


if __name__ == "__main__":
    # Test preprocessing pipeline
    logger.info("Testing preprocessing pipeline...")
    
    # Load data
    df = pd.read_csv(f"{config.DATA_DIR}/sensor_data.csv", index_col=0, parse_dates=True)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Scale features
    X_scaled, feature_names = engineer.scale_features(df_features, fit=True)
    
    logger.info(f"Preprocessing complete. Scaled features shape: {X_scaled.shape}")
    logger.info(f"Feature names: {feature_names[:10]}... (showing first 10)")

