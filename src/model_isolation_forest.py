"""
Isolation Forest Anomaly Detection Model
Implements statistical/unsupervised anomaly detection using Isolation Forest algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Anomaly detection using Isolation Forest algorithm.
    
    Intuition:
    Isolation Forest works on the principle that anomalies are "few and different".
    It isolates observations by randomly selecting a feature and then randomly 
    selecting a split value between the maximum and minimum values of the selected feature.
    Anomalies are easier to isolate (require fewer splits) than normal points.
    
    The algorithm is particularly effective for:
    - High-dimensional data
    - Large datasets
    - Detecting global anomalies
    """
    
    def __init__(self, contamination: float = config.ISOLATION_FOREST_PARAMS['contamination'],
                 n_estimators: int = config.ISOLATION_FOREST_PARAMS['n_estimators'],
                 random_state: int = config.RANDOM_SEED):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            n_estimators: Number of base estimators (trees) in the ensemble
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples='auto',
            n_jobs=-1,
            verbose=0
        )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.anomaly_scores = None
        self.threshold = None
        
        logger.info(f"Initialized IsolationForest with contamination={contamination}, "
                   f"n_estimators={n_estimators}")
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Training data (features)
            
        Returns:
            Self
        """
        logger.info(f"Training Isolation Forest on data with shape {X.shape}...")
        self.model.fit(X)
        
        # Calculate anomaly scores for training data
        self.anomaly_scores = -self.model.score_samples(X)  # Negative for intuitive interpretation
        self.threshold = np.percentile(self.anomaly_scores, 100 * (1 - self.contamination))
        
        logger.info(f"Training complete. Anomaly score threshold: {self.threshold:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        predictions = self.model.predict(X)
        # Convert from {-1, 1} to {1, 0}
        predictions = np.where(predictions == -1, 1, 0)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (probabilities).
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores
        """
        scores = -self.model.score_samples(X)
        return scores
    
    def get_feature_importance(self, X: np.ndarray, feature_names: list) -> pd.DataFrame:
        """
        Calculate feature importance based on anomaly detection.
        
        Args:
            X: Data array
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        logger.info("Calculating feature importance...")
        
        # Get baseline anomaly scores
        baseline_scores = self.predict_proba(X)
        baseline_mean = np.mean(baseline_scores)
        
        importance_scores = []
        
        for i, feature_name in enumerate(feature_names):
            # Shuffle feature
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])
            
            # Get new scores
            shuffled_scores = self.predict_proba(X_shuffled)
            shuffled_mean = np.mean(shuffled_scores)
            
            # Importance is the change in mean anomaly score
            importance = abs(baseline_mean - shuffled_mean)
            importance_scores.append(importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filename: str = 'isolation_forest_model.pkl'):
        """
        Save the trained model.
        
        Args:
            filename: Output filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = 'isolation_forest_model.pkl'):
        """
        Load a trained model.
        
        Args:
            filename: Input filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class LOFDetector:
    """
    Anomaly detection using Local Outlier Factor algorithm.
    
    Intuition:
    LOF measures the local deviation of density of a given sample with respect to 
    its neighbors. It considers as outliers the samples that have a substantially 
    lower density than their neighbors.
    
    The algorithm is particularly effective for:
    - Detecting local anomalies (anomalies in specific regions)
    - Datasets with varying densities
    - Smaller datasets
    """
    
    def __init__(self, contamination: float = config.LOF_PARAMS['contamination'],
                 n_neighbors: int = config.LOF_PARAMS['n_neighbors']):
        """
        Initialize LOF detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_neighbors: Number of neighbors to consider
        """
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=True,  # Enable prediction on new data
            n_jobs=-1
        )
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        
        logger.info(f"Initialized LOF with contamination={contamination}, "
                   f"n_neighbors={n_neighbors}")
    
    def fit(self, X: np.ndarray) -> 'LOFDetector':
        """
        Fit the LOF model.
        
        Args:
            X: Training data
            
        Returns:
            Self
        """
        logger.info(f"Training LOF on data with shape {X.shape}...")
        self.model.fit(X)
        logger.info("Training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary predictions
        """
        predictions = self.model.predict(X)
        # Convert from {-1, 1} to {1, 0}
        predictions = np.where(predictions == -1, 1, 0)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores
        """
        scores = -self.model.score_samples(X)
        return scores
    
    def save_model(self, filename: str = 'lof_model.pkl'):
        """
        Save the trained model.
        
        Args:
            filename: Output filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = 'lof_model.pkl'):
        """
        Load a trained model.
        
        Args:
            filename: Input filename
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray, 
                        model_name: str, save_path: str = None):
    """
    Plot distribution of anomaly scores.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Score distribution
    ax1 = axes[0]
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
    ax1.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{model_name} - Anomaly Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [normal_scores, anomaly_scores]
    bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title(f'{model_name} - Anomaly Score Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Anomaly score plot saved to {save_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Test Isolation Forest
    logger.info("Testing Isolation Forest detector...")
    
    # Load preprocessed data
    from data_preprocessing import DataPreprocessor, FeatureEngineer
    
    df = pd.read_csv(f"{config.DATA_DIR}/sensor_data.csv", index_col=0, parse_dates=True)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    X_scaled, feature_names = engineer.scale_features(df_features, fit=True)
    y = df_features['anomaly'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    
    # Train Isolation Forest
    if_detector = IsolationForestDetector()
    if_detector.fit(X_train)
    
    # Predict
    y_pred = if_detector.predict(X_test)
    scores = if_detector.predict_proba(X_test)
    
    # Plot
    plot_anomaly_scores(scores, y_test, 'Isolation Forest',
                       f"{config.PLOTS_DIR}/if_anomaly_scores.png")
    
    logger.info(f"Predictions: {np.unique(y_pred, return_counts=True)}")
    logger.info(f"True labels: {np.unique(y_test, return_counts=True)}")

