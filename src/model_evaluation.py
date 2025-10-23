"""
Model Evaluation and Comparison Module
Provides comprehensive evaluation metrics and visualizations for anomaly detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple
import json

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for anomaly detection.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {}
        logger.info(f"Initialized ModelEvaluator for {model_name}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_scores: np.ndarray = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Calculating metrics for {self.model_name}...")
        
        # Basic metrics
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            self.metrics['true_negatives'] = int(tn)
            self.metrics['false_positives'] = int(fp)
            self.metrics['false_negatives'] = int(fn)
            self.metrics['true_positives'] = int(tp)
            
            # Specificity
            self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # False positive rate
            self.metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # ROC-AUC if scores are provided
        if y_scores is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                self.metrics['roc_auc'] = auc(fpr, tpr)
                
                # Precision-Recall AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                self.metrics['pr_auc'] = auc(recall_curve, precision_curve)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                self.metrics['roc_auc'] = None
                self.metrics['pr_auc'] = None
        
        logger.info(f"Metrics calculated: {self.metrics}")
        return self.metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      save_path: str = None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   save_path: str = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_anomaly_timeline(self, timestamps: pd.DatetimeIndex,
                             y_true: np.ndarray, y_pred: np.ndarray,
                             sensor_data: np.ndarray = None,
                             sensor_name: str = 'Sensor',
                             save_path: str = None):
        """
        Plot anomaly detection results over time.
        
        Args:
            timestamps: Time index
            y_true: True labels
            y_pred: Predicted labels
            sensor_data: Sensor values to plot (optional)
            sensor_name: Name of the sensor
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot sensor data with anomalies
        ax1 = axes[0]
        if sensor_data is not None:
            ax1.plot(timestamps, sensor_data, alpha=0.7, linewidth=0.5, label='Sensor Data')
        
        # True anomalies
        true_anomaly_mask = y_true == 1
        ax1.scatter(timestamps[true_anomaly_mask], 
                   sensor_data[true_anomaly_mask] if sensor_data is not None else np.ones(np.sum(true_anomaly_mask)),
                   color='red', s=20, alpha=0.6, label='True Anomaly', marker='o')
        
        ax1.set_title(f'{sensor_name} with True Anomalies', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predictions
        ax2 = axes[1]
        if sensor_data is not None:
            ax2.plot(timestamps, sensor_data, alpha=0.7, linewidth=0.5, label='Sensor Data')
        
        # Predicted anomalies
        pred_anomaly_mask = y_pred == 1
        
        # True positives (correctly detected anomalies)
        tp_mask = (y_true == 1) & (y_pred == 1)
        ax2.scatter(timestamps[tp_mask],
                   sensor_data[tp_mask] if sensor_data is not None else np.ones(np.sum(tp_mask)),
                   color='green', s=20, alpha=0.6, label='True Positive', marker='o')
        
        # False positives (incorrectly flagged as anomalies)
        fp_mask = (y_true == 0) & (y_pred == 1)
        ax2.scatter(timestamps[fp_mask],
                   sensor_data[fp_mask] if sensor_data is not None else np.ones(np.sum(fp_mask)),
                   color='orange', s=20, alpha=0.6, label='False Positive', marker='x')
        
        # False negatives (missed anomalies)
        fn_mask = (y_true == 1) & (y_pred == 0)
        ax2.scatter(timestamps[fn_mask],
                   sensor_data[fn_mask] if sensor_data is not None else np.ones(np.sum(fn_mask)),
                   color='red', s=20, alpha=0.6, label='False Negative', marker='^')
        
        ax2.set_title(f'{sensor_name} with Predicted Anomalies - {self.model_name}', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly timeline saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=['Normal', 'Anomaly'],
                                      zero_division=0)
        return report
    
    def save_metrics(self, filename: str = None):
        """
        Save metrics to JSON file.
        
        Args:
            filename: Output filename
        """
        if filename is None:
            filename = f"{self.model_name.lower().replace(' ', '_')}_metrics.json"
        
        filepath = f"{config.RESULTS_DIR}/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")


class ModelComparator:
    """
    Compare multiple anomaly detection models.
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.models_results = {}
        logger.info("Initialized ModelComparator")
    
    def add_model_results(self, model_name: str, metrics: Dict):
        """
        Add model results for comparison.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        self.models_results[model_name] = metrics
        logger.info(f"Added results for {model_name}")
    
    def plot_metrics_comparison(self, save_path: str = None):
        """
        Plot comparison of key metrics across models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.models_results:
            logger.warning("No model results to compare")
            return
        
        # Prepare data
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(self.models_results.keys())
        
        data = {metric: [] for metric in metrics_to_compare}
        
        for model in models:
            for metric in metrics_to_compare:
                value = self.models_results[model].get(metric, 0)
                if value is None:
                    value = 0
                data[metric].append(value)
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics_to_compare):
            ax = axes[idx]
            x = np.arange(len(models))
            bars = ax.bar(x, data[metric], alpha=0.8)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(data[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_confusion_matrices_comparison(self, y_true: np.ndarray,
                                          predictions_dict: Dict[str, np.ndarray],
                                          save_path: str = None):
        """
        Plot confusion matrices for all models side by side.
        
        Args:
            y_true: True labels
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Path to save the plot
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       ax=ax)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label' if idx == 0 else '')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices comparison saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate comparison report as DataFrame.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.models_results:
            logger.warning("No model results to compare")
            return pd.DataFrame()
        
        # Create DataFrame
        metrics_to_include = ['accuracy', 'precision', 'recall', 'f1_score', 
                             'roc_auc', 'pr_auc', 'specificity']
        
        data = []
        for model_name, metrics in self.models_results.items():
            row = {'Model': model_name}
            for metric in metrics_to_include:
                value = metrics.get(metric, None)
                row[metric] = value if value is not None else 'N/A'
            data.append(row)
        
        df = pd.DataFrame(data)
        
        logger.info("Comparison report generated")
        return df
    
    def save_comparison_report(self, filename: str = 'model_comparison.csv'):
        """
        Save comparison report to CSV.
        
        Args:
            filename: Output filename
        """
        df = self.generate_comparison_report()
        filepath = f"{config.RESULTS_DIR}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Comparison report saved to {filepath}")


if __name__ == "__main__":
    # Test evaluation
    logger.info("Testing model evaluation...")
    
    # Create dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_scores = np.random.random(1000)
    
    # Evaluate
    evaluator = ModelEvaluator("Test Model")
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Generate report
    report = evaluator.generate_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

