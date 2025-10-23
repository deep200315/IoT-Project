"""
Exploratory Data Analysis and Visualization Module
Creates comprehensive visualizations for understanding the sensor data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


class EDAVisualizer:
    """
    Creates visualizations for exploratory data analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer.
        
        Args:
            df: Input DataFrame
        """
        self.df = df
        self.sensor_cols = [col for col in df.columns if col in config.SENSORS.keys()]
        logger.info(f"Initialized EDAVisualizer with {len(self.sensor_cols)} sensors")
    
    def plot_time_series(self, save_path: str = None):
        """
        Plot time series for all sensors.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating time series plot...")
        
        fig, axes = plt.subplots(len(self.sensor_cols), 1, figsize=(15, 3*len(self.sensor_cols)))
        
        if len(self.sensor_cols) == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(self.sensor_cols):
            ax = axes[idx]
            
            # Plot normal data
            normal_mask = self.df['anomaly'] == 0
            ax.plot(self.df.index[normal_mask], self.df.loc[normal_mask, sensor], 
                   label='Normal', alpha=0.7, linewidth=0.5)
            
            # Plot anomalies
            anomaly_mask = self.df['anomaly'] == 1
            ax.scatter(self.df.index[anomaly_mask], self.df.loc[anomaly_mask, sensor],
                      color='red', label='Anomaly', s=10, alpha=0.6)
            
            ax.set_title(f'{sensor.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{sensor} ({config.SENSORS[sensor]["unit"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_distributions(self, save_path: str = None):
        """
        Plot distributions for all sensors.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating distribution plots...")
        
        fig, axes = plt.subplots(2, len(self.sensor_cols), figsize=(5*len(self.sensor_cols), 10))
        
        if len(self.sensor_cols) == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, sensor in enumerate(self.sensor_cols):
            # Histogram
            ax1 = axes[0, idx]
            normal_data = self.df[self.df['anomaly'] == 0][sensor]
            anomaly_data = self.df[self.df['anomaly'] == 1][sensor]
            
            ax1.hist(normal_data, bins=50, alpha=0.7, label='Normal', density=True)
            ax1.hist(anomaly_data, bins=50, alpha=0.7, label='Anomaly', density=True)
            ax1.set_title(f'{sensor.replace("_", " ").title()} Distribution', fontweight='bold')
            ax1.set_xlabel(f'{sensor} ({config.SENSORS[sensor]["unit"]})')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2 = axes[1, idx]
            data_to_plot = [normal_data, anomaly_data]
            bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax2.set_title(f'{sensor.replace("_", " ").title()} Box Plot', fontweight='bold')
            ax2.set_ylabel(f'{sensor} ({config.SENSORS[sensor]["unit"]})')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_matrix(self, save_path: str = None):
        """
        Plot correlation matrix for sensors.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating correlation matrix...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Normal data correlation
        normal_data = self.df[self.df['anomaly'] == 0][self.sensor_cols]
        corr_normal = normal_data.corr()
        
        sns.heatmap(corr_normal, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Correlation Matrix - Normal Data', fontsize=14, fontweight='bold')
        
        # Anomaly data correlation
        anomaly_data = self.df[self.df['anomaly'] == 1][self.sensor_cols]
        if len(anomaly_data) > 1:
            corr_anomaly = anomaly_data.corr()
            sns.heatmap(corr_anomaly, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
            axes[1].set_title('Correlation Matrix - Anomaly Data', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_statistical_summary(self, save_path: str = None):
        """
        Create statistical summary visualization.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating statistical summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Missing values
        ax1 = axes[0, 0]
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_counts.plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Missing Values by Column', fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Column')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax1.set_title('Missing Values by Column', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Anomaly distribution over time
        ax2 = axes[0, 1]
        # Resample to daily frequency for better visualization
        daily_anomalies = self.df['anomaly'].resample('D').sum()
        daily_anomalies.plot(ax=ax2, color='red', linewidth=2)
        ax2.set_title('Anomaly Count Over Time (Daily)', fontweight='bold')
        ax2.set_ylabel('Number of Anomalies')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sensor statistics comparison
        ax3 = axes[1, 0]
        stats_data = []
        for sensor in self.sensor_cols:
            normal_mean = self.df[self.df['anomaly'] == 0][sensor].mean()
            anomaly_mean = self.df[self.df['anomaly'] == 1][sensor].mean()
            stats_data.append({
                'Sensor': sensor,
                'Normal Mean': normal_mean,
                'Anomaly Mean': anomaly_mean
            })
        
        stats_df = pd.DataFrame(stats_data)
        x = np.arange(len(self.sensor_cols))
        width = 0.35
        
        ax3.bar(x - width/2, stats_df['Normal Mean'], width, label='Normal', alpha=0.8)
        ax3.bar(x + width/2, stats_df['Anomaly Mean'], width, label='Anomaly', alpha=0.8)
        ax3.set_title('Mean Values: Normal vs Anomaly', fontweight='bold')
        ax3.set_ylabel('Mean Value')
        ax3.set_xlabel('Sensor')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace('_', '\n') for s in self.sensor_cols])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Anomaly percentage
        ax4 = axes[1, 1]
        anomaly_counts = self.df['anomaly'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        ax4.pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax4.set_title('Normal vs Anomaly Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistical summary saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_statistical_report(self) -> Dict:
        """
        Generate comprehensive statistical report.
        
        Returns:
            Dictionary with statistical information
        """
        logger.info("Generating statistical report...")
        
        report = {
            'dataset_shape': self.df.shape,
            'date_range': {
                'start': str(self.df.index.min()),
                'end': str(self.df.index.max()),
                'duration_days': (self.df.index.max() - self.df.index.min()).days
            },
            'missing_values': self.df.isnull().sum().to_dict(),
            'anomaly_stats': {
                'total_anomalies': int(self.df['anomaly'].sum()),
                'anomaly_percentage': float(self.df['anomaly'].mean() * 100),
                'normal_count': int((self.df['anomaly'] == 0).sum())
            },
            'sensor_statistics': {}
        }
        
        for sensor in self.sensor_cols:
            normal_data = self.df[self.df['anomaly'] == 0][sensor]
            anomaly_data = self.df[self.df['anomaly'] == 1][sensor]
            
            report['sensor_statistics'][sensor] = {
                'normal': {
                    'mean': float(normal_data.mean()),
                    'std': float(normal_data.std()),
                    'min': float(normal_data.min()),
                    'max': float(normal_data.max()),
                    'median': float(normal_data.median())
                },
                'anomaly': {
                    'mean': float(anomaly_data.mean()),
                    'std': float(anomaly_data.std()),
                    'min': float(anomaly_data.min()),
                    'max': float(anomaly_data.max()),
                    'median': float(anomaly_data.median())
                }
            }
        
        return report


if __name__ == "__main__":
    # Test visualization
    logger.info("Testing EDA visualization...")
    
    # Load data
    df = pd.read_csv(f"{config.DATA_DIR}/sensor_data.csv", index_col=0, parse_dates=True)
    
    # Create visualizer
    visualizer = EDAVisualizer(df)
    
    # Generate plots
    visualizer.plot_time_series(f"{config.PLOTS_DIR}/01_time_series.png")
    visualizer.plot_distributions(f"{config.PLOTS_DIR}/02_distributions.png")
    visualizer.plot_correlation_matrix(f"{config.PLOTS_DIR}/03_correlation_matrix.png")
    visualizer.plot_statistical_summary(f"{config.PLOTS_DIR}/04_statistical_summary.png")
    
    # Generate report
    report = visualizer.generate_statistical_report()
    logger.info("Statistical Report:")
    for key, value in report.items():
        logger.info(f"{key}: {value}")

