"""
Complete Pipeline Runner
Executes the entire anomaly detection pipeline from data generation to model evaluation.
"""

import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
import config
from src.data_generator import SensorDataGenerator
from src.data_preprocessing import DataPreprocessor, FeatureEngineer
from src.eda_visualization import EDAVisualizer
from src.model_isolation_forest import IsolationForestDetector, plot_anomaly_scores
from src.model_lstm_autoencoder import LSTMAutoencoder
from src.model_evaluation import ModelEvaluator, ModelComparator
from sklearn.model_selection import train_test_split


def main():
    """Run the complete pipeline."""
    
    print("\n" + "="*80)
    print("TIME SERIES ANOMALY DETECTION FOR IOT SENSORS")
    print("Complete Pipeline Execution")
    print("="*80 + "\n")
    
    # =========================================================================
    # STEP 1: DATA GENERATION
    # =========================================================================
    logger.info("STEP 1: Generating synthetic sensor data...")
    generator = SensorDataGenerator(num_samples=config.NUM_SAMPLES)
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    print(f"\n✓ Data generated: {df.shape}")
    print(f"  - Sensors: {len([c for c in df.columns if c != 'anomaly'])}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Anomalies: {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.2f}%)")
    
    # =========================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    logger.info("\nSTEP 2: Performing exploratory data analysis...")
    visualizer = EDAVisualizer(df)
    
    visualizer.plot_time_series(f"{config.PLOTS_DIR}/01_time_series.png")
    visualizer.plot_distributions(f"{config.PLOTS_DIR}/02_distributions.png")
    visualizer.plot_correlation_matrix(f"{config.PLOTS_DIR}/03_correlation_matrix.png")
    visualizer.plot_statistical_summary(f"{config.PLOTS_DIR}/04_statistical_summary.png")
    
    report = visualizer.generate_statistical_report()
    print(f"\n✓ EDA complete: 4 visualizations generated")
    
    # =========================================================================
    # STEP 3: DATA PREPROCESSING
    # =========================================================================
    logger.info("\nSTEP 3: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    print(f"\n✓ Preprocessing complete")
    print(f"  - Missing values handled: {df.isnull().sum().sum()} → 0")
    
    # =========================================================================
    # STEP 4: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\nSTEP 4: Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    X_scaled, feature_names = engineer.scale_features(df_features, fit=True)
    y = df_features['anomaly'].values
    engineer.save_scaler()
    
    print(f"\n✓ Feature engineering complete")
    print(f"  - Original features: 4")
    print(f"  - Engineered features: {len(feature_names)}")
    print(f"  - Feature increase: {len(feature_names) - 4}x")
    
    # =========================================================================
    # STEP 5: TRAIN-TEST SPLIT
    # =========================================================================
    logger.info("\nSTEP 5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    timestamps_train, timestamps_test = train_test_split(
        df_features.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    print(f"\n✓ Data split complete")
    print(f"  - Training: {len(X_train)} samples ({y_train.mean()*100:.2f}% anomalies)")
    print(f"  - Testing: {len(X_test)} samples ({y_test.mean()*100:.2f}% anomalies)")
    
    # =========================================================================
    # STEP 6: TRAIN ISOLATION FOREST
    # =========================================================================
    logger.info("\nSTEP 6: Training Isolation Forest...")
    if_detector = IsolationForestDetector(
        contamination=config.ISOLATION_FOREST_PARAMS['contamination'],
        n_estimators=config.ISOLATION_FOREST_PARAMS['n_estimators']
    )
    if_detector.fit(X_train)
    if_detector.save_model()
    
    y_pred_if = if_detector.predict(X_test)
    y_scores_if = if_detector.predict_proba(X_test)
    
    print(f"\n✓ Isolation Forest trained")
    print(f"  - Predicted anomalies: {y_pred_if.sum()} ({y_pred_if.mean()*100:.2f}%)")
    
    # Plot anomaly scores
    plot_anomaly_scores(
        y_scores_if, y_test, 'Isolation Forest',
        save_path=f"{config.PLOTS_DIR}/05_if_anomaly_scores.png"
    )
    
    # =========================================================================
    # STEP 7: TRAIN LSTM AUTOENCODER
    # =========================================================================
    logger.info("\nSTEP 7: Training LSTM Autoencoder...")
    logger.info("This may take several minutes...")
    
    lstm_ae = LSTMAutoencoder(
        sequence_length=config.LSTM_PARAMS['sequence_length'],
        encoding_dim=config.LSTM_PARAMS['encoding_dim'],
        lstm_units=config.LSTM_PARAMS['lstm_units']
    )
    
    lstm_ae.fit(
        X_train, y_train,
        epochs=config.LSTM_PARAMS['epochs'],
        batch_size=config.LSTM_PARAMS['batch_size'],
        validation_split=config.LSTM_PARAMS['validation_split']
    )
    lstm_ae.save_model()
    
    y_pred_lstm = lstm_ae.predict(X_test)
    y_scores_lstm = lstm_ae.predict_proba(X_test)
    
    print(f"\n✓ LSTM Autoencoder trained")
    print(f"  - Predicted anomalies: {y_pred_lstm.sum()} ({y_pred_lstm.mean()*100:.2f}%)")
    
    # Plot training history and reconstructions
    lstm_ae.plot_training_history(f"{config.PLOTS_DIR}/06_lstm_training_history.png")
    lstm_ae.plot_reconstruction_examples(
        X_test, n_examples=3,
        save_path=f"{config.PLOTS_DIR}/07_lstm_reconstructions.png"
    )
    
    # =========================================================================
    # STEP 8: EVALUATE ISOLATION FOREST
    # =========================================================================
    logger.info("\nSTEP 8: Evaluating Isolation Forest...")
    evaluator_if = ModelEvaluator("Isolation Forest")
    metrics_if = evaluator_if.calculate_metrics(y_test, y_pred_if, y_scores_if)
    evaluator_if.save_metrics()
    
    print(f"\n✓ Isolation Forest Metrics:")
    print(f"  - Accuracy: {metrics_if['accuracy']:.4f}")
    print(f"  - Precision: {metrics_if['precision']:.4f}")
    print(f"  - Recall: {metrics_if['recall']:.4f}")
    print(f"  - F1-Score: {metrics_if['f1_score']:.4f}")
    if metrics_if.get('roc_auc'):
        print(f"  - ROC-AUC: {metrics_if['roc_auc']:.4f}")
    
    # Generate plots
    evaluator_if.plot_confusion_matrix(
        y_test, y_pred_if,
        save_path=f"{config.PLOTS_DIR}/08_if_confusion_matrix.png"
    )
    evaluator_if.plot_roc_curve(
        y_test, y_scores_if,
        save_path=f"{config.PLOTS_DIR}/09_if_roc_curve.png"
    )
    evaluator_if.plot_precision_recall_curve(
        y_test, y_scores_if,
        save_path=f"{config.PLOTS_DIR}/10_if_pr_curve.png"
    )
    
    # Timeline
    temp_test = df_features.loc[timestamps_test, 'temperature'].values
    evaluator_if.plot_anomaly_timeline(
        timestamps_test, y_test, y_pred_if,
        sensor_data=temp_test, sensor_name='Temperature',
        save_path=f"{config.PLOTS_DIR}/11_if_timeline.png"
    )
    
    # =========================================================================
    # STEP 9: EVALUATE LSTM AUTOENCODER
    # =========================================================================
    logger.info("\nSTEP 9: Evaluating LSTM Autoencoder...")
    evaluator_lstm = ModelEvaluator("LSTM Autoencoder")
    metrics_lstm = evaluator_lstm.calculate_metrics(y_test, y_pred_lstm, y_scores_lstm)
    evaluator_lstm.save_metrics()
    
    print(f"\n✓ LSTM Autoencoder Metrics:")
    print(f"  - Accuracy: {metrics_lstm['accuracy']:.4f}")
    print(f"  - Precision: {metrics_lstm['precision']:.4f}")
    print(f"  - Recall: {metrics_lstm['recall']:.4f}")
    print(f"  - F1-Score: {metrics_lstm['f1_score']:.4f}")
    if metrics_lstm.get('roc_auc'):
        print(f"  - ROC-AUC: {metrics_lstm['roc_auc']:.4f}")
    
    # Generate plots
    evaluator_lstm.plot_confusion_matrix(
        y_test, y_pred_lstm,
        save_path=f"{config.PLOTS_DIR}/12_lstm_confusion_matrix.png"
    )
    evaluator_lstm.plot_roc_curve(
        y_test, y_scores_lstm,
        save_path=f"{config.PLOTS_DIR}/13_lstm_roc_curve.png"
    )
    evaluator_lstm.plot_precision_recall_curve(
        y_test, y_scores_lstm,
        save_path=f"{config.PLOTS_DIR}/14_lstm_pr_curve.png"
    )
    
    # Timeline
    evaluator_lstm.plot_anomaly_timeline(
        timestamps_test, y_test, y_pred_lstm,
        sensor_data=temp_test, sensor_name='Temperature',
        save_path=f"{config.PLOTS_DIR}/15_lstm_timeline.png"
    )
    
    # =========================================================================
    # STEP 10: MODEL COMPARISON
    # =========================================================================
    logger.info("\nSTEP 10: Comparing models...")
    comparator = ModelComparator()
    comparator.add_model_results("Isolation Forest", metrics_if)
    comparator.add_model_results("LSTM Autoencoder", metrics_lstm)
    
    comparison_df = comparator.generate_comparison_report()
    comparator.save_comparison_report()
    
    print(f"\n✓ Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Generate comparison plots
    comparator.plot_metrics_comparison(
        save_path=f"{config.PLOTS_DIR}/16_model_comparison.png"
    )
    comparator.plot_confusion_matrices_comparison(
        y_test,
        {"Isolation Forest": y_pred_if, "LSTM Autoencoder": y_pred_lstm},
        save_path=f"{config.PLOTS_DIR}/17_confusion_matrices_comparison.png"
    )
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Generated Outputs:")
    print(f"  - Data: {config.DATA_DIR}/sensor_data.csv")
    print(f"  - Models: {config.MODELS_DIR}/")
    print(f"  - Plots: {config.PLOTS_DIR}/ (17 visualizations)")
    print(f"  - Metrics: {config.RESULTS_DIR}/ (JSON and CSV files)")
    print(f"\n📈 Performance Summary:")
    print(f"  Isolation Forest - F1: {metrics_if['f1_score']:.4f}, ROC-AUC: {metrics_if.get('roc_auc', 0):.4f}")
    print(f"  LSTM Autoencoder - F1: {metrics_lstm['f1_score']:.4f}, ROC-AUC: {metrics_lstm.get('roc_auc', 0):.4f}")
    print(f"\n🎉 All tasks completed successfully!")
    print("\nNext steps:")
    print("  1. Review visualizations in results/plots/")
    print("  2. Check metrics in results/*.json")
    print("  3. Read SUMMARY.md for detailed insights")
    print("  4. Open anomaly_detection_notebook.ipynb for interactive exploration")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

