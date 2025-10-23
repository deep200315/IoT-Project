"""
LSTM Autoencoder for Anomaly Detection
Implements deep learning-based anomaly detection using LSTM Autoencoder.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


class LSTMAutoencoder:
    """
    LSTM-based Autoencoder for time series anomaly detection.
    
    Intuition:
    An autoencoder learns to compress (encode) and reconstruct (decode) normal data.
    When trained on normal data, it becomes good at reconstructing normal patterns
    but struggles with anomalies, resulting in high reconstruction errors for anomalies.
    
    LSTM (Long Short-Term Memory) networks are particularly suited for time series
    because they can capture temporal dependencies and patterns over time.
    
    The algorithm is particularly effective for:
    - Sequential/temporal data
    - Complex patterns and dependencies
    - Learning normal behavior automatically
    - Detecting subtle anomalies
    """
    
    def __init__(self, 
                 sequence_length: int = config.LSTM_PARAMS['sequence_length'],
                 encoding_dim: int = config.LSTM_PARAMS['encoding_dim'],
                 lstm_units: list = None):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            sequence_length: Length of input sequences
            encoding_dim: Dimension of the encoded representation
            lstm_units: List of LSTM units for each layer
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.lstm_units = lstm_units or config.LSTM_PARAMS['lstm_units']
        self.model = None
        self.history = None
        self.threshold = None
        self.n_features = None
        
        logger.info(f"Initialized LSTMAutoencoder with sequence_length={sequence_length}, "
                   f"encoding_dim={encoding_dim}, lstm_units={self.lstm_units}")
    
    def build_model(self, n_features: int):
        """
        Build the LSTM Autoencoder architecture.
        
        Args:
            n_features: Number of input features
        """
        self.n_features = n_features
        
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, n_features))
        
        # Encoder
        encoded = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            encoded = layers.LSTM(
                units, 
                activation='tanh',
                return_sequences=return_sequences,
                name=f'encoder_lstm_{i+1}'
            )(encoded)
            encoded = layers.Dropout(0.2, name=f'encoder_dropout_{i+1}')(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Repeat vector for decoder
        decoded = layers.RepeatVector(self.sequence_length, name='repeat_vector')(encoded)
        
        # Decoder
        for i, units in enumerate(reversed(self.lstm_units)):
            decoded = layers.LSTM(
                units,
                activation='tanh',
                return_sequences=True,
                name=f'decoder_lstm_{i+1}'
            )(decoded)
            decoded = layers.Dropout(0.2, name=f'decoder_dropout_{i+1}')(decoded)
        
        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(n_features),
            name='output'
        )(decoded)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LSTM_PARAMS['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model architecture built")
        logger.info(f"Total parameters: {self.model.count_params():,}")
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Input data
            y: Labels (optional)
            
        Returns:
            Tuple of (sequences, labels if provided)
        """
        sequences = []
        labels = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                # Label is 1 if any point in sequence is anomaly
                labels.append(int(np.any(y[i:i + self.sequence_length])))
        
        sequences = np.array(sequences)
        
        if y is not None:
            labels = np.array(labels)
            return sequences, labels
        
        return sequences
    
    def fit(self, X: np.ndarray, y: np.ndarray = None,
            validation_split: float = config.LSTM_PARAMS['validation_split'],
            epochs: int = config.LSTM_PARAMS['epochs'],
            batch_size: int = config.LSTM_PARAMS['batch_size']) -> 'LSTMAutoencoder':
        """
        Train the LSTM Autoencoder.
        
        Args:
            X: Training data
            y: Labels (for threshold calculation)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Self
        """
        logger.info(f"Preparing sequences from data with shape {X.shape}...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X.shape[1])
        
        # Create sequences
        if y is not None:
            X_seq, y_seq = self.create_sequences(X, y)
            logger.info(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
        else:
            X_seq = self.create_sequences(X)
            logger.info(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model (autoencoder learns to reconstruct input)
        logger.info(f"Training model for {epochs} epochs...")
        self.history = self.model.fit(
            X_seq, X_seq,  # Input and target are the same
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Calculate threshold based on reconstruction error
        logger.info("Calculating anomaly threshold...")
        reconstructions = self.model.predict(X_seq, verbose=0)
        reconstruction_errors = np.mean(np.abs(X_seq - reconstructions), axis=(1, 2))
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(
            reconstruction_errors, 
            config.THRESHOLD_PERCENTILE
        )
        
        logger.info(f"Training complete. Threshold set to {self.threshold:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        # Create sequences
        X_seq = self.create_sequences(X)
        
        # Get reconstruction errors
        reconstruction_errors = self.predict_proba(X)
        
        # Threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction errors (anomaly scores).
        
        Args:
            X: Data to score
            
        Returns:
            Reconstruction errors
        """
        # Create sequences
        X_seq = self.create_sequences(X)
        
        # Reconstruct
        reconstructions = self.model.predict(X_seq, verbose=0)
        
        # Calculate reconstruction error (MAE across time and features)
        reconstruction_errors = np.mean(np.abs(X_seq - reconstructions), axis=(1, 2))
        
        # Pad to match original length
        padded_errors = np.zeros(len(X))
        padded_errors[:len(reconstruction_errors)] = reconstruction_errors
        
        return padded_errors
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1 = axes[0]
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Model Loss Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2 = axes[1]
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Model MAE Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_reconstruction_examples(self, X: np.ndarray, n_examples: int = 3,
                                    save_path: str = None):
        """
        Plot reconstruction examples.
        
        Args:
            X: Data to reconstruct
            n_examples: Number of examples to plot
            save_path: Path to save the plot
        """
        # Create sequences
        X_seq = self.create_sequences(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_seq[:n_examples], verbose=0)
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(15, 4*n_examples))
        
        if n_examples == 1:
            axes = [axes]
        
        for i in range(n_examples):
            ax = axes[i]
            
            # Plot first feature only for clarity
            ax.plot(X_seq[i, :, 0], label='Original', linewidth=2)
            ax.plot(reconstructions[i, :, 0], label='Reconstructed', 
                   linewidth=2, linestyle='--')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.set_title(f'Reconstruction Example {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reconstruction examples saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def save_model(self, filename: str = 'lstm_autoencoder_model.keras'):
        """
        Save the trained model.
        
        Args:
            filename: Output filename (with .keras extension)
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        self.model.save(filepath)
        
        # Save threshold and other parameters
        import json
        params = {
            'threshold': float(self.threshold),
            'sequence_length': self.sequence_length,
            'encoding_dim': self.encoding_dim,
            'n_features': self.n_features
        }
        params_file = filepath.replace('.keras', '_params.json')
        with open(params_file, 'w') as f:
            json.dump(params, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = 'lstm_autoencoder_model.keras'):
        """
        Load a trained model.
        
        Args:
            filename: Input filename (with .keras extension)
        """
        filepath = f"{config.MODELS_DIR}/{filename}"
        self.model = keras.models.load_model(filepath)
        
        # Load parameters
        import json
        params_file = filepath.replace('.keras', '_params.json')
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        self.threshold = params['threshold']
        self.sequence_length = params['sequence_length']
        self.encoding_dim = params['encoding_dim']
        self.n_features = params['n_features']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


if __name__ == "__main__":
    # Test LSTM Autoencoder
    logger.info("Testing LSTM Autoencoder...")
    
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
    
    # Train LSTM Autoencoder
    lstm_ae = LSTMAutoencoder()
    lstm_ae.fit(X_train, y_train, epochs=10)  # Reduced epochs for testing
    
    # Plot training history
    lstm_ae.plot_training_history(f"{config.PLOTS_DIR}/lstm_training_history.png")
    
    # Plot reconstruction examples
    lstm_ae.plot_reconstruction_examples(X_test, n_examples=3,
                                        save_path=f"{config.PLOTS_DIR}/lstm_reconstructions.png")
    
    # Predict
    y_pred = lstm_ae.predict(X_test)
    scores = lstm_ae.predict_proba(X_test)
    
    logger.info(f"Predictions: {np.unique(y_pred, return_counts=True)}")
    logger.info(f"True labels: {np.unique(y_test, return_counts=True)}")
    logger.info(f"Model summary:\n{lstm_ae.get_model_summary()}")

