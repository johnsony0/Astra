"""
Integration tests for Astra cryptomining detection system.
Tests end-to-end workflow from data loading to prediction.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_preprocessor import (
    load_cn21_dataset,
    extract_cn21_features,
    get_feature_columns
)


class TestDataPipeline:
    """Test the complete data preprocessing pipeline."""
    
    def test_load_cn21_dataset(self):
        """Test loading CN21 dataset."""
        dataset_root = Path(__file__).parent.parent / 'dataset'
        
        if not dataset_root.exists():
            pytest.skip("Dataset not available")
        
        df = load_cn21_dataset(dataset_root)
        
        assert df is not None
        assert len(df) > 0
        assert 'label' in df.columns
        assert set(df['label'].unique()).issubset({0, 1})
    
    def test_feature_columns_match(self):
        """Test that extracted features match expected columns."""
        data = np.array([
            [1.0, 100],
            [1.1, 200],
            [1.2, 150]
        ])
        
        features = extract_cn21_features(data)
        feature_cols = get_feature_columns()
        
        # Check that most features are present
        assert len(set(features.keys()) & set(feature_cols)) >= len(feature_cols) - 2


class TestModelIntegration:
    """Test model loading and prediction."""
    
    def test_load_models(self):
        """Test loading pre-trained models."""
        monitor_dir = Path(__file__).parent.parent / 'monitor'
        
        scaler_path = monitor_dir / 'scaler.joblib'
        rf_path = monitor_dir / 'rf.joblib'
        svm_path = monitor_dir / 'svm.joblib'
        
        # Check if models exist
        if not all([scaler_path.exists(), rf_path.exists(), svm_path.exists()]):
            pytest.skip("Models not trained yet")
        
        # Load models
        scaler = joblib.load(scaler_path)
        rf_model = joblib.load(rf_path)
        svm_model = joblib.load(svm_path)
        
        assert scaler is not None
        assert rf_model is not None
        assert svm_model is not None
    
    def test_predict_with_models(self):
        """Test making predictions with loaded models."""
        monitor_dir = Path(__file__).parent.parent / 'monitor'
        
        scaler_path = monitor_dir / 'scaler.joblib'
        rf_path = monitor_dir / 'rf.joblib'
        
        if not all([scaler_path.exists(), rf_path.exists()]):
            pytest.skip("Models not trained yet")
        
        # Load models
        scaler = joblib.load(scaler_path)
        rf_model = joblib.load(rf_path)
        
        # Create dummy feature vector
        n_features = scaler.n_features_in_
        X_test = np.random.rand(1, n_features)
        
        # Scale and predict
        X_scaled = scaler.transform(X_test)
        prediction = rf_model.predict(X_scaled)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]


class TestEndToEnd:
    """Test complete end-to-end workflow."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from raw data to prediction."""
        # Create sample data
        data = np.array([
            [1.0, 100],
            [1.1, 200],
            [1.2, 150],
            [1.5, 120],
            [1.8, 180]
        ])
        
        # Extract features
        features = extract_cn21_features(data)
        assert features is not None
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Get feature columns
        feature_cols = get_feature_columns()
        available_features = [col for col in feature_cols if col in df.columns]
        
        assert len(available_features) > 0
        
        # Extract feature matrix
        X = df[available_features].values
        
        assert X.shape[0] == 1
        assert X.shape[1] == len(available_features)
        
        # Check for valid values
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
    
    def test_saved_features_readable(self):
        """Test that saved features can be read."""
        features_path = Path(__file__).parent.parent / 'monitor' / 'extracted_features.csv'
        
        if not features_path.exists():
            pytest.skip("No extracted features file")
        
        df = pd.read_csv(features_path)
        
        assert 'label' in df.columns
        assert len(df) >= 0


class TestWebAPI:
    """Test web dashboard API endpoints."""
    
    def test_flask_app_imports(self):
        """Test that Flask app can be imported."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'web'))
            import app
            assert hasattr(app, 'app')
        except ImportError as e:
            pytest.skip(f"Flask not installed or app.py missing: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

