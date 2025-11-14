"""
Unit tests for data preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_preprocessor import (
    parse_flow_file,
    extract_cn21_features,
    get_feature_columns
)


class TestParseFlowFile:
    """Tests for parse_flow_file function."""
    
    def test_parse_valid_file(self, tmp_path):
        """Test parsing a valid flow file."""
        # Create a temporary test file
        test_file = tmp_path / "test_flow.txt"
        data = "1.0 100\n2.0 200\n3.0 150\n"
        test_file.write_text(data)
        
        result = parse_flow_file(test_file)
        
        assert result is not None
        assert result.shape == (3, 2)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 100
    
    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = parse_flow_file(test_file)
        # Empty file should return None or empty array
        assert result is None or len(result) == 0
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = parse_flow_file(Path("/nonexistent/file.txt"))
        assert result is None


class TestExtractCN21Features:
    """Tests for extract_cn21_features function."""
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        # Create sample data: timestamps and packet sizes
        data = np.array([
            [1.0, 100],
            [1.1, 200],
            [1.3, 150],
            [1.5, 120]
        ])
        
        features = extract_cn21_features(data)
        
        assert features is not None
        assert 'duration' in features
        assert 'total_packets' in features
        assert 'total_bytes' in features
        assert 'avg_packet_size' in features
        
        # Validate values
        assert features['duration'] == pytest.approx(0.5, rel=1e-5)
        assert features['total_packets'] == 4
        assert features['total_bytes'] == 570
        assert features['avg_packet_size'] == pytest.approx(142.5, rel=1e-5)
    
    def test_extract_iat_features(self):
        """Test inter-arrival time feature extraction."""
        data = np.array([
            [1.0, 100],
            [1.1, 200],
            [1.2, 150]
        ])
        
        features = extract_cn21_features(data)
        
        assert 'avg_iat' in features
        assert 'std_iat' in features
        assert 'cv_iat' in features
        
        # IAT should be [0.1, 0.1]
        assert features['avg_iat'] == pytest.approx(0.1, rel=1e-5)
    
    def test_extract_burst_ratio(self):
        """Test burst ratio calculation."""
        # Create data with some bursts (IAT < 0.1s)
        data = np.array([
            [1.0, 100],
            [1.05, 200],  # Burst (IAT=0.05)
            [1.08, 150],  # Burst (IAT=0.03)
            [1.3, 120]    # Not burst (IAT=0.22)
        ])
        
        features = extract_cn21_features(data)
        
        assert 'burst_ratio' in features
        # 2 out of 3 IATs are < 0.1
        assert features['burst_ratio'] == pytest.approx(2/3, rel=1e-5)
    
    def test_extract_packet_ratios(self):
        """Test small/large packet ratio calculation."""
        data = np.array([
            [1.0, 50],     # Small
            [1.1, 200],    # Medium
            [1.2, 1500],   # Large
            [1.3, 80]      # Small
        ])
        
        features = extract_cn21_features(data)
        
        assert 'small_packet_ratio' in features
        assert 'large_packet_ratio' in features
        
        # 2 small (< 100), 1 large (> 1000) out of 4
        assert features['small_packet_ratio'] == pytest.approx(0.5, rel=1e-5)
        assert features['large_packet_ratio'] == pytest.approx(0.25, rel=1e-5)
    
    def test_extract_rates(self):
        """Test packet and byte rate calculation."""
        data = np.array([
            [1.0, 100],
            [2.0, 200],
            [3.0, 150]
        ])
        
        features = extract_cn21_features(data)
        
        assert 'packet_rate' in features
        assert 'byte_rate' in features
        
        # Duration = 2.0s, 3 packets, 450 bytes
        assert features['packet_rate'] == pytest.approx(1.5, rel=1e-5)
        assert features['byte_rate'] == pytest.approx(225.0, rel=1e-5)
    
    def test_extract_single_packet(self):
        """Test feature extraction with single packet."""
        data = np.array([[1.0, 100]])
        
        features = extract_cn21_features(data)
        
        assert features is not None
        assert features['total_packets'] == 1
        assert features['duration'] == 0
    
    def test_extract_empty_data(self):
        """Test feature extraction with empty data."""
        result = extract_cn21_features(np.array([]))
        assert result is None
        
        result = extract_cn21_features(None)
        assert result is None
    
    def test_cv_iat_zero_division(self):
        """Test that CV of IAT handles zero division gracefully."""
        # All packets at same time
        data = np.array([
            [1.0, 100],
            [1.0, 200]
        ])
        
        features = extract_cn21_features(data)
        
        # Should not raise error and should be 0 or finite
        assert 'cv_iat' in features
        assert np.isfinite(features['cv_iat'])


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0
    
    def test_no_label_in_features(self):
        """Test that label is not in feature columns."""
        cols = get_feature_columns()
        assert 'label' not in cols
        assert 'category' not in cols
        assert 'filename' not in cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

