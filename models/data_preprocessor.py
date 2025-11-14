"""Data preprocessing pipeline for CN21 dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import os


def download_unsw_nb15_dataset():
    """Download UNSW-NB15 dataset from Kaggle if not present."""
    external_dir = Path('dataset/external')
    train_file = external_dir / 'UNSW_NB15_training-set.parquet'
    test_file = external_dir / 'UNSW_NB15_testing-set.parquet'
    
    if train_file.exists() and test_file.exists():
        print("UNSW-NB15 dataset already downloaded")
        return True
    
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_config.exists():
        print("\n⚠️  Kaggle credentials not found")
        print("To download UNSW-NB15 dataset:")
        print("1. Get API key from https://www.kaggle.com/settings")
        print("2. Place kaggle.json in ~/.kaggle/")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("\nDownloading UNSW-NB15 dataset from Kaggle...")
    external_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'dhoogla/unswnb15', 
             '-p', str(external_dir), '--unzip'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ UNSW-NB15 dataset downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}")
        print("\nManual download:")
        print("  kaggle datasets download -d dhoogla/unswnb15 -p dataset/external --unzip")
        return False
    except FileNotFoundError:
        print("⚠️  Kaggle CLI not installed")
        print("Install: pip install kaggle")
        return False


def parse_flow_file(filepath: Path) -> Optional[np.ndarray]:
    """Parse CN21 flow file (timestamp, packet_size pairs)."""
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def extract_cn21_features(data: np.ndarray) -> Optional[Dict]:
    """Extract 15 traffic features from raw CN21 data."""
    if data is None or len(data) == 0:
        return None
    
    timestamps = data[:, 0]
    packet_sizes = data[:, 1]
    
    if len(timestamps) > 1:
        iats = np.diff(timestamps)
    else:
        iats = np.array([0])
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    total_packets = len(packet_sizes)
    total_bytes = np.sum(packet_sizes)
    avg_packet_size = np.mean(packet_sizes)
    std_packet_size = np.std(packet_sizes)
    min_packet_size = np.min(packet_sizes)
    max_packet_size = np.max(packet_sizes)
    avg_iat = np.mean(iats) if len(iats) > 0 else 0
    std_iat = np.std(iats) if len(iats) > 0 else 0
    cv_iat = std_iat / avg_iat if avg_iat > 0 else 0
    cv_iat = np.nan_to_num(cv_iat, nan=0.0, posinf=0.0, neginf=0.0)
    packet_rate = total_packets / duration if duration > 0 else 0
    byte_rate = total_bytes / duration if duration > 0 else 0
    small_packet_ratio = np.sum(packet_sizes < 100) / total_packets
    large_packet_ratio = np.sum(packet_sizes > 1000) / total_packets
    burst_ratio = np.sum(iats < 0.1) / len(iats) if len(iats) > 0 else 0
    
    return {
        'duration': duration,
        'total_packets': total_packets,
        'total_bytes': total_bytes,
        'avg_packet_size': avg_packet_size,
        'std_packet_size': std_packet_size,
        'min_packet_size': min_packet_size,
        'max_packet_size': max_packet_size,
        'avg_iat': avg_iat,
        'std_iat': std_iat,
        'cv_iat': cv_iat,
        'burst_ratio': burst_ratio,
        'small_packet_ratio': small_packet_ratio,
        'large_packet_ratio': large_packet_ratio,
        'packet_rate': packet_rate,
        'byte_rate': byte_rate
    }


def load_cn21_dataset(dataset_root: Path, include_metadata: bool = True) -> pd.DataFrame:
    """Load all CN21 files and extract features."""
    benign_categories = ['Youtube', 'Skype', 'Office']
    malicious_categories = ['Bitcoin', 'Monero', 'Bytecoin']
    all_features = []
    for category in benign_categories:
        category_path = dataset_root / category
        if not category_path.exists():
            print(f"Warning: Category path not found: {category_path}")
            continue
            
        for file_path in category_path.glob('*.txt'):
            data = parse_flow_file(file_path)
            if data is None or len(data) == 0:
                continue
                
            features = extract_cn21_features(data)
            if features:
                features['label'] = 0
                if include_metadata:
                    parts = file_path.stem.split('_')
                    features['category'] = category
                    features['flow_direction'] = parts[-2] if len(parts) > 2 else 'unknown'
                    features['vpn_type'] = parts[-1] if len(parts) > 1 else 'unknown'
                    features['filename'] = file_path.name
                    
                all_features.append(features)
    for category in malicious_categories:
        category_path = dataset_root / category
        if not category_path.exists():
            print(f"Warning: Category path not found: {category_path}")
            continue
        for subdir in ['Miner', 'Full node']:
            subdir_path = category_path / subdir
            if not subdir_path.exists():
                continue
                
            for file_path in subdir_path.glob('*.txt'):
                data = parse_flow_file(file_path)
                if data is None or len(data) == 0:
                    continue
                    
                features = extract_cn21_features(data)
                if features:
                    features['label'] = 1
                    if include_metadata:
                        parts = file_path.stem.split('_')
                        features['category'] = category
                        features['subcategory'] = subdir
                        features['flow_direction'] = parts[-2] if len(parts) > 2 else 'unknown'
                        features['vpn_type'] = parts[-1] if len(parts) > 1 else 'unknown'
                        features['filename'] = file_path.name
                        
                    all_features.append(features)
    
    df = pd.DataFrame(all_features)
    print(f"Loaded {len(df)} flows: {(df['label']==0).sum()} benign, {(df['label']==1).sum()} malicious")
    
    return df


def get_feature_columns() -> List[str]:
    """Get feature column names (excluding label and metadata)."""
    return [
        'duration', 'total_packets', 'total_bytes', 'avg_packet_size', 
        'std_packet_size', 'min_packet_size', 'max_packet_size',
        'avg_iat', 'std_iat', 'cv_iat', 'burst_ratio', 
        'small_packet_ratio', 'large_packet_ratio', 'packet_rate', 'byte_rate'
    ]


def merge_with_captured_flows(cn21_df: pd.DataFrame, captured_flows_path: Path) -> pd.DataFrame:
    """Merge CN21 dataset with captured flows."""
    if not captured_flows_path.exists():
        return cn21_df
    
    captured_df = pd.read_csv(captured_flows_path)
    common_cols = list(set(cn21_df.columns) & set(captured_df.columns))
    merged_df = pd.concat([
        cn21_df[common_cols],
        captured_df[common_cols]
    ], ignore_index=True)
    
    print(f"Merged dataset: {len(merged_df)} total flows")
    return merged_df


if __name__ == "__main__":
    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Download UNSW-NB15 if needed
    print("\n[1/2] Checking UNSW-NB15 dataset...")
    download_unsw_nb15_dataset()
    
    # Step 2: Process CN21 dataset
    print("\n[2/2] Processing CN21 dataset...")
    dataset_root = Path(__file__).parent.parent / 'dataset'
    cn21_df = load_cn21_dataset(dataset_root)
    output_path = dataset_root / 'cn21_processed.csv'
    cn21_df.to_csv(output_path, index=False)
    print(f"Loaded {len(cn21_df)} flows: {(cn21_df['label']==0).sum()} benign, {(cn21_df['label']==1).sum()} malicious")
    print(f"Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete")
    print("=" * 70)
    print("\nNext steps:")
    print("  python3 models/augment_dataset.py")
    print("  python3 train_models.py --augmented")

