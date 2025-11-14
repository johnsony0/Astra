"""
Data augmentation for CN21 dataset.
Generates synthetic samples by adding realistic noise to existing flows.

Why augmentation makes sense:
1. Small dataset (54 flows) leads to overfitting → 100% accuracy
2. Real networks have natural variability (jitter, congestion, routing changes)
3. Models should generalize, not memorize specific flow patterns
4. Augmentation simulates realistic network conditions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.data_preprocessor import get_feature_columns


def select_cryptomining_like_traffic(confidence_threshold=0.6):
    """Use original CN21 model to identify cryptomining-like botnet traffic."""
    import joblib
    
    train_path = Path('dataset/external/UNSW_NB15_training-set.parquet')
    if not train_path.exists():
        return None
    
    scaler = joblib.load('models/scaler_original.joblib')
    rf = joblib.load('models/rf_original.joblib')
    svm = joblib.load('models/svm_original.joblib')
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(Path('dataset/external/UNSW_NB15_testing-set.parquet'))
    nids = pd.concat([train, test], ignore_index=True)
    
    botnet = nids[nids['attack_cat'].isin(['Generic', 'Backdoor', 'Worms'])].copy()
    benign = nids[nids['label'] == 0].copy()
    
    mapped = pd.DataFrame()
    mapped['duration'] = botnet['dur']
    mapped['total_packets'] = botnet['spkts'] + botnet['dpkts']
    mapped['total_bytes'] = botnet['sbytes'] + botnet['dbytes']
    mapped['avg_packet_size'] = (botnet['smean'] + botnet['dmean']) / 2
    mapped['std_packet_size'] = np.abs(botnet['smean'] - botnet['dmean'])
    mapped['avg_iat'] = (botnet['sinpkt'] + botnet['dinpkt']) / 2000.0
    mapped['std_iat'] = (botnet['sjit'] + botnet['djit']) / 2000.0
    mapped['cv_iat'] = mapped['std_iat'] / (mapped['avg_iat'] + 1e-6)
    mapped['packet_rate'] = botnet['rate']
    mapped['byte_rate'] = botnet['sload'] + botnet['dload']
    mapped['burst_ratio'] = botnet['sbytes'] / (botnet['sbytes'] + botnet['dbytes'] + 1)
    mapped['small_packet_ratio'] = ((botnet['smean'] < 100) | (botnet['dmean'] < 100)).astype(float) * 0.5
    mapped['large_packet_ratio'] = ((botnet['smean'] > 1000) | (botnet['dmean'] > 1000)).astype(float) * 0.3
    mapped = mapped.fillna(0).replace([np.inf, -np.inf], [0, 0]).clip(lower=0)
    
    feature_cols = ['duration', 'total_packets', 'total_bytes', 'avg_packet_size', 
                    'std_packet_size', 'avg_iat', 'std_iat', 'cv_iat', 'burst_ratio', 
                    'small_packet_ratio', 'large_packet_ratio', 'packet_rate', 'byte_rate']
    
    X = mapped[feature_cols]
    X_scaled = scaler.transform(X)
    
    rf_probas = rf.predict_proba(X_scaled)[:, 1]
    svm_probas = svm.predict_proba(X_scaled)[:, 1]
    
    both_agree = (rf_probas >= confidence_threshold) & (svm_probas >= confidence_threshold)
    selected_botnet = botnet[both_agree].copy()
    
    return {
        'cryptomining_like': selected_botnet,
        'benign': benign,
        'stats': {
            'total_botnet': len(botnet),
            'selected': both_agree.sum(),
            'rf_avg_conf': rf_probas[both_agree].mean() if both_agree.sum() > 0 else 0,
            'svm_avg_conf': svm_probas[both_agree].mean() if both_agree.sum() > 0 else 0
        }
    }


def map_nids_to_features(nids_df, label_value):
    """Map UNSW-NB15 features to CN21 feature space."""
    mapped = pd.DataFrame()
    
    mapped['duration'] = nids_df['dur']
    mapped['total_packets'] = nids_df['spkts'] + nids_df['dpkts']
    mapped['total_bytes'] = nids_df['sbytes'] + nids_df['dbytes']
    mapped['avg_packet_size'] = (nids_df['smean'] + nids_df['dmean']) / 2
    mapped['avg_iat'] = (nids_df['sinpkt'] + nids_df['dinpkt']) / 2000.0
    mapped['std_iat'] = (nids_df['sjit'] + nids_df['djit']) / 2000.0
    mapped['cv_iat'] = mapped['std_iat'] / (mapped['avg_iat'] + 1e-6)
    mapped['packet_rate'] = nids_df['rate']
    mapped['byte_rate'] = nids_df['sload'] + nids_df['dload']
    mapped['burst_ratio'] = nids_df['sbytes'] / (nids_df['sbytes'] + nids_df['dbytes'] + 1)
    mapped['small_packet_ratio'] = ((nids_df['smean'] < 100) | (nids_df['dmean'] < 100)).astype(float) * 0.5
    mapped['large_packet_ratio'] = ((nids_df['smean'] > 1000) | (nids_df['dmean'] > 1000)).astype(float) * 0.3
    
    mapped = mapped.fillna(0).replace([np.inf, -np.inf], [0, 0])
    mapped = mapped.clip(lower=0)
    mapped['label'] = label_value
    
    return mapped


def add_realistic_noise(df, n_augmented_per_sample=3, noise_level=0.1):
    """Generate augmented samples by adding Gaussian noise to simulate network variance."""
    feature_cols = get_feature_columns()
    # Remove features not in df
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    augmented_samples = []
    
    for idx, row in df.iterrows():
        # Keep original with source ID
        orig_dict = row.to_dict()
        orig_dict['source_id'] = f'cn21_orig_{idx}'
        augmented_samples.append(orig_dict)
        
        # Generate noisy versions with same source ID
        for aug_num in range(n_augmented_per_sample):
            noisy_row = row.copy()
            noisy_row['source_id'] = f'cn21_orig_{idx}'
            
            for feat in feature_cols:
                original_value = row[feat]
                
                # Add Gaussian noise proportional to value
                # Prevents negative values for strictly positive features
                noise = np.random.normal(0, abs(original_value) * noise_level)
                noisy_value = original_value + noise
                
                # Enforce constraints
                if feat in ['duration', 'total_packets', 'total_bytes', 'packet_rate', 'byte_rate']:
                    # Must be positive
                    noisy_value = max(0.001, noisy_value)
                elif feat in ['burst_ratio', 'small_packet_ratio', 'large_packet_ratio']:
                    # Must be in [0, 1]
                    noisy_value = np.clip(noisy_value, 0, 1)
                elif feat == 'cv_iat':
                    # Coefficient of variation can be large but must be positive
                    noisy_value = max(0, noisy_value)
                
                noisy_row[feat] = noisy_value
            
            augmented_samples.append(noisy_row)
    
    return pd.DataFrame(augmented_samples)


def augment_with_feature_correlation(df, n_samples=200):
    """Generate samples preserving feature correlations."""
    feature_cols = get_feature_columns()
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    # Separate by class with full rows for source tracking
    benign = df[df['label'] == 0]
    malicious = df[df['label'] == 1]
    
    augmented = []
    
    for _ in range(n_samples // 2):
        # Sample from benign distribution
        idx = np.random.randint(0, len(benign))
        base_sample = benign.iloc[idx].copy()
        
        # Add correlated noise
        for feat in feature_cols:
            noise = np.random.normal(0, benign[feat].std() * 0.15)
            base_sample[feat] += noise
            
            # Constraints
            if feat in ['duration', 'total_packets', 'total_bytes']:
                base_sample[feat] = max(0.001, base_sample[feat])
            elif 'ratio' in feat:
                base_sample[feat] = np.clip(base_sample[feat], 0, 1)
        
        sample_dict = base_sample.to_dict()
        sample_dict['label'] = 0
        augmented.append(sample_dict)
    
    for _ in range(n_samples // 2):
        # Sample from malicious distribution
        idx = np.random.randint(0, len(malicious))
        base_sample = malicious.iloc[idx].copy()
        
        for feat in feature_cols:
            noise = np.random.normal(0, malicious[feat].std() * 0.15)
            base_sample[feat] += noise
            
            if feat in ['duration', 'total_packets', 'total_bytes']:
                base_sample[feat] = max(0.001, base_sample[feat])
            elif 'ratio' in feat:
                base_sample[feat] = np.clip(base_sample[feat], 0, 1)
        
        sample_dict = base_sample.to_dict()
        sample_dict['label'] = 1
        augmented.append(sample_dict)
    
    return pd.DataFrame(augmented)


def main():
    print("=" * 70)
    print("DATA AUGMENTATION - MODEL-GUIDED SELECTION")
    print("=" * 70)
    
    dataset_path = Path('dataset/cn21_processed.csv')
    if not dataset_path.exists():
        print("Error: Run 'python3 models/data_preprocessor.py' first")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"CN21: {len(df)} flows ({(df['label']==0).sum()} benign, {(df['label']==1).sum()} malicious)")
    
    augmented_noise = add_realistic_noise(df, n_augmented_per_sample=5, noise_level=0.15)
    augmented_dist = augment_with_feature_correlation(df, n_samples=300)
    combined = pd.concat([augmented_noise, augmented_dist], ignore_index=True)
    
    print(f"Noise augmentation: {len(combined)} flows")
    
    nids_data = select_cryptomining_like_traffic(confidence_threshold=0.6)
    if nids_data:
        stats = nids_data['stats']
        print(f"\nNIDS selection (threshold=0.6, both models agree):")
        print(f"  Examined: {stats['total_botnet']} botnet flows")
        print(f"  Selected: {stats['selected']} cryptomining-like ({stats['selected']/stats['total_botnet']*100:.2f}%)")
        print(f"  RF confidence: {stats['rf_avg_conf']:.3f}, SVM confidence: {stats['svm_avg_conf']:.3f}")
        
        crypto_like = nids_data['cryptomining_like']
        benign_nids = nids_data['benign']
        
        crypto_sample = crypto_like.sample(min(len(crypto_like), 680), random_state=42)
        benign_sample = benign_nids.sample(min(1500, len(benign_nids)), random_state=42)
        
        crypto_mapped = map_nids_to_features(crypto_sample, label_value=1)
        crypto_mapped['source_id'] = [f'nids_crypto_{i}' for i in range(len(crypto_mapped))]
        
        benign_mapped = map_nids_to_features(benign_sample, label_value=0)
        benign_mapped['source_id'] = [f'nids_benign_{i}' for i in range(len(benign_mapped))]
        
        print(f"Added: {len(crypto_mapped)} crypto-like, {len(benign_mapped)} benign")
        
        combined = pd.concat([combined, crypto_mapped, benign_mapped], ignore_index=True)
    
    feature_cols = get_feature_columns()
    feature_cols = [f for f in feature_cols if f in combined.columns]
    final_cols = ['source_id'] + feature_cols + ['label']
    combined = combined[final_cols]
    
    print(f"\nTotal: {len(combined)} flows ({(combined['label']==0).sum()} benign, {(combined['label']==1).sum()} malicious)")
    print(f"Ratio: {len(combined) / len(df):.1f}x")
    
    output_path = Path('dataset/cn21_augmented.csv')
    combined.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
