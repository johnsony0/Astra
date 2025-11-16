"""Train cryptomining detection models."""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_models(use_augmented=False, use_additional_features=False):
    """Train and save models."""
    print("=" * 60)
    print("CRYPTOMINING DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    if use_augmented:
        data_path = Path('dataset/cn21_augmented.csv')
        if not data_path.exists():
            print("Error: Augmented dataset not found")
            return
    else:
        data_path = Path('dataset/cn21_processed.csv')
        if not data_path.exists():
            print("Error: Processed dataset not found")
            return

    if use_additional_features:
        additional_features_path = Path('monitor/extracted_features.csv')
        if not additional_features_path.exists():
            print("Error: Processed dataset not found")
            return

    df = pd.read_csv(data_path)
    
    print(f"{'Augmented' if use_augmented else 'Original'} dataset: {len(df)} flows")
    print(f"Split: {(df['label']==0).sum()} benign, {(df['label']==1).sum()} malicious")
    
    feature_cols = ['duration', 'total_packets', 'total_bytes', 'avg_packet_size', 
                    'std_packet_size', 'avg_iat', 'std_iat', 'cv_iat', 'burst_ratio', 
                    'small_packet_ratio', 'large_packet_ratio', 'packet_rate', 'byte_rate']
    
    if 'source_id' in df.columns and not use_additional_features:
        unique_sources = df['source_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_sources)
        
        split_idx = int(len(unique_sources) * 0.8)
        train_sources = unique_sources[:split_idx]
        test_sources = unique_sources[split_idx:]
        
        train_mask = df['source_id'].isin(train_sources)
        test_mask = df['source_id'].isin(test_sources)
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
        y = df['label']

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        print(f"Source-based split: {len(train_sources)} train sources, {len(test_sources)} test sources")
        print(f"Flows: {len(X_train)} train ({(y_train==0).sum()} benign, {(y_train==1).sum()} malicious)")
        print(f"       {len(X_test)} test ({(y_test==0).sum()} benign, {(y_test==1).sum()} malicious)")
    else:
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
        y = df['label']

        if use_additional_features:
            add_feat_df = pd.read_csv(additional_features_path)
            X_add_feat = add_feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
            y_add_feat = add_feat_df['label']

            X = pd.concat([X, X_add_feat], ignore_index=True)
            y = pd.concat([y, y_add_feat], ignore_index=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Random split: {len(X_train)} train, {len(X_test)} test")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Malicious']))
    
    print("=" * 60)
    print("SVM")
    #can increase accuracy through increasing C value
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    svm.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=['Benign', 'Malicious']))
    
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head())
    
    print("=" * 60)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / 'scaler_v1.joblib')
    joblib.dump(rf, models_dir / 'rf_v1.joblib')
    joblib.dump(svm, models_dir / 'svm_v1.joblib')
    
    print(f"✓ Saved to {models_dir}/")


if __name__ == "__main__":
    import sys
    use_augmented = '--augmented' in sys.argv
    use_additional_features = '--add' in sys.argv
    train_models(use_augmented=use_augmented,use_additional_features=use_additional_features)

