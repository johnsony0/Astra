"""Real-time flow inference for cryptomining detection."""

import numpy as np
import pandas as pd 
import os 
import joblib
import time

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')

def clear_csv(filepath):
  """Clear CSV file content."""
  open(filepath, "w+").close()

def save_to_csv(df):
  SAVE_PATH = os.path.join(os.path.dirname(__file__), 'extracted_features.csv')
  df_copy = df.copy()
  df_copy['flow_id'] = 0
  df_copy['label'] = 0
  df_ordered = df_copy[['flow_id', 'label'] + [c for c in df_copy.columns if c not in ['flow_id', 'label']]]
  df_ordered.to_csv(SAVE_PATH, mode='a', header=not os.path.exists(SAVE_PATH), index=False)

def extract_data(df):
  iat_threshold = 0.1
  duration = df['flow_duration'].sum()
  total_packets = (df['tot_fwd_pkts']+df['tot_bwd_pkts']).sum()
  total_bytes = (df['totlen_fwd_pkts']+df['totlen_bwd_pkts']).sum()
  avg_packet_size = df['pkt_len_mean'].sum()/len(df)
  avg_iat = df['flow_iat_mean'].sum()/len(df)
  std_iat = df['flow_iat_std'].sum()/len(df)
  cv_iat = np.nan_to_num(std_iat/avg_iat if avg_iat != 0 else 0, nan=0.0, posinf=0.0, neginf=0.0)
  burst_ratio = ((df['flow_iat_mean']/df['flow_duration'] < iat_threshold).sum())/len(df)
  small_packet_ratio = ((df['pkt_len_mean'] < 100).sum())/len(df)
  large_packet_ratio = ((df['pkt_len_mean'] > 1000).sum())/len(df)
  packet_rate = df['flow_pkts_s'].sum()/len(df)
  byte_rate = df['flow_byts_s'].sum()/len(df)

  return pd.DataFrame({
    'duration': [duration], 'total_packets': [total_packets], 'total_bytes': [total_bytes],
    'avg_packet_size': [avg_packet_size], 'avg_iat': [avg_iat], 'std_iat': [std_iat],
    'cv_iat': [cv_iat], 'burst_ratio': [burst_ratio], 'small_packet_ratio': [small_packet_ratio],
    'large_packet_ratio': [large_packet_ratio], 'packet_rate': [packet_rate], 'byte_rate': [byte_rate]
  })

def monitor():
  CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')
  SCALER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
  RF_FILE_PATH = os.path.join(os.path.dirname(__file__), 'rf.joblib')
  SVM_FILE_PATH = os.path.join(os.path.dirname(__file__), 'svm.joblib')

  try:
    loaded_scaler = joblib.load(SCALER_FILE_PATH)
    loaded_rf = joblib.load(RF_FILE_PATH)
    loaded_svm = joblib.load(SVM_FILE_PATH)
  except:
    return
  
  try:
    while True:
      try:
        df = pd.read_csv(CSV_FILE_PATH).dropna(how='all').drop_duplicates()
        df.iloc[0:0].to_csv(CSV_FILE_PATH, index=False)
        
        if len(df) > 0:
          feature_df = extract_data(df)
          save_to_csv(feature_df)
          X_scaled = loaded_scaler.transform(feature_df)
          rf_pred, rf_proba = loaded_rf.predict(X_scaled)[0], loaded_rf.predict_proba(X_scaled)[0, 1]
          svm_pred, svm_proba = loaded_svm.predict(X_scaled)[0], loaded_svm.predict_proba(X_scaled)[0, 1]
          print(f"\nRF: {'MALICIOUS' if rf_pred == 1 else 'NORMAL'} ({rf_proba:.4f})")
          print(f"SVM: {'MALICIOUS' if svm_pred == 1 else 'NORMAL'} ({svm_proba:.4f})")
      except pd.errors.EmptyDataError:
        pass
      except Exception as e:
        time.sleep(10)
      time.sleep(10)
  except KeyboardInterrupt:
    print("\nStopped.")


if __name__ == "__main__":
  monitor()
