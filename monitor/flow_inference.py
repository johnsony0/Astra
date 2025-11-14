"""Real-time flow inference for cryptomining detection."""

import numpy as np
import pandas as pd 
import os 
import joblib
import time
import requests
import json
from requests.exceptions import ConnectionError, Timeout, RequestException 

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
  std_packet_size = df['pkt_len_std'].sum()/len(df)
  avg_iat = df['flow_iat_mean'].sum()/len(df)
  std_iat = df['flow_iat_std'].sum()/len(df)
  cv_iat = np.nan_to_num(std_iat/avg_iat if avg_iat != 0 else 0, nan=0.0, posinf=0.0, neginf=0.0)
  burst_ratio = ((df['flow_iat_mean']/df['flow_duration'] < iat_threshold).sum())/len(df)
  small_packet_ratio = ((df['pkt_len_mean'] < 100).sum())/len(df)
  large_packet_ratio = ((df['pkt_len_mean'] > 1000).sum())/len(df)
  packet_rate = df['flow_pkts_s'].sum()/len(df)
  byte_rate = df['flow_byts_s'].sum()/len(df)

  features_dict = {
    'duration': [duration], 'total_packets': [total_packets], 'total_bytes': [total_bytes],
    'avg_packet_size': [avg_packet_size], 'std_packet_size': [std_packet_size], 'avg_iat': [avg_iat], 'std_iat': [std_iat],
    'cv_iat': [cv_iat], 'burst_ratio': [burst_ratio], 'small_packet_ratio': [small_packet_ratio],
    'large_packet_ratio': [large_packet_ratio], 'packet_rate': [packet_rate], 'byte_rate': [byte_rate]
  }

  return features_dict

def run_local_inference(features_dict):
  feature_df = pd.DataFrame(features_dict)
  save_to_csv(feature_df)

  SCALER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
  RF_FILE_PATH = os.path.join(os.path.dirname(__file__), 'rf.joblib')
  SVM_FILE_PATH = os.path.join(os.path.dirname(__file__), 'svm.joblib')

  try:
    loaded_scaler = joblib.load(SCALER_FILE_PATH)
    loaded_rf = joblib.load(RF_FILE_PATH)
    loaded_svm = joblib.load(SVM_FILE_PATH)
  except Exception as e:
    print(e)
    return

  X_scaled = loaded_scaler.transform(feature_df)
  rf_pred, rf_proba_0, rf_proba_1, = loaded_rf.predict(X_scaled)[0], loaded_rf.predict_proba(X_scaled)[0, 0], loaded_rf.predict_proba(X_scaled)[0, 1]
  svm_pred, svm_proba_0, svm_proba_1 = loaded_svm.predict(X_scaled)[0], loaded_svm.predict_proba(X_scaled)[0, 0], loaded_svm.predict_proba(X_scaled)[0, 1]
  print(f"Local Inference")
  print(f"RF: {f'MALICIOUS ({rf_proba_1:.4f})' if rf_pred == 1 else f'NORMAL ({rf_proba_0:.4f})'}")
  print(f"SVM: {f'MALICIOUS ({svm_proba_1:.4f})' if svm_pred == 1 else f'NORMAL ({svm_proba_0:.4f})'}\n")

def monitor():
  CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')
  
  try:
    while True:
      try:
        df = pd.read_csv(CSV_FILE_PATH).dropna(how='all').drop_duplicates()
        #df.iloc[0:0].to_csv(CSV_FILE_PATH, index=False)
        
        if len(df) > 0:
          feature_dict = extract_data(df)

          cleaned_dict = {}
          for key, value_list in feature_dict.items():
              if isinstance(value_list, list) and len(value_list) == 1:
                  value = value_list[0]
                  if isinstance(value, (int, float, np.generic)):
                      cleaned_dict[key] = float(value)
                  else:
                      cleaned_dict[key] = value
              else:
                  cleaned_dict[key] = value_list
                
          try:
            response = requests.post(
            "http://localhost:3000/api/flows", 
            json={
              'device_id': 'monitor_device_01',
              'features': cleaned_dict
            }, 
            timeout=5
            )

            if response.status_code == 200:
              result = response.json()
              rf_pred = result['rf_prediction']
              rf_proba_1 = result['rf_confidence']
              rf_proba_0 = 1-rf_proba_1
              svm_pred = result['svm_prediction']
              svm_proba_1 = result['svm_prediction']
              svm_proba_0 = 1-svm_proba_1

              print(f"Server Inference: Timestamp: {result['timestamp']}, Device ID: {result['device_id']}")
              print(f"RF: {f'MALICIOUS ({rf_proba_1:.4f})' if rf_pred == 1 else f'NORMAL ({rf_proba_0:.4f})'}")
              print(f"SVM: {f'MALICIOUS ({svm_proba_1:.4f})' if svm_pred == 1 else f'NORMAL ({svm_proba_0:.4f})'}\n")
            else:
              run_local_inference(feature_dict)
          except (ConnectionError, Timeout, RequestException) as e:
            run_local_inference(feature_dict)
          except json.JSONDecodeError:
            print(f'JSON error: {e}')
          
      except pd.errors.EmptyDataError:
        pass
      except Exception as e:
        print(e)
        time.sleep(10)
      time.sleep(10)
  except KeyboardInterrupt:
    print("\nStopped.")


if __name__ == "__main__":
  monitor()
