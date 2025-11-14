import numpy as np
import pandas as pd 
import os 
import joblib
import time
import requests
import json
from requests.exceptions import ConnectionError, Timeout, RequestException 

def extract_data(df):
  iat_threshold = 0.1
  duration = df['duration'].sum()
  total_packets = (df['total_packets']).sum()
  total_bytes = (df['total_bytes']).sum()
  avg_packet_size = df['avg_packet_size'].sum()/len(df)
  std_packet_size = df['std_packet_size'].sum()/len(df)
  avg_iat = df['avg_iat'].sum()/len(df)
  std_iat = df['std_iat'].sum()/len(df)
  cv_iat = df['std_iat'].sum()/len(df)
  burst_ratio = df['burst_ratio'].sum()/len(df)
  small_packet_ratio = df['small_packet_ratio'].sum()/len(df)
  large_packet_ratio = df['large_packet_ratio'].sum()/len(df)
  packet_rate = df['packet_rate'].sum()/len(df)
  byte_rate = df['byte_rate'].sum()/len(df)

  features_dict = {
    'duration': [duration], 'total_packets': [total_packets], 'total_bytes': [total_bytes],
    'avg_packet_size': [avg_packet_size], 'std_packet_size': [std_packet_size], 'avg_iat': [avg_iat], 'std_iat': [std_iat],
    'cv_iat': [cv_iat], 'burst_ratio': [burst_ratio], 'small_packet_ratio': [small_packet_ratio],
    'large_packet_ratio': [large_packet_ratio], 'packet_rate': [packet_rate], 'byte_rate': [byte_rate]
  }

  return features_dict

def run_local_inference(features_dict):
  feature_df = pd.DataFrame(features_dict)

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

def inference():
  CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'extracted_features.csv')
  df = pd.read_csv(CSV_FILE_PATH).dropna(how='all').drop_duplicates()
  
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
        print(f"Response not 200 error:{e}")
        run_local_inference(feature_dict)
    except (ConnectionError, Timeout, RequestException) as e:
      print(f"Server fail error:{e}")
      run_local_inference(feature_dict)
    except json.JSONDecodeError:
      print(f'JSON error: {e}')

if __name__ == "__main__":
  inference()