# prerequisites: windows npcap for windows packet capture 
# pip install cicflowmeter==0.4.2
# know your network interface name (probably "Wi-Fi" if on WiFi or "Eth1" if on ethernet or so on should check network configs)
# CLI Command for CICFlowMeter to send data to this csv:
# cicflowmeter -i "Wi-Fi" -c flows.csv

# then it should populate flows.csv with flow data 
# once flows.csv has at least one data point, run this script:
# python flow_inference.py

import numpy as np
import pandas as pd 
import os 
import joblib
import time

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')

def clear_csv(filepath):
  """Deletes all content, leaving a completely empty file."""
  f = open(filepath, "w+")
  f.close()

def extract_data(df):
  iat_threshold = 0.1

  duration = df['flow_duration'].sum() #duration
  total_packets = (df['tot_fwd_pkts']+df['tot_bwd_pkts']).sum() #total packets
  total_bytes = (df['totlen_fwd_pkts']+df['totlen_bwd_pkts']).sum() #total bytes
  avg_packet_size = df['pkt_len_mean'].sum()/len(df) #avg packet size
  avg_iat = df['flow_iat_mean'].sum()/len(df)  #avg inter-arrival time
  std_iat = df['flow_iat_std'].sum()/len(df)  #stddev inter-arrival time
  cv_iat_raw = std_iat/avg_iat #cv of inter-arrival time
  cv_iat = cv_iat_raw if avg_iat != 0 else 0
  cv_iat = np.nan_to_num(cv_iat, nan=0.0, posinf=0.0, neginf=0.0)
  burst_ratio = ((df['flow_iat_mean']/df['flow_duration'] < iat_threshold).sum())/len(df)
  packet_rate = df['flow_pkts_s'].sum()/len(df)  #packet rate
  byte_rate = df['flow_byts_s'].sum()/len(df)  #byte rate

  features = {
    'duration': [duration],
    'total_packets': [total_packets],
    'total_bytes': [total_bytes],
    'avg_packet_size': [avg_packet_size],
    'avg_iat': [avg_iat],
    'std_iat': [std_iat],
    'cv_iat': [cv_iat],
    'burst_ratio': [burst_ratio],
    'packet_rate': [packet_rate],
    'byte_rate': [byte_rate]
  }
  df_features = pd.DataFrame(features)
  print(df_features)
  return df_features

def monitor():
  CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')
  SCALER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
  RF_FILE_PATH = os.path.join(os.path.dirname(__file__), 'rf.joblib')
  SVM_FILE_PATH = os.path.join(os.path.dirname(__file__), 'svm.joblib')

  try:
    loaded_scaler = joblib.load(SCALER_FILE_PATH)
    loaded_rf = joblib.load(RF_FILE_PATH)
    loaded_svm = joblib.load(SVM_FILE_PATH)
  except Exception as e:
    return
  
  try:
    while True:
      # Read all new content from the current file position
      try:
        df = pd.read_csv(CSV_FILE_PATH).dropna(how='all').drop_duplicates()

        #clear csv for next batch
        df_header_only = df.iloc[0:0]
        df_header_only.to_csv(CSV_FILE_PATH, index=False)

        if len(df) > 0:
          feature_df = extract_data(df)  
          X_new_scaled = loaded_scaler.transform(feature_df)
          rf_pred = loaded_rf.predict(X_new_scaled)[0]
          rf_proba = loaded_rf.predict_proba(X_new_scaled)[0, 1]
          svm_pred = loaded_svm.predict(X_new_scaled)[0]
          svm_proba = loaded_svm.predict_proba(X_new_scaled)[0, 1]
          print("\n   [PREDICTION RESULTS]")
          print(f"   RF Classification: {'MALICIOUS' if rf_pred == 1 else 'NORMAL'} (Confidence: {rf_proba:.4f})")
          print(f"   SVM Classification: {'MALICIOUS' if svm_pred == 1 else 'NORMAL'} (Confidence: {svm_proba:.4f})")
          print("-" * 30)
      except pd.errors.EmptyDataError:
        pass
      except Exception as e:
        print(e)
        time.sleep(10)
      time.sleep(10)  
  except KeyboardInterrupt:
    print("\nMonitor stopped by user.")


if __name__ == "__main__":
  monitor()
