# prerequisites: windows npcap for windows packet capture 
# pip install cicflowmeter==0.4.2
# know your network interface name (probably "Wi-Fi" if on WiFi or "Eth1" if on ethernet or so on should check network configs)
# CLI Command for CICFlowMeter to send data to this server:
# cicflowmeter -i "Wi-Fi" -c flows.csv
# then it should populate flows.csv with flow data 

import numpy as np
import pandas as pd 
import os 
import joblib
import time
import io

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'flows.csv')

def load_csv(filename):
  """Loads a CSV file into a Pandas DataFrame and displays key information."""
  
  if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found.")
    print("Please ensure 'sample_data.csv' is in the same directory as this script.")
    return
  try:
    df = pd.read_csv(filename)
    return df
  except pd.errors.EmptyDataError:
      print("Error: The CSV file is empty.")
  except pd.errors.ParserError as e:
      print(f"Error: Could not parse the CSV file. Check for formatting issues. Details: {e}")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")

def extract_data(df):
  duration = df['flow_duration'] #duration
  total_packets = df['tot_fwd_pkts']+df['tot_bwd_pkts'] #total packets
  total_bytes = df['totlen_fwd_pkts']+df['totlen_bwd_pkts'] #total bytes
  avg_packet_size = df['pkt_len_mean'] #avg packet size
  avg_iat = df['flow_iat_mean'] #avg inter-arrival time
  std_iat = df['flow_iat_std'] #stddev inter-arrival time
  cv_iat = df['flow_iat_std']/df['flow_iat_mean'] #cv of inter-arrival time
  cv_iat = cv_iat.replace([np.inf, -np.inf, np.nan], 0)
  packet_rate = df['flow_pkts_s'] #packet rate
  byte_rate = df['flow_byts_s'] #byte rate

  features = {
    'duration': duration,
    'total_packets': total_packets,
    'total_bytes': total_bytes,
    'avg_packet_size': avg_packet_size,
    'avg_iat': avg_iat,
    'std_iat': std_iat,
    'cv_iat': cv_iat,
    'packet_rate': packet_rate,
    'byte_rate': byte_rate
  }
  df_features = pd.DataFrame(features)
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
      df = pd.read_csv(CSV_FILE_PATH)
      
      if len(df) > 0:
        feature_df = extract_data(df.iloc[-1:])  
        X_new_scaled = loaded_scaler.transform(feature_df)
        rf_pred = loaded_rf.predict(X_new_scaled)[0]
        rf_proba = loaded_rf.predict_proba(X_new_scaled)[0, 1]
        svm_pred = loaded_svm.predict(X_new_scaled)[0]
        svm_proba = loaded_svm.predict_proba(X_new_scaled)[0, 1]
        print("\n   [PREDICTION RESULTS]")
        print(f"   RF Classification: {'MALICIOUS' if rf_pred == 1 else 'NORMAL'} (Confidence: {rf_proba:.4f})")
        print(f"   SVM Classification: {'MALICIOUS' if svm_pred == 1 else 'NORMAL'} (Confidence: {svm_proba:.4f})")
        print("-" * 30)
        time.sleep(120) # wait for 2 minutes before checking again
  except KeyboardInterrupt:
    print("\nMonitor stopped by user.")


if __name__ == "__main__":
  monitor()
