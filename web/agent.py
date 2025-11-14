"""Lightweight agent for devices to send flow data to central server."""

import pandas as pd
import numpy as np
import requests
import time
import os
from pathlib import Path
from datetime import datetime
import json

API_URL = os.getenv('ASTRA_API_URL', 'http://localhost:3000')
DEVICE_ID = os.getenv('DEVICE_ID', f'device-{os.uname().nodename}')
FLOWS_CSV = Path('monitor/flows.csv')


def extract_features_from_cicflowmeter(df):
    """Extract 13 features from CICFlowMeter output."""
    iat_threshold = 0.1
    duration = df['flow_duration'].sum()
    total_packets = (df['tot_fwd_pkts'] + df['tot_bwd_pkts']).sum()
    total_bytes = (df['totlen_fwd_pkts'] + df['totlen_bwd_pkts']).sum()
    avg_packet_size = df['pkt_len_mean'].sum() / len(df)
    avg_iat = df['flow_iat_mean'].sum() / len(df)
    std_iat = df['flow_iat_std'].sum() / len(df)
    cv_iat = np.nan_to_num(std_iat / avg_iat if avg_iat != 0 else 0, nan=0.0, posinf=0.0, neginf=0.0)
    burst_ratio = ((df['flow_iat_mean'] / df['flow_duration'] < iat_threshold).sum()) / len(df)
    small_packet_ratio = ((df['pkt_len_mean'] < 100).sum()) / len(df)
    large_packet_ratio = ((df['pkt_len_mean'] > 1000).sum()) / len(df)
    packet_rate = df['flow_pkts_s'].sum() / len(df)
    byte_rate = df['flow_byts_s'].sum() / len(df)
    
    return {
        'duration': float(duration),
        'total_packets': int(total_packets),
        'total_bytes': int(total_bytes),
        'avg_packet_size': float(avg_packet_size),
        'std_packet_size': 0.0,
        'avg_iat': float(avg_iat),
        'std_iat': float(std_iat),
        'cv_iat': float(cv_iat),
        'burst_ratio': float(burst_ratio),
        'small_packet_ratio': float(small_packet_ratio),
        'large_packet_ratio': float(large_packet_ratio),
        'packet_rate': float(packet_rate),
        'byte_rate': float(byte_rate)
    }


def send_flow_data(features):
    """Send flow features to central server."""
    payload = {
        'device_id': DEVICE_ID,
        'timestamp': datetime.utcnow().isoformat(),
        'features': features
    }
    
    try:
        response = requests.post(
            f'{API_URL}/api/flows',
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {e}")
        return None


def monitor_and_send():
    """Monitor flows.csv and send to server."""
    last_size = 0
    
    while True:
        try:
            if not FLOWS_CSV.exists():
                time.sleep(10)
                continue
            
            current_size = FLOWS_CSV.stat().st_size
            if current_size == last_size:
                time.sleep(10)
                continue
            
            df = pd.read_csv(FLOWS_CSV)
            if len(df) == 0:
                time.sleep(10)
                continue
            
            features = extract_features_from_cicflowmeter(df)
            result = send_flow_data(features)
            
            if result:
                print(f"Sent: {result.get('prediction', 'unknown')} "
                      f"(RF: {result.get('rf_confidence', 0):.3f}, "
                      f"SVM: {result.get('svm_confidence', 0):.3f})")
            
            df.iloc[0:0].to_csv(FLOWS_CSV, index=False)
            last_size = FLOWS_CSV.stat().st_size
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
        
        time.sleep(10)


if __name__ == '__main__':
    print(f"Agent starting for device: {DEVICE_ID}")
    print(f"API URL: {API_URL}")
    monitor_and_send()

