"""
Web dashboard for Astra cryptomining detection system.
Provides real-time monitoring and historical data visualization.
Supports both local and distributed monitoring.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
import sqlite3
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Paths
MONITOR_DIR = Path(__file__).parent.parent / 'monitor'
EXTRACTED_FEATURES_PATH = MONITOR_DIR / 'extracted_features.csv'
DB_PATH = Path(__file__).parent / 'detections.db'

# Load models
SCALER_PATH = MONITOR_DIR / 'scaler.joblib'
RF_PATH = MONITOR_DIR / 'rf.joblib'
SVM_PATH = MONITOR_DIR / 'svm.joblib'

scaler = None
rf_model = None
svm_model = None

def load_models():
    """Load ML models."""
    global scaler, rf_model, svm_model
    try:
        scaler = joblib.load(SCALER_PATH)
        rf_model = joblib.load(RF_PATH)
        svm_model = joblib.load(SVM_PATH)
        print("✓ Models loaded")
    except Exception as e:
        print(f"⚠️  Models not loaded: {e}")

def init_database():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            timestamp TEXT,
            rf_prediction INTEGER,
            rf_confidence REAL,
            svm_prediction INTEGER,
            svm_confidence REAL,
            features TEXT,
            label INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            name TEXT,
            last_seen TEXT,
            total_flows INTEGER DEFAULT 0,
            malicious_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

load_models()
init_database()


@app.route('/')
def index():
    """Main dashboard page."""
    return jsonify({
        'message': 'Astra API Server',
        'dashboard': 'http://localhost:5173',
        'endpoints': [
            'POST /api/flows - Send flow data from device',
            'GET /api/detections - Get recent detections',
            'GET /api/statistics - Get detection statistics',
            'GET /api/devices - List monitored devices'
        ]
    })


@app.route('/api/flows', methods=['POST'])
def api_receive_flows():
    """Receive flow data from devices and run inference."""
    if not scaler or not rf_model or not svm_model:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        device_id = data.get('device_id', 'unknown')
        features_dict = data.get('features', {})
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        
        feature_cols = ['duration', 'total_packets', 'total_bytes', 'avg_packet_size',
                       'std_packet_size', 'avg_iat', 'std_iat', 'cv_iat', 'burst_ratio',
                       'small_packet_ratio', 'large_packet_ratio', 'packet_rate', 'byte_rate']
        
        features = np.array([[features_dict.get(f, 0) for f in feature_cols]])
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        X_scaled = scaler.transform(features)
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_proba = rf_model.predict_proba(X_scaled)[0, 1]
        svm_pred = svm_model.predict(X_scaled)[0]
        svm_proba = svm_model.predict_proba(X_scaled)[0, 1]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (device_id, timestamp, rf_prediction, rf_confidence, svm_prediction, svm_confidence, features, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, timestamp, int(rf_pred), float(rf_proba),
            int(svm_pred), float(svm_proba), json.dumps(features_dict),
            int(rf_pred)  # Use RF as ground truth for now
        ))
        
        cursor.execute('''
            INSERT OR REPLACE INTO devices (device_id, last_seen, total_flows, malicious_count)
            VALUES (?, ?, 
                COALESCE((SELECT total_flows FROM devices WHERE device_id = ?), 0) + 1,
                COALESCE((SELECT malicious_count FROM devices WHERE device_id = ?), 0) + ?
            )
        ''', (device_id, timestamp, device_id, device_id, int(rf_pred)))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'device_id': device_id,
            'prediction': 'malicious' if rf_pred == 1 else 'benign',
            'rf_prediction': int(rf_pred),
            'rf_confidence': float(rf_proba),
            'svm_prediction': int(svm_pred),
            'svm_confidence': float(svm_proba),
            'timestamp': timestamp
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Get current system status."""
    try:
        # Check if monitoring is active by looking at files
        flows_exist = (MONITOR_DIR / 'flows.csv').exists()
        features_exist = EXTRACTED_FEATURES_PATH.exists()
        models_loaded = all([
            (MONITOR_DIR / 'rf.joblib').exists(),
            (MONITOR_DIR / 'svm.joblib').exists(),
            (MONITOR_DIR / 'scaler.joblib').exists()
        ])
        
        return jsonify({
            'status': 'active' if flows_exist else 'inactive',
            'features_available': features_exist,
            'models_loaded': models_loaded,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detections')
def api_detections():
    """Get recent detection results."""
    try:
        limit = request.args.get('limit', 50, type=int)
        device_id = request.args.get('device_id', None)
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if device_id:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE device_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (device_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in rows:
            features = json.loads(row['features'])
            detections.append({
                'id': row['id'],
                'device_id': row['device_id'],
                'timestamp': row['timestamp'],
                'rf_prediction': row['rf_prediction'],
                'rf_confidence': row['rf_confidence'],
                'svm_prediction': row['svm_prediction'],
                'svm_confidence': row['svm_confidence'],
                'label': row['label'],
                'label_text': 'Malicious' if row['label'] == 1 else 'Benign',
                'features': features
            })
        
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def api_statistics():
    """Get detection statistics."""
    try:
        device_id = request.args.get('device_id', None)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if device_id:
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as malicious
                FROM detections WHERE device_id = ?
            ''', (device_id,))
        else:
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as malicious
                FROM detections
            ''')
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        malicious = row[1] or 0
        benign = total - malicious
        
        return jsonify({
            'total_flows': int(total),
            'benign_count': int(benign),
            'malicious_count': int(malicious),
            'detection_rate': float(malicious / total * 100) if total > 0 else 0.0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/devices')
def api_devices():
    """List all monitored devices."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM devices ORDER BY last_seen DESC')
        rows = cursor.fetchall()
        conn.close()
        
        devices = []
        for row in rows:
            devices.append({
                'device_id': row['device_id'],
                'name': row['name'] or row['device_id'],
                'last_seen': row['last_seen'],
                'total_flows': row['total_flows'],
                'malicious_count': row['malicious_count']
            })
        
        return jsonify({'devices': devices})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature_distribution')
def api_feature_distribution():
    """Get feature distribution data for visualization."""
    try:
        if not EXTRACTED_FEATURES_PATH.exists():
            return jsonify({'features': {}})
        
        df = pd.read_csv(EXTRACTED_FEATURES_PATH).tail(100)
        
        features = {}
        feature_cols = ['packet_rate', 'byte_rate', 'avg_packet_size', 'cv_iat', 'burst_ratio']
        
        for col in feature_cols:
            if col in df.columns:
                features[col] = {
                    'benign': df[df['label'] == 0][col].tolist(),
                    'malicious': df[df['label'] == 1][col].tolist()
                }
        
        return jsonify({'features': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("ASTRA WEB DASHBOARD")
    print("=" * 70)
    print(f"API URL: http://localhost:3000")
    print(f"Database: {DB_PATH}")
    print("\nEndpoints:")
    print("  POST /api/flows - Receive flow data from devices")
    print("  GET  /api/detections - Get recent detections")
    print("  GET  /api/statistics - Get detection statistics")
    print("  GET  /api/devices - List monitored devices")
    print("\nStarting server...")
    app.run(debug=True, host='0.0.0.0', port=3000)

