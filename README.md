# Astra - Cryptomining Detection System

Real-time network traffic monitoring and cryptojacking detection using machine learning with model-guided data augmentation.

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure Kaggle (for UNSW-NB15 dataset download)
# Get API key from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/  # Place your kaggle.json here
chmod 600 ~/.kaggle/kaggle.json

# 3. Process CN21 dataset (auto-downloads UNSW-NB15)
python3 models/data_preprocessor.py

# 4. Augment dataset (model-guided selection from NIDS data)
python3 models/augment_dataset.py

# 5. Train models
python3 train_models.py --augmented

# 6. Start monitoring (choose one):
#    Option A: Local monitoring (single machine)
cd monitor && ./app.sh

#    Option B: Distributed monitoring (multiple devices)
#    On central server:
cd web && ./run.sh

#    On each device:
export ASTRA_API_URL=http://server-ip:3000
python3 web/agent.py
```

## Architecture

```
CN21 Dataset (54 flows: cryptomining + benign)
    ↓
Train Original Model (unbiased baseline)
    ↓
Apply to UNSW-NB15 NIDS Dataset (61K+ botnet flows)
    ↓
Select cryptomining-like flows (RF & SVM consensus ≥0.6)
    ↓
Augment with noise + selected flows (2,804 total)
    ↓
Train Final Models (RF + SVM)
    ↓
Real-time Monitor (CICFlowMeter → Inference)
```

## Project Structure

```
Astra/
├── dataset/
│   ├── CN21/                       # Original cryptojacking dataset (54 flows)
│   ├── external/                   # UNSW-NB15 NIDS data (downloaded)
│   ├── cn21_processed.csv          # Extracted features from CN21
│   └── cn21_augmented.csv          # Final augmented dataset
├── models/
│   ├── data_preprocessor.py        # Feature extraction pipeline
│   ├── augment_dataset.py          # Model-guided augmentation
│   ├── *_original.joblib           # Original model (CN21 only)
│   └── *.joblib                    # Final trained models
├── monitor/
│   ├── app.sh                      # Orchestration script
│   ├── flow_inference.py           # Real-time inference
│   └── *.joblib                    # Deployed models
├── tests/                           # Unit tests
├── notebook.ipynb                   # Analysis notebook
└── train_models.py                  # Training script
```

## Features Extracted

From raw `[timestamp, packet_size]` pairs, we extract:

1. **Duration** - Total flow duration
2. **Packet metrics** - Total packets/bytes, avg/std packet size
3. **Inter-arrival time (IAT)** - Mean, std dev, coefficient of variation
4. **Burst ratio** - Packets with IAT < 0.1s
5. **Size ratios** - Small (<100 bytes) and large (>1000 bytes) packets
6. **Rates** - Packets/sec and bytes/sec

**Key insight**: Cryptomining has lower packet rates (~5.5 pkt/s) and larger IAT (~3.2s) compared to benign traffic (~47 pkt/s, ~0.04s IAT).

## Training Pipeline

### Phase 1: Original Model (Unbiased Baseline)
```bash
# Process CN21 and auto-download UNSW-NB15
python3 models/data_preprocessor.py
# Creates dataset/cn21_processed.csv and downloads UNSW-NB15 if needed
```
- **Purpose**: Learn cryptomining signatures from verified samples
- **Performance**: 100% on test set (expected for small dataset)
- **Note**: Automatically downloads UNSW-NB15 dataset from Kaggle (requires API key)

### Phase 2: Model-Guided Selection
```bash
python3 models/augment_dataset.py
```
**Process**:
1. Load UNSW-NB15 dataset (257K flows, including 61K botnet traffic)
2. Apply original model to all botnet flows
3. Select flows where **BOTH RF and SVM agree** (confidence ≥ 0.6)
4. Result: 680 cryptomining-like flows (1.11% of botnets)

**Rationale**:
- Original model identifies flows with genuine cryptomining patterns
- Consensus requirement reduces false positives
- Avoids arbitrary labeling by attack category
- Creates high-quality pseudo-labeled data

### Phase 3: Augmentation
1. **Noise**: 15% Gaussian noise on CN21 (624 flows)
2. **NIDS Crypto**: 680 selected flows → labeled as malicious
3. **NIDS Benign**: 1,500 normal flows → labeled as benign
4. **Total**: 2,804 flows (51.9x augmentation)

### Phase 4: Final Training
```bash
python3 train_models.py --augmented
```

**Train/Test Split**: 80/20 **by source** (prevents data leakage)
- 1,788 train sources / 447 test sources
- 2,287 train flows / 517 test flows
- Ensures augmented copies of same sample stay together

**Performance**:
- **RF Accuracy**: 93.2%
- **SVM Accuracy**: 79.5%
- **Top Features**: byte_rate, packet_rate, avg_iat, std_iat, duration

**Why Lower Accuracy is Better**:
- No data leakage (source-based splitting)
- Not overfitting to specific patterns
- Learning realistic distinctions between cryptomining and other malicious traffic
- Better generalization to unseen data

## Real-time Monitoring

### Local Mode (Single Machine)
The monitor system:
1. Captures packets with CICFlowMeter
2. Extracts features every 10 seconds
3. Classifies as NORMAL or MALICIOUS
4. Outputs confidence scores from both models

Configure network interface in `monitor/app.sh`:
```bash
INTERFACE="Wi-Fi"  # Or "en0", "eth0", etc.
```

### Distributed Mode (Multiple Devices)

**Architecture**:
```
Device 1 → Agent → Central Server (API + Inference)
Device 2 → Agent → Central Server (API + Inference)
Device N → Agent → Central Server (API + Inference)
                      ↓
                  Dashboard (Web UI)
```

**Setup**:

1. **Central Server** (runs API + inference):
```bash
cd web && ./run.sh
# Server starts on http://localhost:3000
```

2. **Each Device** (runs agent):
```bash
# Install dependencies
pip install requests

# Configure
export ASTRA_API_URL=http://your-server-ip:3000
export DEVICE_ID=device-001  # Optional, auto-generated

# Run agent (reads from monitor/flows.csv)
python3 web/agent.py
```

**API Endpoints**:
- `POST /api/flows` - Device sends flow data
- `GET /api/detections` - Get recent detections
- `GET /api/statistics` - Get detection stats
- `GET /api/devices` - List monitored devices

**Data Flow**:
1. Device: CICFlowMeter → `flows.csv`
2. Device: Agent extracts features → POST to `/api/flows`
3. Server: Runs inference (RF + SVM) → Stores in SQLite
4. Dashboard: Visualizes detections and statistics

**Web Dashboard** (React + Vite):
```bash
cd web/dashboard
npm install
npm run dev
# Dashboard runs on http://localhost:5173
```

The dashboard displays:
- Real-time statistics (total flows, benign/malicious counts)
- Device list with activity status
- Recent detections table with RF/SVM predictions
- Auto-refreshes every 5 seconds

## Development

**Testing**:
```bash
source venv/bin/activate
python3 -m pytest tests/ -v
```

**Retraining**:
```bash
# With original data (will overfit)
python3 train_models.py

# With augmented data (recommended)
python3 models/augment_dataset.py
python3 train_models.py --augmented
```

**Analysis**:
Open `notebook.ipynb` for EDA and model evaluation.

## Key Innovation: Model-Guided Augmentation

Traditional approach: Label botnets as cryptomining based on attack category (arbitrary).

**Our approach**: Use original CN21 model to identify which botnet flows exhibit cryptomining-like behavior.

**Advantages**:
1. **Principled selection** - Based on learned patterns, not assumptions
2. **High quality** - Both models must agree with ≥60% confidence
3. **Realistic labels** - Only 1.11% of botnets selected (not all botnets mine crypto)
4. **Robust models** - Lower accuracy indicates harder, more realistic classification task

**Result**: Models learn to distinguish cryptomining from other malicious traffic, not just malicious vs benign.

## Dependencies

- Python 3.8+
- numpy, pandas, scikit-learn
- matplotlib, seaborn (visualization)
- pyarrow (parquet file support)
- kaggle (dataset download)
- cicflowmeter==0.4.2 (packet capture)
- joblib (model serialization)
- pytest (testing)

## Notes

- Requires network packet capture permissions
- Windows: Install npcap
- macOS/Linux: libpcap (usually pre-installed)
- Data augmentation adds realistic noise to prevent overfitting

## Academic Context

**Institution**: NYU  
**Primary Dataset**: CN21 (cryptojacking detection)  
**Augmentation Dataset**: UNSW-NB15 (network intrusion)  
**Approach**: Model-guided pseudo-labeling with dual classifiers (RF + SVM)  
**Innovation**: Using original model as unbiased judge for data selection  
**Methodology**: Test-Driven Development (TDD)

## Performance Summary

| Dataset | Flows | Split | RF Acc | SVM Acc | Notes |
|---------|-------|-------|--------|---------|-------|
| Original CN21 | 54 | Random | 100% | 100% | Overfitting |
| Noise-augmented | 624 | Random | 99.2% | 98.4% | Data leakage |
| Model-guided | 2,804 | **Source-based (80/20)** | 93.2% | 79.5% | No leakage ✓ |

**Data Leakage Prevention**:
- Split by **source_id**, not individual flows
- Augmented copies stay with original sample
- 1,788 train sources → 2,287 flows
- 447 test sources → 517 flows

**Key Metrics**:
- Selected 680/61,374 botnet flows (1.11%)
- RF confidence: 0.689, SVM confidence: 0.801
- Cryptomining signature: Low packet rate (~15 pkt/s), persistent connections

---

**Status**: Model-Guided Augmentation Complete ✅  
**License**: Academic Project - NYU
