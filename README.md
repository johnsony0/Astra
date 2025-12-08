# Astra - Cryptomining Detection System

Real-time network traffic monitoring and cryptojacking detection using machine learning with model-guided data augmentation.

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies for dashboard
cd web/dashboard
npm install
cd ../..
```

### 2. Data Setup (Optional for pre-trained models)

If you need to reproduce the dataset and training:

```bash
# Configure Kaggle (for UNSW-NB15 dataset)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Process datasets
python3 models/data_preprocessor.py
python3 models/augment_dataset.py

# Train models
python3 train_models.py --augmented
```

## Running the Application

You can run the system using the provided script or manually.

### Option A: Using the Run Script (Recommended)

This script starts both the Flask API backend and the React frontend.

```bash
cd web
./run.sh
```

- **Dashboard**: http://localhost:5173
- **API**: http://localhost:3000

### Option B: Manual Startup

**1. Start the Backend (Flask API)**

```bash
cd web
source ../venv/bin/activate
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
```

**2. Start the Frontend (React Dashboard)**

Open a new terminal:

```bash
cd web/dashboard
npm run dev
```

### Option C: Run Agent on Devices

To monitor traffic on a specific device and send data to the central server:

```bash
# On the device to be monitored
source venv/bin/activate
export ASTRA_API_URL=http://<SERVER_IP>:3000
python3 web/agent.py
```

## Project Structure

```
Astra/
├── dataset/                # Datasets (CN21, UNSW-NB15)
├── models/                 # ML models and preprocessing scripts
├── monitor/                # Network monitoring scripts
├── web/                    # Web application
│   ├── dashboard/          # React frontend
│   ├── app.py              # Flask backend
│   ├── agent.py            # Device monitoring agent
│   └── run.sh              # Startup script
├── tests/                  # Unit tests
├── notebook.ipynb          # Analysis notebook
└── train_models.py         # Training script
```

## Features

- **Real-time Detection**: Monitors network flows and classifies them as benign or malicious.
- **Hybrid Detection**: Uses Random Forest (91.5% accuracy) and SVM (79.5% accuracy) models.
- **Dashboard**: Visualizes network traffic, detections, and device status.
- **Model-Guided Augmentation**: Uses a novel approach to select cryptomining-like flows from botnet datasets.
