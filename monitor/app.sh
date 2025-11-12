#!/bin/bash

# --- Configuration ---
INTERFACE="Wi-Fi"
CSV_FILE="flows.csv"
PYTHON_SCRIPT="flow_inference.py"

# --- Cleanup Function ---
# This function is executed when the script receives an interrupt signal (Ctrl+C) or exits.
cleanup() {
    echo -e "\n--- Stopping Pipeline Processes ---"
    
    # Check if cicflowmeter PID exists and kill it
    if [ ! -z "$FLOW_PID" ]; then
        echo "Terminating cicflowmeter (PID: $FLOW_PID)..."
        kill -9 "$FLOW_PID" 2>/dev/null
    fi

    # Check if python script PID exists and kill it
    if [ ! -z "$INFERENCE_PID" ]; then
        echo "Terminating Python script (PID: $INFERENCE_PID)..."
        kill "$INFERENCE_PID" 2>/dev/null
    fi

    echo "Pipeline stopped."
}

# Trap the EXIT and INT (Ctrl+C) signals to run the cleanup function
trap cleanup EXIT INT

# --- Pipeline Execution ---

echo "Starting CICFlowMeter on interface ${INTERFACE}..."
# 1. Start cicflowmeter in the background (&)
cicflowmeter -i "$INTERFACE" -c "$CSV_FILE" &
FLOW_PID=$!  # Capture the Process ID ($!) of the flow meter job

echo "Starting Python inference script: ${PYTHON_SCRIPT}..."
# 2. Start Python script in the background (&)
python "$PYTHON_SCRIPT" &
INFERENCE_PID=$! # Capture the Process ID of the inference job

echo "--- Pipeline Running ---"
echo "Press Ctrl+C to stop both processes gracefully."

# Wait for both background jobs to finish. 
# Since they run indefinitely, this command keeps the bash script alive.
# chmod +x app.sh
# ./app.sh
wait "$FLOW_PID" "$INFERENCE_PID"