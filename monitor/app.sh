#!/bin/bash

# chmod +x app.sh
# ./app.sh

INTERFACE="Wi-Fi"
CSV_FILE="flows.csv"
PYTHON_SCRIPT="flow_inference.py"

cleanup() {
    echo -e "\nStopping processes..."
    [ ! -z "$FLOW_PID" ] && kill -9 "$FLOW_PID" 2>/dev/null
    [ ! -z "$INFERENCE_PID" ] && kill "$INFERENCE_PID" 2>/dev/null
    echo "Stopped."
}

trap cleanup EXIT INT

echo "Starting CICFlowMeter on ${INTERFACE}..."
cicflowmeter -i "$INTERFACE" -c "$CSV_FILE" &
FLOW_PID=$!

echo "Starting inference..."
python "$PYTHON_SCRIPT" &
INFERENCE_PID=$!

echo "Running. Press Ctrl+C to stop."
wait "$FLOW_PID" "$INFERENCE_PID"