#!/bin/bash

cd "$(dirname "$0")"
source ../../venv/bin/activate

export FLASK_APP=app.py
export FLASK_ENV=development

# Cleanup function
cleanup() {
    echo -e "\nStopping servers..."
    [ ! -z "$FLASK_PID" ] && kill "$FLASK_PID" 2>/dev/null
    [ ! -z "$VITE_PID" ] && kill "$VITE_PID" 2>/dev/null
    echo "Stopped."
    exit 0
}

trap cleanup EXIT INT TERM

echo "Starting Astra Web Dashboard..."
echo "============================================================"

# Start Flask API
echo "Starting Flask API on http://localhost:3000..."
python app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 2

# Start Vite dev server
echo "Starting Vite dev server on http://localhost:5173..."
cd dashboard
npm run dev &
VITE_PID=$!

echo ""
echo "✓ Both servers running:"
echo "  - Flask API: http://localhost:3000"
echo "  - React Dashboard: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait for both processes
wait "$FLASK_PID" "$VITE_PID"

