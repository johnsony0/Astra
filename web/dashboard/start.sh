#!/bin/bash

cd "$(dirname "$0")"

echo "Starting Astra Dashboard..."
echo "Dashboard: http://localhost:5173"
echo "API: http://localhost:3000"
echo ""
echo "Make sure the Flask API server is running:"
echo "  cd .. && ./run.sh"
echo ""

npm run dev

