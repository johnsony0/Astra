#!/bin/bash

cd "$(dirname "$0")"
source ../../venv/bin/activate

export FLASK_APP=app.py
export FLASK_ENV=development

echo "Starting Astra Web Dashboard..."
python app.py

