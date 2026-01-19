#!/bin/bash

# Run the Diabetes Analytics Streamlit App

echo "🏥 Starting Diabetes Hospital Analytics..."
echo "========================================"

# Navigate to the project directory
cd "$(dirname "$0")"

# Source UV
source $HOME/.local/bin/env 2>/dev/null || true

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Run streamlit
echo "Opening application in browser..."
uv run streamlit run app.py

