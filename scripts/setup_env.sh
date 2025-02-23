#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/log.sh"

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

log "Installing package tools..."
pip install --upgrade pip setuptools wheel

log "Installing dependencies..."
pip install uv
uv pip install -r requirements.txt

log "Environment setup complete. Virtual environment is activated!"