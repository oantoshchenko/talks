#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Anatomy of AI Agents..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv --python 3.11
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -r pyproject.toml

# Install Jupyter kernel (always reinstall to ensure correct path)
echo "Setting up Jupyter kernel..."
.venv/bin/python -m ipykernel install --user --name=anatomy-of-ai-agents --display-name="Python (Anatomy of AI Agents)"

# Start Jupyter using the venv's jupyter
echo "Starting Jupyter Notebook..."
.venv/bin/python -m jupyter notebook anatomy_of_ai_agents.ipynb

