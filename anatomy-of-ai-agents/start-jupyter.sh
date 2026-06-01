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

# Scope everything to this venv: uv installs go here, and `jupyter`
# subcommands resolve to .venv/bin/* instead of any system jupyter on PATH.
export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
echo "Installing dependencies..."
uv pip install -r pyproject.toml

# Install Jupyter kernel (always reinstall to ensure correct path)
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=anatomy-of-ai-agents --display-name="Python (Anatomy of AI Agents)"

# Start Jupyter using the venv's notebook module directly — `python -m jupyter notebook`
# dispatches via PATH lookup of jupyter-notebook and can pick up a system install.
echo "Starting Jupyter Notebook..."
exec python -m notebook anatomy_of_ai_agents.ipynb

