#!/bin/bash

# to be run on riker

ENV_DIR="$HOME/riker_venv"

# List packages here
PACKAGES=(
    numpy
    mpi4py
)

module load Core/26.01
module load python/3.14.5

# Setup environment
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating virtual environment in '$ENV_DIR'..."
  python3 -m venv "$ENV_DIR"

  echo "Installing packages..."
  "$ENV_DIR/bin/pip3" install "${PACKAGES[@]}"
else
  echo "Using existing virtual environment '$ENV_DIR'."
fi

# Activate environment
# (only works when you run: source create_riker_venv.sh)
source "$ENV_DIR/bin/activate"

echo "Environment active: $(python -c 'import sys; print(sys.executable)')"