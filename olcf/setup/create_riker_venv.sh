#!/bin/bash

# to be run on riker

module load Core/26.01
module load python/3.14.5
python -m venv $HOME/riker_venv
source riker_venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install mpi4py
