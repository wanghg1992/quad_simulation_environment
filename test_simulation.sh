#!/bin/bash -i
conda activate py37
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
python3 test_simulation.py
