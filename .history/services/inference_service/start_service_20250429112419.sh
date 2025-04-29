#!/bin/bash

# Get the environment name and port from arguments
ENV_NAME=$1
PORT=$2

# Activate the Conda environment in a subshell
(
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    uvicorn src.app:app --host 0.0.0.0 --port $PORT
) 