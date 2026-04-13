#!/bin/bash
# Run this once on Snellius to set up the project
set -euo pipefail

PROJECT_DIR="$HOME/Tracing-representation-geometry-reproduction"
HF_CACHE="/projects/prjs1815/hf_cache"

cd "${PROJECT_DIR}"

# Install deps via uv (creates .venv automatically)
uv sync
