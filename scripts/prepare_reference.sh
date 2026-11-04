#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPRO_ROOT"
PYTHON=/home/cy518/miniconda3/envs/env_lerobot/bin/python
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
"$PYTHON" -B scripts/download_reference.py
"$PYTHON" -B openpi/examples/convert_jax_model_to_pytorch.py \
 --checkpoint-dir "$REPRO_ROOT/checkpoints/pi05_libero_jax" \
 --config-name pi05_libero \
 --output-path "$REPRO_ROOT/checkpoints/pi05_libero_pytorch" \
 --precision bfloat16
