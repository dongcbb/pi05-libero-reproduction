#!/usr/bin/env bash
set -euo pipefail
cd /home/cy518/workspace/dcb/pi05-libero-reproduction
while kill -0 1047531 2>/dev/null; do sleep 10; done
if ! grep -q DOWNLOAD_COMPLETE logs/reference-download.log; then
 printf 'Download did not complete; conversion not started.\n'
 exit 1
fi
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 /home/cy518/miniconda3/envs/env_lerobot/bin/python -B openpi/examples/convert_jax_model_to_pytorch.py --checkpoint-dir "$PWD/checkpoints/pi05_libero_jax" --config-name pi05_libero --output-path "$PWD/checkpoints/pi05_libero_pytorch" --precision bfloat16
