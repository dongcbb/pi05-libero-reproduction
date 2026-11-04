#!/usr/bin/env bash
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPRO_ROOT/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$REPRO_ROOT/configs/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONDONTWRITEBYTECODE=1
