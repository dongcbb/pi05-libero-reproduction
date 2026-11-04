# Transformers 5.5.4 compatibility

Uses env_lerobot with Torch 2.11.0+cu128. Openpi revision is recorded in source-lock.json.

Project-local changes use LeRobot 0.6.2 PiGemma components for adaptive normalization and gated residuals, instead of overwriting Transformers site-packages. Fixed image feature output handling, configurable attention reshape and vision projection widths. Removed obsolete upstream patched-install check because the replacement backend is explicit.

Transformers 5 attention mutates a supplied cache even when use_cache=False. Suffix inference now operates on a deepcopy so each denoising step preserves the prefix cache. This has a potential memory/time cost and must be profiled on actual checkpoint inference.

Validation: scripts/check_model_compat.py tests joint/cached equivalence, prefix cache preservation and adaptive norm gradients on a dummy model. scripts/check_full_path.py tests three-camera encoding, loss backward and two denoising iterations on a dummy Pi05 model. FP32 CUDA tests passed; real pretrained checkpoint parity, BF16 training, full-size memory and LIBERO policy success rate have not been validated. This is a compatibility adaptation, not an unchanged official reproduction.
