# π0.5 LIBERO 独立复现

上游源码位于 `openpi/`，版本记录在 `docs/source-lock.json`。不修改既有 workcell、LeRobot 和虚拟环境。

本地修改后的 openpi 源码单独保存在 [openpi-libero](https://github.com/dongcbb/openpi-libero)。模型权重和数据不包含在本仓库中。

## 环境决策

不能直接复用 env_lerobot：当前 Torch 2.11.0+cu128、Transformers 5.5.4、NumPy 2.2.6，与上游固定的 Torch 2.7.1、Transformers 4.53.2、NumPy <2 不同，且缺少 JAX/Flax。不要在该环境中执行 openpi 的依赖安装或 transformers 文件覆盖。

计划将模型服务/训练环境与 LIBERO 仿真客户端环境隔离。官方服务环境 Python >=3.11；官方评估示例另用 Python 3.8，并固定旧版 robosuite、NumPy 和 Torch。评估客户端的旧 CUDA Torch 不应承担 RTX 5080 模型推理，模型在独立服务进程运行。RTX 5080 与固定 JAX/CUDA 依赖的兼容性必须实测，尚未验证。

## 验证顺序

1. 安装隔离环境与 LIBERO 子模块，检查 5080 上模型加载及 EGL 渲染。
2. 使用官方 pi05_libero 权重评估固定任务/种子，先验证评估链路。
3. 下载并固定公开数据的单任务子集，微调 pi05_base，记录显存、配置、随机种子。
4. 同条件比较基础模型、官方模型和新训练模型，再扩展任务组。

当前仅已拉取上游源码并完成环境审计，尚未安装新环境、下载权重/数据或启动训练。

## 当前进度

按用户要求直接使用 env_lerobot。已安装公共依赖、openpi/openpi-client/LIBERO，保留 Torch/Transformers/NumPy 版本。JAX CPU-only，不用于 GPU 训练。通过 `bash -c "source scripts/env.sh; /home/cy518/miniconda3/envs/env_lerobot/bin/python scripts/render_smoke.py"` 验证 LIBERO EGL 渲染。小模型构造通过，前向暴露新版 Transformers 的 language_model 路径兼容问题，尚未完成模型接口适配。尚未下载权重、数据或开始训练。

模型接口兼容修改及验证范围见 `docs/transformers-compatibility.md`。小模型完整前向、反向、两步动作采样已通过；真实权重与正式精度验证待完成。

## 官方权重准备

`scripts/prepare_reference.sh` 下载官方 gs://openpi-assets/checkpoints/pi05_libero 并转换 BF16 权重。转换器记录 missing/unexpected keys，仅容许未使用的 lm_head 缺失，其他差异报错。当前下载日志 `logs/reference-download.log`；当前已启动的后续转换日志 `logs/reference-conversion.log`。下载与转换不是闭环评估通过的证明。
