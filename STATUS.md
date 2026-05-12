# 当前状态

## 当前目标

- 将 `Self-Forcing` 场景下的 KV cache 低精度量化从 `QVG` 实验仓中独立出来，形成面向论文方法开发的 `HeadWiseKVQuant` 代码库，并在此基础上推进 `head-wise quant`。

## 正在做什么

- 对比四条实验线的视频质量：BF16 baseline、QVG INT2 baseline、R-HWQ-4h（triton PRQ）、R-HWQ-4h（naive int2/int4，无 QVG PRQ）。
- `R-HWQ-4h` 第一版真实推理已跑通，说明现有 `headwise` 框架可作为后续方法底座。
- 后续主线从随机 head-wise baseline 转向 `importance-based top-k head-wise quant`。
- 当前最重要的研究工作聚焦三块：
  - `importance metric`：定义 head 重要性，例如量化敏感性、attention output 变化、denoising prediction 影响、跨 chunk 稳定性，或 identity/scene/motion 相关敏感性。
  - `importance collection`：确定离线 calibration、在线估计，或前几个 chunk calibration 后固定 policy。
  - `policy granularity`：确定全模型统一、per-layer、per-chunk、sink/history/tail、K/V 分开，或按 prompt 类型自适应的 top-k 策略。

## 最近完成

- **新增 per-layer top-k fixed policy 初版**（2026-05-12）：
  - 新增 `TopKHeadPolicy` 和 `load_topk_head_policy()`，支持从 JSON/CSV/TXT 读取 head 重要性或显式 top heads
  - 新增 `hwq.head_importance` 选头模块，库内支持读取 focused-forcing head-ablation DMD loss JSON、聚合均值、per-layer 选择 top-k heads、写出 policy JSON
  - 新增 vendored Self-Forcing head-ablation 分析链路：
    - `wan/modules/causal_model.py` 支持 `ablation_global_head_ids`，按 sample mask 指定 global head
    - `pipeline/causal_inference.py` 支持 latent-only analysis，避免 DMD 分析时额外 VAE decode
    - `backends/self_forcing/analyze_head_importance.py` 可直接跑 mask-head generation + DMD loss + policy 聚合
    - `scripts/self_forcing/run_head_importance_analysis.sh` 一键生成 `assets/head_importance/top4_dmd_loss.json`
  - `CausalInferencePipeline` 新增 `headwise_mode=topk`，量化每层 KV cache 时按 `layer_idx` 使用对应 top-k high-precision heads
  - packed-naive launcher 支持 `HEADWISE_MODE=topk`、`HEAD_IMPORTANCE_PATH`、`HEAD_IMPORTANCE_SCORE_DIRECTION`
  - 新增脚本：`HeadWiseKVQuant/scripts/self_forcing/run_packed_naive_topk_hwq.sh`
  - 新增聚合脚本：`HeadWiseKVQuant/scripts/aggregate_head_importance.py`，作为 `hwq.head_importance` 的 CLI wrapper，可把 focused-forcing ablation JSON 聚合成 top-k policy JSON
  - 新增文档：`HeadWiseKVQuant/docs/head_importance_topk.md`，包含跨机器路径处理和运行命令
  - 已通过 `py_compile`、`bash -n`、聚合脚本 smoke test、11 个单测
- **新增 packed-naive real-compression 支路**（2026-05-11）：
  - 新增 quant types：`packed-naive-int2`、`packed-naive-int4`、`packed-naive-int8`
  - 区别于旧 `naive-int2/int4` fake quant，packed-naive 会存储 uint8 packed codes + per-block min/scale metadata
  - 解压路径已接入 `uncompress_single_cache()`，head-wise mixed groups 可直接复用
  - 新增脚本：`HeadWiseKVQuant/scripts/self_forcing/run_packed_naive_hwq.sh`
  - 已通过 `py_compile`、`bash -n`、5 个单测和 `packed-naive-int8` smoke test
- **R-HWQ-4h naive int2/int4 跑通**（2026-05-10）：
  - 配置：4 high-precision heads (naive-int4) + 8 low-precision heads (naive-int2)，block_size=64
  - 不用 QVG 的 triton-nstages-kmeans PRQ，直接用 blockwise 量化
  - 需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才能在 A100 80G 上跑通（naive 量化返回 bf16 张量，不压缩显存，峰值 >76G）
  - 产出 2 条视频：`results/selfforcing/rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1/`
- **R-HWQ-4h（triton PRQ）首次真实 Self-Forcing 推理跑通**（2026-05-09）：
  - 配置：4 high-precision heads (int4) + remaining heads (int2)，block_size=64，256 K/V centroids，1 PRQ stage，seed=0
  - 产出 2 条视频：`results/selfforcing/rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/`
- **QVG INT2 baseline 在 A100 上跑通**：
  - 对应 Slurm 作业：`24309`，状态 `COMPLETED`，耗时 `00:24:58`
  - 产出 2 条视频：`results/selfforcing/triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/`
- **修复 A100 兼容性**：
  - `fp8e4nv` 自动回退：`quant_pack.py` 新增 `_gpu_supports_fp8e4nv()`，非 Hopper GPU 自动降级 bf16
  - 视频保存：`inference.py` 从已废弃的 `torchvision.io.write_video` 切到 `imageio.mimsave`
- 从 `QVG` 的 `quant_videogen` 中抽出可复用量化核心，整理到独立库 `HeadWiseKVQuant/src/hwq/`。
- 新增 `hwq.headwise`：`RandomHeadPolicy`、`compress_headwise_kv_cache`。
- 新增 `hwq.self_forcing`：`compress_self_forcing_cache_span`。
- 已通过：`py_compile`、`python -m unittest discover -s tests -v`。
- 实验产物目录统一为 `HeadWiseKVQuant/results/`（替代 `outputs/`）。

## 当前阻塞 / 未完成

- 尚未评估各实验线输出视频的质量退化情况（vs BF16 baseline）。
- 旧 naive fake-quant 量化需要 `expandable_segments:True` 才能跑通；新的 packed-naive real-compression 分支尚未跑真实 Self-Forcing 视频。
- `importance-based top-k policy` 已有初版代码路径，但尚未用真实 focused-forcing 重要性文件跑 Self-Forcing 视频。
- 尚未系统比较 `random HWQ` 与 `importance top-k HWQ` 的视频质量差异。
- head importance 目前采用 focused-forcing head ablation 的 DMD loss 聚合；后续仍需评估它和 identity / scene / motion 质量维度的相关性。

## 下一步

- 对比四条实验线的视频质量：
  - BF16 baseline
  - QVG INT2 baseline（triton PRQ）
  - R-HWQ-4h（triton PRQ，4 heads int4 / 8 heads int2）
  - R-HWQ-4h（naive int2/int4，无 PRQ，4 heads int4 / 8 heads int2）
- 跑通 packed-naive R-HWQ-4h：`bash scripts/self_forcing/run_packed_naive_hwq.sh`。
- 跑完整 head-importance analysis 生成 policy：
  `bash scripts/self_forcing/run_head_importance_analysis.sh`。
- 如果已有外部 focused-forcing JSON，也可生成 policy：
  `python scripts/aggregate_head_importance.py --input /path/to/jsons --output assets/head_importance/top4_dmd_loss.json --top_k 4`。
- 跑通 packed-naive importance top-k HWQ：
  `HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json bash scripts/self_forcing/run_packed_naive_topk_hwq.sh`。
- 额外测试 packed-naive-int8：用 `HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 LOW_PRECISION_QUANT_TYPE=packed-naive-int8 QUANT_TYPE=packed-naive-int8` 覆盖脚本变量。
- 在真实实验中验证 focused-forcing DMD ablation score 能否作为 head importance metric。
- 统一实验矩阵，建立对比主线：
  - BF16 baseline
  - INT2-all baseline
  - R-HWQ-4h（已跑通）
  - importance top-k HWQ
  - R-HWQ-2h
