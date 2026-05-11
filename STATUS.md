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
- 目前只完成了随机 head-wise mixed precision 路径，尚未实现 importance-based top-k policy。
- 尚未确定 head importance 的 metric、collection 方式和 policy granularity。

## 下一步

- 对比四条实验线的视频质量：
  - BF16 baseline
  - QVG INT2 baseline（triton PRQ）
  - R-HWQ-4h（triton PRQ，4 heads int4 / 8 heads int2）
  - R-HWQ-4h（naive int2/int4，无 PRQ，4 heads int4 / 8 heads int2）
- 跑通 packed-naive R-HWQ-4h：`bash scripts/self_forcing/run_packed_naive_hwq.sh`。
- 额外测试 packed-naive-int8：用 `HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 LOW_PRECISION_QUANT_TYPE=packed-naive-int8 QUANT_TYPE=packed-naive-int8` 覆盖脚本变量。
- 设计第一版 head importance metric，并明确它服务于 identity / scene / motion 哪类一致性。
- 设计 importance collection 流程：优先考虑少量 calibration prompts 离线统计，再固定 top-k policy 跑完整生成。
- 在 `HeadWiseKVQuant/src/hwq/headwise.py` 增加 `TopKHeadPolicy` 或等价 policy，复用现有 `compress_headwise_kv_cache`。
- 统一实验矩阵，建立对比主线：
  - BF16 baseline
  - INT2-all baseline
  - R-HWQ-4h（已跑通）
  - importance top-k HWQ
  - R-HWQ-2h
