# 快速交接

## 当前接力点

- `Self-Forcing` 四条实验线均已跑通，结果位于 `HeadWiseKVQuant/results/selfforcing/`：
  - `bf16/` — BF16 baseline
  - `triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — QVG INT2 baseline (Slurm job 24309)
  - `rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h (QVG PRQ)
  - `rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h (naive blockwise, 无 PRQ)
- 已新增但尚未跑真实视频的支路：
  - `packed-naive-int2/int4/int8` — real packed blockwise quant，区别于旧 naive fake quant
  - 启动脚本：`HeadWiseKVQuant/scripts/self_forcing/run_packed_naive_hwq.sh`
- `head-wise quant` 主线独立代码库：`HeadWiseKVQuant`。
- 后续优先在 `HeadWiseKVQuant` 中发展论文方法，`Quant-VideoGen` 只作为参考。
- 后续论文方法主线已经收敛到 `importance-based top-k head-wise quant`。

## 下一位 agent 先做什么

1. 先看 `STATUS.md`，再看 `MEMORY.md`
2. 确认四条实验线结果都在 `HeadWiseKVQuant/results/selfforcing/` 下
3. 再看独立库结构：
   - `HeadWiseKVQuant/README.md`
   - `HeadWiseKVQuant/docs/self_forcing_integration.md`
   - `HeadWiseKVQuant/docs/workspace_structure.md`
   - `HeadWiseKVQuant/src/hwq/headwise.py`
4. 下一步任务：评估四条实验线的视频质量（identity/scene/motion consistency）
5. 如果继续方法实现，优先围绕 head importance 做 `TopKHeadPolicy` / importance collection，而不是继续扩展 random policy 本身
6. 如继续跑实验：
   - QVG PRQ 路径：直接 `bash scripts/self_forcing/run_random_hwq.sh`
   - Naive 量化路径：需加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（否则 OOM）
   - Packed naive 路径：`bash scripts/self_forcing/run_packed_naive_hwq.sh`
7. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## 当前最重要信息

- 当前核心目标是 `forcing-based long video generation` 的 KV cache 量化。
- 重点质量维度：`identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线：
  - 以 `HeadWiseKVQuant` 为方法代码库推进 `head-wise quant`
  - 以 `Self-Forcing` 为实验集成入口
  - 从 `HeadWiseKVQuant/scripts/self_forcing/` 启动实验
- 当前研究重点：
  - `importance metric`：定义 head 重要性分数。
  - `importance collection`：确定离线 calibration、在线估计，或前几个 chunk calibration 后固定。
  - `policy granularity`：确定 top-k 是全模型统一、per-layer、per-chunk，还是按 sink/history/tail、K/V 或 prompt 类型细分。
- 实验产物统一放到 `HeadWiseKVQuant/results/`（不要放 `outputs/`）。
- 本机权重位于 `HeadWiseKVQuant/ckpts/Self-Forcing/`（不进 git）。
- 显存注意事项：
  - QVG PRQ (triton-nstages-kmeans) 返回 int8 packed 格式，A100 80G 可直接跑
  - naive blockwise 量化返回 bf16 张量，需 `expandable_segments:True` 否则 OOM
  - packed-naive blockwise 返回 packed dict，是真压缩分支，支持 int2/int4/int8
- 当前工程判断：
  - 现有量化入口集中在 cache 管理层，而不是 attention kernel 内部
  - 当前张量与 cache 组织方式天然带 head 维度，适合做 per-head / head-group 策略
  - 当前已经跑通 random head-wise 样例；下一阶段最大缺口是判断 head 重要性并替换 random policy

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 如果确认 `head-wise quant` 的配置格式、分组策略或评测口径，记入 `DECISIONS.md`
- 如果实验结果或 Slurm 日志先落到 `/mnt/users/...` 运行副本，结束后默认同步一份回 `/data2/moweile-20251213/workspace/videoquant/...`
