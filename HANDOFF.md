# 快速交接

## 当前接力点

- `Self-Forcing` 六条实验线均已跑通并完成 VBench 评估，结果位于 `HeadWiseKVQuant/results/selfforcing/`：
  - `bf16/` — BF16 baseline (Final Score: 0.6486)
  - `triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — QVG INT2 baseline (0.6469, ↓0.26%)
  - `rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h PRQ (0.6416, ↓1.07%)
  - `rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Naive (0.5954, ↓8.19%)
  - `rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int8+int4 (0.6479, ↓0.10%)
  - `rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int4+int2 (0.6279, ↓3.19%)
- VBench 评估完整结果：`HeadWiseKVQuant/results/selfforcing/vbench_eval/comparison_summary.json`
- 评估脚本：`HeadWiseKVQuant/scripts/eval/evaluate_experiments.sh` 和 `scripts/eval/aggregate_results.py`
- **head importance analysis 两阶段方案已验证**（2026-05-17）：
  - Smoke test 在 A100 80GB 上完整跑通：Phase 1 (inference, ~25.5 GB) + Phase 2 (scoring, ~1.2 GB)
  - 两阶段拆分彻底解决 OOM 问题，可准备跑全量 360 heads
  - 启动脚本：`bash scripts/self_forcing/run_head_importance_analysis.sh`
- 已新增但尚未用真实 importance 文件跑视频的支路：
  - `headwise_mode=topk` — per-layer fixed top-k high precision heads
  - 启动脚本：`HeadWiseKVQuant/scripts/self_forcing/run_packed_naive_topk_hwq.sh`
  - 需要 `HEAD_IMPORTANCE_PATH=/path/to/topk_policy.json`
- `head-wise quant` 主线独立代码库：`HeadWiseKVQuant`。
- 后续优先在 `HeadWiseKVQuant` 中发展论文方法，`Quant-VideoGen` 只作为参考。
- 后续论文方法主线已经收敛到 `importance-based top-k head-wise quant`。

## 下一位 agent 先做什么

1. 先看 `STATUS.md`，再看 `MEMORY.md`
2. 确认六条实验线结果都在 `HeadWiseKVQuant/results/selfforcing/` 下
3. 查看 VBench 对比结果：`HeadWiseKVQuant/results/selfforcing/vbench_eval/comparison_summary.json`
4. 再看独立库结构：
   - `HeadWiseKVQuant/README.md`
   - `HeadWiseKVQuant/docs/quantization_approaches.md`
   - `HeadWiseKVQuant/docs/self_forcing_integration.md`
   - `HeadWiseKVQuant/docs/workspace_structure.md`
   - `HeadWiseKVQuant/src/hwq/headwise.py`
5. 下一步任务：跑全量 head importance analysis (360 heads) 生成真实 importance policy
   - `PHASE=inference bash scripts/self_forcing/run_head_importance_analysis.sh`
   - 然后：`ALLOW_INCOMPLETE=1 PHASE=scoring bash scripts/self_forcing/run_head_importance_analysis.sh`
6. 然后用 importance policy 跑 top-k HWQ 视频：
   - `HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json bash scripts/self_forcing/run_packed_naive_topk_hwq.sh`
7. 如继续跑实验：
   - Packed-naive R-HWQ-4h (推荐默认)：`bash scripts/self_forcing/run_packed_naive_hwq.sh`
   - Packed-naive int8+int4 版本：`HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 LOW_PRECISION_QUANT_TYPE=packed-naive-int4 QUANT_TYPE=packed-naive-int4 bash scripts/self_forcing/run_packed_naive_hwq.sh`
   - QVG PRQ 路径：`bash scripts/self_forcing/run_random_hwq.sh`
   - 注意：运行前需 `conda activate forcing`（包含 omegaconf 等依赖）
8. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

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
- 当前已落地的第一版 top-k granularity：
  - per-layer fixed top-k
  - K/V 共用同一组 heads
  - 全 prompts / chunks 共享固定 policy
  - `score_direction=higher` 表示 DMD loss 越大越重要
- 当前量化方案对比（VBench Final Score，6 条实验线）：
  | 方案 | Final Score | vs BF16 | 压缩方式 |
  |------|------------|---------|---------|
  | BF16 | 0.6486 | — | 无损 |
  | Packed int8+int4 | 0.6479 | ↓0.10% | 标量 min-max, real |
  | PRQ INT2 (all) | 0.6469 | ↓0.26% | 向量 K-Means, real |
  | PRQ R-HWQ-4h | 0.6416 | ↓1.07% | 向量 K-Means, real |
  | Packed int4+int2 | 0.6279 | ↓3.19% | 标量 min-max, real |
  | Naive R-HWQ-4h | 0.5954 | ↓8.19% | 标量 min-max, fake |
- 实验产物统一放到 `HeadWiseKVQuant/results/`（不要放 `outputs/`）。
- 本机权重位于 `HeadWiseKVQuant/ckpts/Self-Forcing/`（不进 git）。
- 显存注意事项：
  - QVG PRQ (triton-nstages-kmeans) 返回 int8 packed 格式，A100 80G 可直接跑
  - Packed-naive 返回 packed dict，是真压缩，A100 80G 可直接跑
  - Naive blockwise 量化返回 bf16 张量，需 `expandable_segments:True` 否则 OOM
- 运行环境：
  - Self-Forcing 推理：`conda activate forcing`
  - VBench 评估：`conda activate vbench`
- 当前工程判断：
  - 现有量化入口集中在 cache 管理层，而不是 attention kernel 内部
  - 当前张量与 cache 组织方式天然带 head 维度，适合做 per-head / head-group 策略
  - 当前已经跑通 random head-wise 样例；top-k policy 已接入，但需要真实 importance 文件和视频实验验证

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 如果确认 `head-wise quant` 的配置格式、分组策略或评测口径，记入 `DECISIONS.md`
- 如果实验结果或 Slurm 日志先落到 `/mnt/users/...` 运行副本，结束后默认同步一份回 `/data2/moweile-20251213/workspace/videoquant/...`
