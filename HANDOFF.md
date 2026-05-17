# 快速交接

## 当前接力点

- `Self-Forcing` 七条实验线均已跑通并完成 VBench 评估，结果位于 `HeadWiseKVQuant/results/selfforcing/`：
  - `bf16/` — BF16 baseline (Final Score: 0.6486)
  - `triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — QVG INT2 baseline (0.6469, ↓0.26%)
  - `rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h PRQ (0.6416, ↓1.07%)
  - `rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int8+int4 (0.6479, ↓0.10%)
  - `rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int4+int2 (0.6279, ↓3.19%)
  - **`topk_top4_dmd_loss_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — Top-K HWQ Packed int4+int2 (0.6303, ↓2.82%)** ← 新增
- VBench 评估完整结果：`HeadWiseKVQuant/results/selfforcing/vbench_eval/comparison_summary.json`
- 评估脚本：`HeadWiseKVQuant/scripts/eval/evaluate_experiments.sh` 和 `scripts/eval/aggregate_results.py`
- **Importance top-k policy 已就绪**：
  - 完整 per-layer top-4 head policy: `assets/head_importance/top4_dmd_loss.json`（30 layers × 12 heads, 360 scores）
  - 来源: `external/focused-forcing-code/focusedforcing_sf/dm_loss.json` → `scripts/aggregate_head_importance.py`
  - 已验证: Top-K HWQ 推理 + VBench 评估完成
- **head importance analysis 两阶段方案已验证**：
  - Smoke test 在 A100 80GB 上完整跑通：Phase 1 (inference, ~25.5 GB) + Phase 2 (scoring, ~1.2 GB)
  - 两阶段拆分彻底解决 OOM 问题
  - 启动脚本：`bash scripts/self_forcing/run_head_importance_analysis.sh`（PHASE=inference/scoring/all）

## 下一位 agent 先做什么

1. 先看 `STATUS.md`，再看 `MEMORY.md`
2. 查看 VBench 对比结果：`results/selfforcing/vbench_eval/comparison_summary.json`
3. 查看量化方案文档：`docs/quantization_approaches.md`
4. 再看独立库结构：
   - `HeadWiseKVQuant/README.md`
   - `HeadWiseKVQuant/docs/self_forcing_integration.md`
   - `HeadWiseKVQuant/docs/workspace_structure.md`
   - `HeadWiseKVQuant/src/hwq/headwise.py`
5. 优先任务：**Top-K HWQ + int8+int4**，预期逼近 PRQ 质量线
   ```
   HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 LOW_PRECISION_QUANT_TYPE=packed-naive-int4 \
   HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
   bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
   ```
6. 探索更好的 importance metric（当前 DMD loss top-4 仅比 random 高 0.38pp）
7. 如继续跑实验：
   - Packed-naive R-HWQ-4h：`bash scripts/self_forcing/run_packed_naive_hwq.sh`
   - 注意：`conda activate forcing`（包含 omegaconf 等依赖）
8. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## 当前最重要信息

- 当前核心目标是 `forcing-based long video generation` 的 KV cache 量化。
- 重点质量维度：`identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线：
  - 以 `HeadWiseKVQuant` 为方法代码库推进 `head-wise quant`
  - 以 `Self-Forcing` 为实验集成入口
  - 从 `HeadWiseKVQuant/scripts/self_forcing/` 启动实验
- 当前研究重点：
  - `importance metric`：定义 head 重要性分数（当前: DMD loss，Top-K vs Random = +0.38pp）
  - `importance collection`：离线 calibration / 在线估计 / 前几个 chunk 后固定
  - `policy granularity`：per-layer / per-chunk / sink-history-tail / K-V 分开 / prompt 自适应
- 当前量化方案 VBench 对比（180 frames, 717 video frames）：

| 方案 | Final Score | vs BF16 | Peak VRAM | KV Cache |
|------|------------|---------|-----------|----------|
| BF16 Baseline | 0.6486 | — | ~80 GB * | ~78 GB * |
| Packed int8+int4 (random) | 0.6479 | ↓0.10% | ~40 GB ** | ~25 GB ** |
| QVG INT2 (PRQ) | 0.6469 | ↓0.26% | ~20 GB ** | ~12 GB ** |
| R-HWQ-4h PRQ (int4+int2) | 0.6416 | ↓1.07% | ~20 GB ** | ~12 GB ** |
| **Top-K HWQ Packed (int4+int2)** | **0.6303** | ↓2.82% | **26 GB** | **12.4 GB** |
| R-HWQ-4h Packed (int4+int2) | 0.6279 | ↓3.19% | ~26 GB ** | ~12 GB ** |
| R-HWQ-4h Naive (int4+int2) | 0.5954 | ↓8.19% | >76 GB * | >76 GB * |

\* 据 STATUS.md / MEMORY.md 记录
** 据量化类型推算（结果在共享盘生成）
粗体 = 本次实测

- 实验产物统一放到 `HeadWiseKVQuant/results/`（不要放 `outputs/`）。
- 本机权重位于 `HeadWiseKVQuant/ckpts/Self-Forcing/`（不进 git）。
- 运行环境：
  - Self-Forcing 推理：`conda activate forcing`
  - VBench 评估：`conda activate vbench`

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 展示结果时务必包含：显存数据（Peak VRAM / KV Cache）、帧数
- 如果实验结果或 Slurm 日志先落到 `/mnt/users/...` 运行副本，结束后默认同步一份回本地
