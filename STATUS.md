# 当前状态

## 当前目标

- 将 `Self-Forcing` 场景下的 KV cache 低精度量化从 `QVG` 实验仓中独立出来，形成面向论文方法开发的 `HeadWiseKVQuant` 代码库，并在此基础上推进 `head-wise quant`。

## 正在做什么

- importance top-k HWQ 已跑通并完成 VBench 评估（↓2.82% vs BF16），优于 random HWQ（↓3.19%），验证了 DMD-loss 选头的有效性
- 核心论文指标：subject_consistency / background_consistency 几乎无损，退化集中在 aesthetic_quality / motion_smoothness
- 当前最重要的研究工作聚焦三块：
  - `importance metric`：定义 head 重要性，例如量化敏感性、attention output 变化、denoising prediction 影响、跨 chunk 稳定性，或 identity/scene/motion 相关敏感性。
  - `importance collection`：确定离线 calibration、在线估计，或前几个 chunk calibration 后固定 policy。
  - `policy granularity`：确定全模型统一、per-layer、per-chunk、sink/history/tail、K/V 分开，或按 prompt 类型自适应的 top-k 策略。

## 最近完成

- **拷贝 Focused-Forcing 参考代码与 DMD loss 数据**（2026-05-17）：
  - 将 `/data2/moweile-20251213/workspace/focused-forcing-code` 拷贝到 `external/focused-forcing-code/`
  - 新增 `external/README.md`，明确 `external/` 用于存放外部参考代码和分析结果，`HeadWiseKVQuant/` 继续作为主方法代码库
  - 确认 `focusedforcing_sf/cf/rf/longlive/dm_loss.json` 均包含 360 个 global head 的 DMD loss 分数，可用于生成 top-k policy
- **新增每周实验汇报目录**（2026-05-17）：
  - 新增 `HeadWiseKVQuant/weekly_reports/README.md`，规范每周汇报的命名、结构和维护方式
  - 新增 `HeadWiseKVQuant/weekly_reports/2026-W20.md`，整理本周六条实验线、两阶段 head importance 进展、问题和下周计划，方便和老师同步并请老师指导方向
- **补充两条 packed-naive R-HWQ-4h 实验及 VBench 评估**（2026-05-17）：
  - 跑通两条新实验线：
    | 实验线 | 配置 | 最终得分 | vs BF16 |
    |--------|------|---------|---------|
    | R-HWQ-4h Packed (int8+int4) | HIGH=packed-naive-int8, LOW=packed-naive-int4, 4 random heads | 0.6479 | ↓0.10% |
    | R-HWQ-4h Packed (int4+int2) | HIGH=packed-naive-int4, LOW=packed-naive-int2, 4 random heads | 0.6279 | ↓3.19% |
  - int8+int4 packed-naive 几乎无损（↓0.10%），是无需 k-means 的轻量化选项
  - int4+int2 介于 PRQ（↓1.07%）和更大退化之间，提供 5.3× 真压缩
  - 输出目录：`results/selfforcing/rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/` 和 `results/selfforcing/rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/`
  - 更新 `aggregate_results.py` 至 5 条实验线（不含 naive）
- **四条实验线 VBench 视频质量对比**（2026-05-17）：
  - 使用 VBench-Long 8 维度（subject_consistency, background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality, overall_consistency, clip_score）评估 4 条实验线各 2 条视频
  - 修复 2 个 VBench 兼容性 bug（视频名排序解析、split_clip 缓存正则）
  - 评估结果：
    | 实验线 | Final Score | vs BF16 |
    |--------|------------|---------|
    | BF16 Baseline | 0.6486 | - |
    | QVG INT2 (PRQ) | 0.6469 | ↓0.26% |
    | R-HWQ-4h (PRQ) | 0.6416 | ↓1.07% |
  - PRQ-based 量化质量退化极小（全 INT2 ~0.3%，R-HWQ-4h ~1%），naive blockwise 退化显著（~8%）
  - 新增评估脚本：`scripts/eval/evaluate_experiments.sh`、`scripts/eval/aggregate_results.py`
  - 结果存档：`results/selfforcing/vbench_eval/comparison_summary.json`
  - 现有 `run_head_importance_analysis.sh` 已按 `PHASE=inference/scoring/all` 调用两阶段脚本
  - 修复 Phase 2 resume 逻辑：按 chunk 判断已完成 head，避免某个 head 只在部分 chunk 出现时被错误整体跳过
  - launcher 新增实测参数：`HEAD_START`、`HEAD_END`、`ALLOW_INCOMPLETE`、`DELETE_LATENTS_AFTER_SCORING`、`SKIP_EXISTING`
  - 更新 `docs/head_importance_topk.md`，补充两阶段运行、smoke test 和恢复方式
  - 已通过 `py_compile`、`bash -n`、11 个单测
- **head importance analysis 脚本调试**（2026-05-14）：
  - 发现并修复 3 个代码 bug：
    - `self_forcing_dmd.yaml:5`：`real_name: Wan2.1-T2V-14B` → `1.3B`（本地只有 1.3B 权重）
    - `wan_wrapper.py:150-151`：`enable_gradient_checkpointing(enable=True)` 签名不兼容，改为直接设 `self.model.gradient_checkpointing = True`
    - `analyze_head_importance.py:118-121`：DMD text_encoder 未移到 GPU，加入短暂 GPU 编码后立即回 CPU
  - 尝试 3 种内存优化（DMD score CPU offloading、DMD T5 短暂 GPU 使用、heads_per_batch=1），均未能解决 OOM
  - 根因：Self-Forcing 推理阶段 KV cache 膨胀到 78+ GB（126 frames），剩余空间不足以加载 DMD score 模型（~5 GB）
  - 结论：需拆成两阶段脚本 — Phase 1 只推理存 latent，Phase 2 只算 DMD loss
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
- **引入 external focused-forcing-code + 产出完整 top-k policy**（2026-05-17）：
  - Pull 入 `external/focused-forcing-code/`，包含上游 4 份 DMD loss JSON（cf/sf/rf/longlive，360 heads 各一份，内容相同）
  - 用 `scripts/aggregate_head_importance.py` 直接将 `focusedforcing_sf/dm_loss.json` 聚合成 top-4 policy
  - 产出 `assets/head_importance/top4_dmd_loss.json`：30 layers，每层 top-4 heads
  - 新增文档 `docs/quantization_approaches.md`：Naive / Packed-naive / PRQ 三类量化方案对比
  - 两阶段 smoke test 也已验证通过（42 frames, 6 heads），但因 external 已有现成 360-heads 结果，无需自己跑全量

## 当前阻塞 / 未完成

- head importance 目前采用 focused-forcing head ablation 的 DMD loss 聚合；后续仍需评估它和 identity / scene / motion 质量维度的相关性。
- 当前 per-layer top-4 DMD-loss heads 质量提升 vs random 仅 +0.38pp，可能需要更好的 importance metric 或更细粒度的 policy（per-chunk, K/V 分开）
- Top-K HWQ Packed 仍落后 PRQ 路线（↓2.82% vs ↓1.07%），int8+int4 配置的 top-k 版本尚未测试

## 下一步

- **优先**: Top-K HWQ + int8+int4 配置（`HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 LOW_PRECISION_QUANT_TYPE=packed-naive-int4`），预期逼近 PRQ 质量线
- 探索其他 importance metric（量化敏感性、attention output 变化、跨 chunk 稳定性）
- 实验更细粒度 policy：per-chunk top-k、K/V 分开选头
- R-HWQ-2h 消融实验
- 统一实验矩阵当前状态：
  - BF16 baseline：✅ ↓0.00%，~80 GB
  - QVG INT2 (PRQ)：✅ ↓0.26%，~20 GB
  - R-HWQ-4h PRQ (int4+int2)：✅ ↓1.07%，~20 GB
  - R-HWQ-4h Packed (int8+int4)：✅ ↓0.10%，~40 GB
  - R-HWQ-4h Packed (int4+int2)：✅ ↓3.19%，~26 GB
  - **Top-K HWQ Packed (int4+int2)**：✅ ↓2.82%，26 GB
  - Top-K HWQ Packed (int8+int4)：❌ 待跑
  - R-HWQ-2h：❌ 待跑
