# 当前状态

## 当前目标

- 基于 `QVG` 在 `Self-Forcing` 上的现有接入，推进 `head-wise quant` 方向，建立第一版可运行原型与对比基线。

## 正在做什么

- 已确认 `Self-Forcing` 的 `QVG` BF16 baseline 跑通，结果位于共享盘：
  - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/results/selfforcing/bf16/`
  - 对应日志：`/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/qvg_sf_bf16_g-24046.out`
- 已完成 `QVG` 论文公开信息、官方仓库脚本和核心量化代码的第一轮对齐。
- 已确认当前 `QVG/Self-Forcing` 量化实现适合作为后续 `head-wise quant` 的改造底座：
  - 量化入口集中在 `experiments/Self-Forcing/pipeline/causal_inference.py`
  - KV cache 以 `[B, H, S, D]` / `[B, S, H, D]` 组织，便于按 head 做分组策略
  - 当前限制主要是全局单一 `quant_config`，需要扩展为 per-head 或 head-group 配置
- 已开始 `R-HWQ` 第一版实现：
  - `inference.py` 已增加 head-wise 配置入口
  - `causal_inference.py` 已增加 `random` head-group policy 与 mixed quant 分支
  - `uncompress.py` / `kv_cache.py` 已支持按 group 重建完整 head 维度
  - 新增启动脚本：`scripts/Self-Forcing/run_random_hwq.sh`

## 最近完成

- 确认 `Self-Forcing` BF16 baseline 成功生成 2 条视频：
  - `0-0_ema.mp4`
  - `1-0_ema.mp4`
- 确认共享盘工作区 `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/` 是实验日志与结果的重要落点。
- 确认 `QVG` 官方仓库当前提供三条实验集成：
  - `LongCat-Video`
  - `Self-Forcing`
  - `HY-WorldPlay`

## 当前阻塞

- `R-HWQ` 代码虽然已接入，但还没有跑一次真实 `Self-Forcing` 推理验证。
- 目前只完成了第一版两组 mixed precision 路径，尚未扩展到更一般的多组 / importance-based policy。
- 还没有产出 `R-HWQ-4h` 的第一条视频结果和日志。

## 下一步

- 先用 `scripts/Self-Forcing/run_random_hwq.sh` 跑通 `R-HWQ-4h` 的第一条真实样例。
- 检查输出视频、日志、显存和量化误差打印是否正常。
- 第一版默认实验矩阵：
  - `BF16`
  - `INT2-all`
  - `R-HWQ-2h`
  - `R-HWQ-4h`
- 建立新的对比主线：
  - BF16 baseline
  - 现有 QVG INT2 baseline
  - 新的 `R-HWQ` baseline
