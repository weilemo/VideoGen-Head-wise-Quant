# 快速交接

## 当前接力点

- `Self-Forcing` 的 `QVG` BF16 baseline 已确认跑通，主线已切到基于现有量化框架推进 `head-wise quant`。

## 下一位 agent 先做什么

1. 先看 `STATUS.md`
2. 再看 `MEMORY.md`
3. 确认 BF16 结果与日志：
   - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/qvg_sf_bf16_g-24046.out`
   - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/results/selfforcing/bf16/`
4. 再看 `docs/self_forcing_qvg_walkthrough.md`
5. 直接进入 `head-wise quant` 设计与实现
6. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## 当前最重要信息

- 当前核心目标不是一般 `LLM KV quant`，而是面向 `forcing-based long video generation`。
- 重点质量维度是 `identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线是：
  - 以 `Self-Forcing + QVG` 为主战场推进 `head-wise quant`
  - `LLM KV quant` 迁移线暂时退到次要位置
- 当前工程判断：
  - 现有量化入口集中在 cache 管理层，而不是 attention kernel 内部
  - 当前张量与 cache 组织方式天然带 head 维度，适合做 per-head / head-group 策略
  - 当前最大缺口不是“能不能接”，而是“如何设计 per-head 配置并串通压缩与解压元数据”

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 如果确认 `head-wise quant` 的配置格式、分组策略或评测口径，记入 `DECISIONS.md`
