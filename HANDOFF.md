# 快速交接

## 当前接力点

- `Self-Forcing` 的 `QVG` BF16 baseline 已确认跑通。
- `head-wise quant` 主线已进一步拆出独立代码库：`/data2/moweile-20251213/workspace/videoquant/HeadWiseKVQuant`。
- 后续应优先在 `HeadWiseKVQuant` 中发展论文方法，再把 `Quant-VideoGen` / `Self-Forcing` 当成下游集成与实验入口。

## 下一位 agent 先做什么

1. 先看 `STATUS.md`
2. 再看 `MEMORY.md`
3. 确认 BF16 结果与日志：
   - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/qvg_sf_bf16_g-24046.out`
   - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/results/selfforcing/bf16/`
4. 再看独立库：
  - `HeadWiseKVQuant/README.md`
  - `HeadWiseKVQuant/docs/self_forcing_integration.md`
   - `HeadWiseKVQuant/docs/workspace_structure.md`
  - `HeadWiseKVQuant/src/hwq/headwise.py`
5. 如果继续实现，优先进入 `HeadWiseKVQuant` 并运行 `bash scripts/self_forcing/run_random_hwq.sh` 验证第一条真实 `R-HWQ-4h` 样例
6. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## 当前最重要信息

- 当前核心目标不是一般 `LLM KV quant`，而是面向 `forcing-based long video generation`。
- 重点质量维度是 `identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线是：
  - 以 `HeadWiseKVQuant` 为方法代码库推进 `head-wise quant`
  - 以 `Self-Forcing + QVG` 为实验集成入口
  - 从 `HeadWiseKVQuant/scripts/self_forcing/` 启动实验，Self-Forcing backend 已复制到 `HeadWiseKVQuant/backends/self_forcing/`
  - `Quant-VideoGen` 只作为原始参考仓保留；后续没有它也能继续改代码
  - 本机已把大权重复制到 `HeadWiseKVQuant/ckpts/Self-Forcing/`，但该目录不进 git；另一台机器按 `HeadWiseKVQuant/docs/checkpoint_sync.md` 从 Hugging Face 下载同一套权重
  - `LLM KV quant` 迁移线暂时退到次要位置
- 当前工程判断：
  - 现有量化入口集中在 cache 管理层，而不是 attention kernel 内部
  - 当前张量与 cache 组织方式天然带 head 维度，适合做 per-head / head-group 策略
  - 当前已经把下游 Self-Forcing 推理正式切到独立库，最大缺口变成“跑通真实 `R-HWQ-4h` 样例并检查视频质量”

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 如果确认 `head-wise quant` 的配置格式、分组策略或评测口径，记入 `DECISIONS.md`
