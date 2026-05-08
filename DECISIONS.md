# 关键决策

## D-2026-05-06-01 项目级记录放在 `videoquant` 目录下

- 决策：这套协作记录放在 `/data2/moweile-20251213/workspace/videoquant`。
- 原因：每个项目应维护自己的上下文，避免跨项目混用状态。
- 影响：`Codex` 和 `CC` 处理 `videoquant` 时，默认先看本目录记录文件。

## D-2026-05-06-02 长期事实与当前状态分离

- 决策：长期稳定信息写 `MEMORY.md`，当前进展写 `STATUS.md`。
- 原因：减少记录污染，方便 agent 快速判断哪些内容可信且长期有效。
- 影响：调试细节和一次性状态不进入长期记忆。

## D-2026-05-06-03 单次执行过程写入 `logs/`

- 决策：训练、评测、调试、排障、Slurm 提交等单次过程统一写进 `logs/`。
- 原因：便于回溯命令、路径、日志与 `jobid`。
- 影响：`STATUS.md` 可以保持短而清晰。

## D-2026-05-06-04 长任务默认遵守 Slurm 工作规则

- 决策：需要长期占用 CPU/GPU 的任务默认通过 Slurm 运行。
- 原因：更利于资源规范使用、监控和复现。
- 影响：重要任务应记录资源申请、日志路径和作业号。
- 参考：[服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## D-2026-05-07-05 `Self-Forcing + QVG` 作为 `head-wise quant` 主线底座

- 决策：后续主线以 `Self-Forcing + QVG` 现有实现为基础，直接推进 `head-wise quant`，不再把“完整复现更多 benchmark”作为近期主任务。
- 原因：`Self-Forcing` BF16 baseline 已经跑通并产出视频，说明这条链路可作为可靠起点；同时当前量化入口集中在 cache 管理层，改造成本低于重写 attention kernel。
- 影响：后续优先工作从“继续铺开 QVG 复现”转为“扩展现有量化配置、压缩和解压路径以支持 per-head 或 head-group 策略”。

## D-2026-05-07-06 第一版 `head-wise quant` 优先做 head-group mixed precision

- 决策：`head-wise quant` 第一版优先做 `head-group mixed precision`，不直接做 12 个 head 全独立策略。
- 原因：当前实现的主要限制在于全局单一 `quant_config`；先做 head-group 更容易控制配置复杂度、元数据格式和调试成本，同时足以验证“不同 head 量化敏感性不同”这一核心研究假设。
- 影响：第一版实现重点应放在：
  - per-group bitwidth / centroids / stages 配置
  - 压缩后元数据携带 group 配置
  - 解压时按 group 重建 `[B, H, S, D]`
