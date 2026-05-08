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

## D-2026-05-08-07 独立出 `HeadWiseKVQuant` 作为论文方法代码库

- 决策：从 `Quant-VideoGen` 中抽出 KV cache 低精度量化框架，建立独立代码库 `/data2/moweile-20251213/workspace/videoquant/HeadWiseKVQuant`。
- 原因：后续论文方法不应长期绑在 QVG 实验仓里；独立库更适合作为 `head-wise quant` 方法主体，便于模块化、复现实验和后续开源整理。
- 影响：后续方法开发优先发生在 `HeadWiseKVQuant/src/hwq/`；`Quant-VideoGen` 的 `Self-Forcing` 代码应逐步退化为下游调用方，只负责推理调度和实验输出。

## D-2026-05-08-08 `Quant-VideoGen` 作为下游集成入口调用 `hwq`

- 决策：`Quant-VideoGen/experiments/Self-Forcing` 不再维护独立的 head-wise 量化实现，统一从 `HeadWiseKVQuant` 的 `hwq` 包导入缓存、压缩、解压和随机 head-group policy。
- 原因：避免同一研究逻辑在 QVG 实验仓和独立方法库中分叉；后续新增 importance-based / 多组策略时，只需要优先改 `HeadWiseKVQuant`。
- 影响：运行 QVG Self-Forcing 脚本时需要让 Python 找到独立库，例如从 `Quant-VideoGen` 目录运行时使用 `PYTHONPATH=../HeadWiseKVQuant/src:experiments/Self-Forcing:.`。

## D-2026-05-08-09 `HeadWiseKVQuant` 作为实验主工作区

- 决策：后续默认从 `HeadWiseKVQuant` 启动 Self-Forcing 实验。
- 原因：研究主线应围绕 head-wise quant 方法库展开；从方法仓启动实验更符合论文代码组织，也减少“主工作区还在 QVG”带来的概念混乱。
- 影响：新增 `HeadWiseKVQuant/scripts/self_forcing/`；最初可指向外部 QVG backend，后续由 D-2026-05-08-10 升级为默认调用本仓库 vendored backend。

## D-2026-05-08-10 Vendored Self-Forcing backend 进入 `HeadWiseKVQuant`

- 决策：将 `Quant-VideoGen/experiments/Self-Forcing` 的模型和 pipeline 代码复制到 `HeadWiseKVQuant/backends/self_forcing/`，使 `HeadWiseKVQuant` 自身成为可继续开发的完整代码工作区。
- 原因：如果未来只保留或只 clone `HeadWiseKVQuant`，仍应能继续修改 Self-Forcing pipeline 和 head-wise quant 接入，不应依赖旁边必须存在 `Quant-VideoGen` 代码目录。
- 影响：运行脚本默认调用 vendored backend；大模型权重不入库，默认放 `HeadWiseKVQuant/ckpts/Self-Forcing/`，或通过 `SELF_FORCING_CKPT_ROOT` 指向共享权重目录。
