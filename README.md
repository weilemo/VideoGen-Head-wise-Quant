# videoquant 项目协作记录

`videoquant` 聚焦长视频自回归 diffusion 生成中的 KV cache quantization，目标是在显著降低 KV 显存占用的同时，尽量不破坏长时视频质量，尤其是 identity consistency、scene consistency 和 motion continuity。

这套记录文件专门服务 `workspace/videoquant`，供 `Codex` 和 `CC` 在服务器上共同接手开发、训练、调试、评测与作业排查。

## 目录

- `README.md`：说明如何使用这套记录。
- `MEMORY.md`：长期稳定事实。
- `STATUS.md`：当前状态。
- `HANDOFF.md`：agent 交接页。
- `DECISIONS.md`：关键决策。
- `TASK_TEMPLATE.md`：单次任务日志模板。
- `logs/`：单次任务日志目录。

## 每次开始任务先看什么

1. `STATUS.md`
2. `HANDOFF.md`
3. `MEMORY.md`
4. `DECISIONS.md`
5. 需要上下文细节时，再看 `logs/` 最近几条

## 维护规则

- 长期稳定信息写 `MEMORY.md`，不要写临时状态。
- 当前任务进展写 `STATUS.md`，保持短。
- agent 切换前更新 `HANDOFF.md`，只保留下一位立刻需要的信息。
- 单次训练、调试、环境排查、Slurm 作业提交，统一写到 `logs/`。
- 服务器资源使用遵守 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)。

## 建议命名

- 任务日志：`logs/2026-05-06_1530_训练排查.md`
- 如果涉及 Slurm，标题里尽量写脚本名、`jobid` 或节点信息。
