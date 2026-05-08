# 长期记忆

## 项目定位

- 项目目录：`/data2/moweile-20251213/workspace/videoquant`
- 这是一个面向长视频自回归 diffusion 生成的研究项目工作区。
- 项目目标：研究 KV cache quantization，尽量在显著压缩历史 KV 显存占用的同时，保持长时视频质量，重点关注 `identity consistency`、`scene consistency` 和 `motion continuity`。
- 当前可见子目录：
  - `forcing`
  - `Quant-VideoGen`

## 当前问题定义

- 研究对象不是普通 `LLM` 推理，而是 `forcing-based long video generation`。
- 核心矛盾是：视频越长，历史 `KV cache` 越大，显存很快成为瓶颈，进而限制可生成时长和长程一致性。
- 当前需要系统回答的问题包括：
  - `LLM` 的通用 `KV quantization` 方法能否直接迁移到长视频 diffusion。
  - 如果不能直接迁移，主要质量损伤更集中出现在 `motion`、`identity` 还是 `scene layout`。
  - 视频中不同时间段历史、不同 attention heads 是否对量化敏感性不同。

## 当前技术路线

- 技术路线分成两条主线并行推进：
  - 视频原生量化线
  - `LLM KV quant` 方法迁移线

## 视频原生量化线

- 当前最贴题的起点是 `QVG (Quant VideoGen)`。
- `QVG` 直接研究 autoregressive video generation 中的 `2-bit KV-cache quantization`。
- `QVG` 不是简单照搬 `LLM` 量化，而是采用视频特化设计，当前需要重点关注：
  - `Semantic-Aware Smoothing`
  - `Progressive Residual Quantization`

## LLM 方法迁移线

- 这条线的目标是把成熟的 `LLM KV quant` 方法迁移到视频生成框架中，作为可复现 baseline。
- 当前优先级建议：
  - `KIVI`
  - `KVQuant`
  - `GEAR`
- 当前默认优先从 `KIVI` 开始，因为它实现相对简单、`training-free`，更适合作为现有视频代码的第一版迁移基线。

## 当前研究判断

- 单纯把 `LLM KV quant` 原样迁到视频生成中，可能可以节省显存，但不一定能保住长时视频质量，尤其可能先伤害 `motion dynamics`。
- 视频量化更可能需要结构感知设计，而不是全局统一 bit 配置。
- 当前最有希望的后续创新方向：
  - `head-wise mixed precision`
  - `role-aware quantization`，即按 `sink / history / tail` 等时间角色分配不同 bit

## 协作记录约定

- 本目录下的记录文件只服务 `videoquant` 项目。
- `MEMORY.md` 只记录长期稳定事实。
- `STATUS.md` 记录当前任务状态。
- `HANDOFF.md` 用于 agent 快速交接。
- `logs/` 保存单次任务日志。

## 服务器执行约定

- 训练、推理、评测、批处理等长任务，默认通过 Slurm 运行。
- 登录节点只做编辑、查看日志、提交作业、短时间调试。
- 正式任务尽量记录：脚本路径、环境名、资源申请、日志路径、输出路径、`jobid`。
- `videoquant` 相关 Slurm 日志与实验输出不一定落在本地工作区 `/data2/...`，经常会落在共享盘工作区 `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/` 下；查实验结果时需要同时检查这一路径。

## Slurm 与服务器规则

- 总规则参考：[服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)
- 如果数据在 `/mnt`，优先考虑 `a100_global`
- 如果数据在本地盘 `/data` 或 `/data2`，优先考虑对应本地分区
- 正式任务建议显式记录：分区、账号、GPU、CPU、内存、时长、日志路径

## 待后续补充的长期信息

- 主代码仓路径与分工
- 常用训练脚本路径
- 常用评测脚本路径
- 常用环境名与激活方式
- 常用数据路径与输出路径
- 常用日志目录规范
