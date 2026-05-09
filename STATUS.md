# 当前状态

## 当前目标

- 将 `Self-Forcing` 场景下的 KV cache 低精度量化从 `QVG` 实验仓中独立出来，形成面向论文方法开发的 `HeadWiseKVQuant` 代码库，并在此基础上推进 `head-wise quant`。

## 正在做什么

- `R-HWQ-4h` 第一版真实推理已跑通，结果位于本机：
  - `/mnt/workspace/caipeiliang/code/moweile/videoquant/HeadWiseKVQuant/outputs/self_forcing/`
  - 配置：`rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1`
- 已产出 2 条 `R-HWQ-4h` 视频：`0-0_ema.mp4`、`1-0_ema.mp4`。
- 等待评估视频质量（identity consistency、scene consistency、motion continuity）。

## 最近完成

- **R-HWQ-4h 首次真实 Self-Forcing 推理跑通**（2026-05-09 凌晨）：
  - 配置：4 high-precision heads (int4) + remaining heads (int2)，block_size=64，256 K/V centroids，1 PRQ stage，seed=0
  - 成功生成 2 条视频，输出目录：`HeadWiseKVQuant/outputs/self_forcing/rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/`
- **修复 A100 兼容性问题**（commit `839fdbb`）：
  - `fp8e4nv` 自动回退：`quant_pack.py` 新增 `_gpu_supports_fp8e4nv()`，非 Hopper GPU 上 `float8_e4m3fn` 自动降级为 `bfloat16`
  - 视频保存修复：`inference.py` 从已废弃的 `torchvision.io.write_video` 切换到 `imageio.mimsave`
- 从 `QVG` 的 `quant_videogen` 中抽出可复用量化核心，整理到独立库 `HeadWiseKVQuant/src/hwq/`。
- 新增 `hwq.headwise`：`RandomHeadPolicy`、`compress_headwise_kv_cache`。
- 新增 `hwq.self_forcing`：`compress_self_forcing_cache_span`。
- 新增独立库文档与测试：
  - `HeadWiseKVQuant/README.md`
  - `HeadWiseKVQuant/docs/self_forcing_integration.md`
  - `HeadWiseKVQuant/tests/test_headwise.py`
- 已通过：`py_compile`、`python -m unittest discover -s tests -v`。
- 已把 `Quant-VideoGen` 的 `Self-Forcing` 量化路径切到独立库 `HeadWiseKVQuant`。
- 已把实验主入口和 Self-Forcing backend 迁到 `HeadWiseKVQuant`，不再要求存在 `Quant-VideoGen` 代码目录。
- 确认 `Self-Forcing` BF16 baseline 成功生成 2 条视频（已确认）。
- 确认 `QVG` 官方仓库当前提供三条实验集成：`LongCat-Video`、`Self-Forcing`、`HY-WorldPlay`。

## 当前阻塞 / 未完成

- 尚未评估 `R-HWQ-4h` 输出视频的质量退化情况（vs BF16 baseline、vs QVG INT2 baseline）。
- 目前只完成了第一版两组 mixed precision 路径（4h int4/int2），尚未扩展到更一般的多组 / importance-based policy。
- 实验矩阵其他组合（INT2-all、BF16 baseline 对比）尚未在 `HeadWiseKVQuant` 下统一重跑。

## 下一步

- 对比 `R-HWQ-4h` 视频与 BF16 baseline 的视频质量（identity/scene/motion）。
- 检查推理日志中的 Rel L2 量化误差是否合理。
- 统一实验矩阵，建立对比主线：
  - BF16 baseline
  - INT2-all baseline
  - R-HWQ-4h（已跑通）
  - R-HWQ-2h
- 如质量可接受，扩展到 importance-based policy（非均匀 head 分组）。
