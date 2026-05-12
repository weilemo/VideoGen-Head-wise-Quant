# HeadWiseKVQuant

`HeadWiseKVQuant` is a standalone research codebase for low-bit KV-cache
quantization in long video generation.

The first version extracts the reusable KV-cache quantization framework from
`Quant-VideoGen` and makes the head-wise mixed precision policy explicit.  The
goal is to keep the paper-facing method code modular instead of coupling it to
one experiment directory.

## Scope

This repository contains:

- low-bit KV-cache compression and decompression
- chunked KV-cache storage for mixed BF16 / quantized spans
- KMeans / PRQ based quantization kernels
- random head-group mixed precision as the first `HWQ` baseline
- fixed per-layer importance top-k head-group mixed precision
- a small compatibility layer for Self-Forcing style KV tensors
- a vendored Self-Forcing inference backend

This repository intentionally does not contain:

- Wan / Self-Forcing model weights

The recommended research workflow is to work from `HeadWiseKVQuant`.  The
Self-Forcing model and pipeline code are vendored under
`backends/self_forcing/`, while large checkpoints stay outside git.

Expected workspace layout:

```text
HeadWiseKVQuant/
├── src/hwq/                 # method code
├── backends/self_forcing/   # Self-Forcing model and pipeline code
├── scripts/self_forcing/    # launchers
├── assets/t2v.txt           # default prompts
└── ckpts/Self-Forcing/      # optional local checkpoint path, ignored by git
```

If your checkpoints live elsewhere, set
`SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing`.

The local checkpoint directory is ignored by git.  It is safe to keep large
weights under `HeadWiseKVQuant/ckpts/Self-Forcing/` without pushing them.

## Install

From this directory:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Minimal API

```python
from hwq import QuantizeConfig
from hwq.headwise import RandomHeadPolicy, compress_headwise_kv_cache

policy = RandomHeadPolicy(
    num_heads=12,
    num_high_precision_heads=4,
    high_precision_quant_type="triton-nstages-kmeans-int4",
    low_precision_quant_type="triton-nstages-kmeans-int2",
    seed=0,
)

quant_config = QuantizeConfig(
    quant_type="triton-nstages-kmeans-int2",
    quant_block_size=64,
    cache_num_k_centroids=256,
    cache_num_v_centroids=256,
    kmeans_max_iters=2,
    num_prq_stages=1,
)

# k, v: [B, H, S, D]
k_cache, v_cache = compress_headwise_kv_cache(k, v, quant_config, policy)
```

## Current Baseline

The first research baseline is `R-HWQ`:

- random head-wise quantization
- all layers share one randomly selected high-precision head group
- high group: INT4
- low group: INT2

This is a sanity-check baseline.  It verifies that mixed head precision works
before adding importance-based head selection.

The packed naive branch is available as:

- `packed-naive-int2`
- `packed-naive-int4`
- `packed-naive-int8`

Unlike `naive-int2/int4`, these quant types store packed low-bit codes plus
per-block min/scale metadata, so they are real KV-cache compression baselines.

The first importance-based policy is `headwise_mode=topk`: each layer keeps the
top-k important heads at high precision and quantizes the remaining heads with
the low-precision type.  Head selection from focused-forcing ablation JSON is
implemented in `hwq.head_importance`; see `docs/head_importance_topk.md`.

## Self-Forcing Experiments

From this directory:

```bash
bash scripts/self_forcing/run_random_hwq.sh
```

Useful baselines:

```bash
bash scripts/self_forcing/run_bf16.sh
bash scripts/self_forcing/run_int2_all.sh
bash scripts/self_forcing/run_random_hwq.sh
bash scripts/self_forcing/run_packed_naive_hwq.sh
bash scripts/self_forcing/run_head_importance_analysis.sh
HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

By default, outputs are written under:

```text
HeadWiseKVQuant/outputs/self_forcing/
```

On a machine with a different QVG path:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```

See `docs/checkpoint_sync.md` for the full cross-machine setup workflow.
See `docs/head_importance_topk.md` for building and running an importance
top-k packed-naive policy.

## Attribution

The low-bit quantization kernels and PRQ implementation are derived from the
`Quant-VideoGen` codebase.  This repository reorganizes that implementation
into a standalone method framework for head-wise KV-cache quantization research.
