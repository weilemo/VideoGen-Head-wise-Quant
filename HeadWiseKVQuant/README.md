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
- a small compatibility layer for Self-Forcing style KV tensors

This repository intentionally does not contain:

- Wan / Self-Forcing model weights
- full video inference pipelines
- benchmark-specific launch scripts

Model repositories should call this package from their own inference code.

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

## Attribution

The low-bit quantization kernels and PRQ implementation are derived from the
`Quant-VideoGen` codebase.  This repository reorganizes that implementation
into a standalone method framework for head-wise KV-cache quantization research.
