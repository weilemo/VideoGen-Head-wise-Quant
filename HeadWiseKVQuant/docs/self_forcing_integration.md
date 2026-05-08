# Self-Forcing Integration Notes

This repository is model-agnostic.  A Self-Forcing pipeline should call it at
the point where historical KV cache spans are read from the causal cache and
before they are written back as compressed spans.

The expected layouts are:

- Self-Forcing cache span: `[B, S, H, D]`
- HWQ quantizer input: `[B, H, S, D]`
- HWQ decompressed output: `[B, H, S, D]`

The adapter helper is:

```python
from hwq import QuantizeConfig
from hwq.headwise import RandomHeadPolicy
from hwq.self_forcing import compress_self_forcing_cache_span

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

k_cache, v_cache = compress_self_forcing_cache_span(k_bshd, v_bshd, quant_config, policy)
```

For the current `Quant-VideoGen` Self-Forcing code, the old inline logic lives
around:

- `experiments/Self-Forcing/pipeline/causal_inference.py`
- `quantize_kv_cache()`

The clean integration path is:

1. Read `layer["k"].read(...)` and `layer["v"].read(...)`, which returns BSHD.
2. Call `compress_self_forcing_cache_span(...)`.
3. Store the packed result back with `ChunkedKVCache.store_quantized(...)`.
4. Let `ChunkedKVCache.read(...)` call `uncompress_single_cache(...)` when a quantized span is accessed.

This keeps the video model repository responsible for inference scheduling and
keeps this repository responsible for quantization policy, compression metadata,
and decompression.
