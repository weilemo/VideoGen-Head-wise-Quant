# Self-Forcing Integration Notes

This repository is the method workspace.  It keeps quantization policy,
compression metadata, and decompression in `hwq`, while vendoring the
Self-Forcing model backend under `backends/self_forcing/`.

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

For the current `Quant-VideoGen` Self-Forcing code, the integration point is:

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

## Running From HeadWiseKVQuant

The main launcher is:

```bash
cd /path/to/videoquant/HeadWiseKVQuant
bash scripts/self_forcing/run_random_hwq.sh
```

The script uses the vendored backend by default:

```text
HeadWiseKVQuant/
├── src/hwq/
├── backends/self_forcing/
├── scripts/self_forcing/
└── assets/t2v.txt
```

Large checkpoints are intentionally not copied into git.  Put them at:

```text
HeadWiseKVQuant/ckpts/Self-Forcing/
```

or point to an existing checkpoint directory:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```

The launcher sets:

```bash
PYTHONPATH="${HWQ_ROOT}/src:${HWQ_ROOT}/backends/self_forcing"
SELF_FORCING_CKPT_ROOT="${CKPT_ROOT}"
```

This makes `HeadWiseKVQuant` self-contained for code development.  The only
external runtime dependency is the checkpoint directory.

Recommended first experiment order:

```bash
bash scripts/self_forcing/run_random_hwq.sh
bash scripts/self_forcing/run_bf16.sh
bash scripts/self_forcing/run_int2_all.sh
```
