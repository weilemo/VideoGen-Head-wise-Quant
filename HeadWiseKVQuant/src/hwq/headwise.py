from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from .compress import compress_kv_cache, get_quantize_fn


@dataclass(frozen=True)
class HeadGroup:
    name: str
    head_ids: tuple[int, ...]
    quant_type: str


@dataclass(frozen=True)
class RandomHeadPolicy:
    num_heads: int
    num_high_precision_heads: int
    high_precision_quant_type: str
    low_precision_quant_type: str
    seed: int = 0

    def groups(self) -> tuple[HeadGroup, HeadGroup]:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_high_precision_heads <= 0:
            raise ValueError("num_high_precision_heads must be positive")
        if self.num_high_precision_heads >= self.num_heads:
            raise ValueError("num_high_precision_heads must be smaller than num_heads")
        if self.high_precision_quant_type in ("", "none"):
            raise ValueError("high_precision_quant_type must be a real quantization type")
        if self.low_precision_quant_type in ("", "none"):
            raise ValueError("low_precision_quant_type must be a real quantization type")

        rng = np.random.default_rng(self.seed)
        high_heads = tuple(sorted(rng.choice(self.num_heads, size=self.num_high_precision_heads, replace=False).tolist()))
        high_set = set(high_heads)
        low_heads = tuple(head for head in range(self.num_heads) if head not in high_set)

        return (
            HeadGroup("high", high_heads, self.high_precision_quant_type),
            HeadGroup("low", low_heads, self.low_precision_quant_type),
        )


def clone_quant_config(base_config, quant_type: str):
    cfg = vars(base_config).copy() if not isinstance(base_config, dict) else dict(base_config)
    cfg["quant_type"] = quant_type
    return SimpleNamespace(**cfg)


def _pack_single_cache(cache, output_dtype: torch.dtype, quant_config):
    if isinstance(cache, dict):
        cache["info"] = {
            "output_dtype": output_dtype,
            "quant_config": quant_config,
        }
    return cache


def _pack_group_cache(groups: list[dict], output_dtype: torch.dtype, policy: RandomHeadPolicy) -> dict:
    return {
        "groups": groups,
        "info": {
            "output_dtype": output_dtype,
            "num_heads": policy.num_heads,
            "headwise_mode": "random",
            "headwise_seed": policy.seed,
        },
    }


def compress_headwise_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    base_quant_config,
    policy: RandomHeadPolicy,
) -> tuple[dict, dict]:
    """Compress K/V tensors with a random head-group mixed precision policy.

    Args:
        k: Key tensor in [B, H, S, D].
        v: Value tensor in [B, H, S, D].
        base_quant_config: Quantization config object or dict.
        policy: Random head-group policy.

    Returns:
        Packed K/V cache dictionaries consumable by ``uncompress_kv_cache``.
    """
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {k.shape} vs {v.shape}")
    if k.shape[1] != policy.num_heads:
        raise ValueError(f"policy.num_heads={policy.num_heads} does not match tensor heads={k.shape[1]}")

    k_groups = []
    v_groups = []

    for group in policy.groups():
        if not group.head_ids:
            continue

        group_config = clone_quant_config(base_quant_config, group.quant_type)
        quantize_fn = get_quantize_fn(group_config.quant_type, group_config)
        head_index = torch.tensor(group.head_ids, device=k.device, dtype=torch.long)
        k_group = k.index_select(1, head_index)
        v_group = v.index_select(1, head_index)

        k_quant, v_quant = compress_kv_cache(
            k_group,
            v_group,
            group_config.quant_type,
            group_config,
            quantize_fn,
        )

        k_quant = _pack_single_cache(k_quant, k.dtype, group_config)
        v_quant = _pack_single_cache(v_quant, v.dtype, group_config)

        k_groups.append({
            "name": group.name,
            "head_ids": list(group.head_ids),
            "quant_type": group.quant_type,
            "quant_config": vars(group_config),
            "payload": k_quant,
        })
        v_groups.append({
            "name": group.name,
            "head_ids": list(group.head_ids),
            "quant_type": group.quant_type,
            "quant_config": vars(group_config),
            "payload": v_quant,
        })

    return _pack_group_cache(k_groups, k.dtype, policy), _pack_group_cache(v_groups, v.dtype, policy)
