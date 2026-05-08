from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from hwq.headwise import RandomHeadPolicy, compress_headwise_kv_cache
from hwq.uncompress import uncompress_single_cache


class HeadwiseTests(TestCase):
    def test_random_policy_is_deterministic(self):
        policy_a = RandomHeadPolicy(
            num_heads=12,
            num_high_precision_heads=4,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=7,
        )
        policy_b = RandomHeadPolicy(
            num_heads=12,
            num_high_precision_heads=4,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=7,
        )

        self.assertEqual(policy_a.groups(), policy_b.groups())

    def test_mixed_cache_reconstructs_head_order(self):
        cache = {
            "groups": [
                {
                    "head_ids": [1, 3],
                    "quant_config": {"quant_type": "triton-nstages-kmeans-int4", "quant_block_size": 64},
                    "payload": torch.full((1, 2, 5, 7), 4.0),
                },
                {
                    "head_ids": [0, 2],
                    "quant_config": {"quant_type": "triton-nstages-kmeans-int2", "quant_block_size": 64},
                    "payload": torch.full((1, 2, 5, 7), 2.0),
                },
            ],
            "info": {"output_dtype": torch.float32, "num_heads": 4},
        }

        out = uncompress_single_cache(cache)

        self.assertEqual(out.shape, (1, 4, 5, 7))
        self.assertTrue(torch.all(out[:, [1, 3]] == 4.0))
        self.assertTrue(torch.all(out[:, [0, 2]] == 2.0))

    def test_headwise_compress_uses_expected_groups(self):
        seen = []

        def fake_get_quantize_fn(quant_type, quant_config):
            return lambda x: x

        def fake_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn):
            seen.append((quant_type, k.shape[1]))
            return k, v

        policy = RandomHeadPolicy(
            num_heads=4,
            num_high_precision_heads=1,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=0,
        )
        quant_config = SimpleNamespace(
            quant_type="triton-nstages-kmeans-int2",
            quant_block_size=64,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            num_prq_stages=1,
        )
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        with patch("hwq.headwise.get_quantize_fn", fake_get_quantize_fn), patch(
            "hwq.headwise.compress_kv_cache", fake_compress_kv_cache
        ):
            k_cache, v_cache = compress_headwise_kv_cache(k, v, quant_config, policy)

        self.assertEqual(
            [item[0] for item in seen],
            ["triton-nstages-kmeans-int4", "triton-nstages-kmeans-int2"],
        )
        self.assertEqual(sorted(item[1] for item in seen), [1, 3])
        self.assertEqual(k_cache["info"]["num_heads"], 4)
        self.assertEqual(v_cache["info"]["headwise_mode"], "random")
