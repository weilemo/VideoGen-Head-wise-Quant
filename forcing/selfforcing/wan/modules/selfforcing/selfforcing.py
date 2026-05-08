import torch
from wan.modules.attention import attention
from .utils import causal_rope_apply, rope_apply_temporal_shift

def selfforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start, meta):
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    local_attn_size = meta["local_attn_size"]
    sink_size = meta["sink_size"]
    video_index = meta["video_index"]
    chunk_index = meta["chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]

    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    current_start_frame = current_start // frame_seqlen

    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]

    if local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

        # sink delta rotation
        if sink_size > 0:
            desired_sink_start_frame = current_start_frame - kv_cache_size // frame_seqlen
            if "sink_start_frame" not in kv_cache:
                kv_cache["sink_start_frame"] = torch.tensor(desired_sink_start_frame, device=kv_cache["k"].device)
            else:
                delta = int(desired_sink_start_frame - kv_cache["sink_start_frame"].item())
                rope_apply_temporal_shift(kv_cache["k"][:, :sink_tokens], freqs, delta)
                kv_cache["sink_start_frame"].fill_(desired_sink_start_frame)

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    key = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
    value = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    x = attention(roped_query, key, value)

    return x
