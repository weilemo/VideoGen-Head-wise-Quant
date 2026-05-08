import torch
from wan.modules.attention import attention
from .utils import causal_rope_apply

def longlive(kv_cache, q, k, v, grid_sizes, freqs, current_start, sink_recache_after_switch, meta):
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
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(
        k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = sink_size * frame_seqlen
    # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]
    # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
    #     print("***********before attention***********")
    #     print(f"kv_cache_size = {kv_cache_size / frame_seqlen}")
    #     print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
    #     print(f"current_end = {current_end / frame_seqlen}")
    #     print(f"current_start = {current_start / frame_seqlen}")
    #     print(f"kv_cache['global_end_index'] = {kv_cache['global_end_index']}")
    #     print(f"kv_cache['local_end_index'] = {kv_cache['local_end_index']}")
    #     print(f"num_new_tokens = {num_new_tokens}")

    # Compute cache update parameters without modifying kv_cache directly
    cache_update_info = None
    is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
    if local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
        #     print(f"need roll")
        #     print(f"num_rolled_tokens: {num_rolled_tokens / frame_seqlen}")
        #     print(f"num_evicted_tokens: {num_evicted_tokens / frame_seqlen}")
        #     print(f"sink_tokens: {sink_tokens / frame_seqlen}")

        # Compute updated local indices
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens

        # Construct full k, v for attention computation (without modifying the original cache)
        # Create temporary k, v for computation
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        
        # Apply rolling update to the temporary cache
        temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        # Insert new key/value into the temporary cache
        # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

        # Save cache update info for later use
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": sink_tokens,
            "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": roped_key[:, roped_offset:roped_offset + write_len],
            "new_v": v[:, roped_offset:roped_offset + write_len],
            "current_end": current_end,
            "is_recompute": is_recompute
        }

        # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
        #     print(f"used kv cache size: local_end_index - local_start_index = {local_end_index - local_start_index}")
    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens

        # Construct full k, v for attention computation (without modifying the original cache)
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        if sink_recache_after_switch:
            write_start_index = local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

        # Save cache update info for later use
        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": roped_key[:, roped_offset:roped_offset + write_len],
            "new_v": v[:, roped_offset:roped_offset + write_len],
            "current_end": current_end,
            "is_recompute": is_recompute
        }

    # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
    #     print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")

    # Use temporary k, v to compute attention
    if sink_tokens > 0:
        # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
        local_budget = max_attention_size - sink_tokens
        k_sink = temp_k[:, :sink_tokens]
        v_sink = temp_v[:, :sink_tokens]
        # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
        #     print(f"local_budget: {local_budget}")
        if local_budget > 0:
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            k_local = temp_k[:, local_start_for_window:local_end_index]
            v_local = temp_v[:, local_start_for_window:local_end_index]
            k_cat = torch.cat([k_sink, k_local], dim=1)
            v_cat = torch.cat([v_sink, v_local], dim=1)
        else:
            k_cat = k_sink
            v_cat = v_sink
        x = attention(
            roped_query,
            k_cat,
            v_cat
        )
    else:
        window_start = max(0, local_end_index - max_attention_size)
        x = attention(
            roped_query,
            temp_k[:, window_start:local_end_index],
            temp_v[:, window_start:local_end_index]
        )

    return x, (current_end, local_end_index, cache_update_info)