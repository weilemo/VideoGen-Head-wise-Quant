#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"
self_forcing_root="${SELF_FORCING_ROOT:-${hwq_root}/backends/self_forcing}"

if [ -n "${SELF_FORCING_CKPT_ROOT:-}" ]; then
  ckpt_root="${SELF_FORCING_CKPT_ROOT}"
elif [ -d "${hwq_root}/ckpts/Self-Forcing" ]; then
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
elif [ -n "${QVG_ROOT:-}" ]; then
  ckpt_root="${QVG_ROOT}/ckpts/Self-Forcing"
else
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
fi

prompts_path="${PROMPTS_PATH:-${hwq_root}/assets/t2v.txt}"
ckpt_path="${CKPT_PATH:-${ckpt_root}/self_forcing_dmd.pt}"
num_layers="${NUM_LAYERS:-30}"
num_heads="${NUM_HEADS:-12}"
top_k="${TOP_K:-4}"
num_output_frames="${NUM_OUTPUT_FRAMES:-126}"
heads_per_batch="${HEADS_PER_BATCH:-3}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
analysis_dir="${ANALYSIS_OUTPUT_DIR:-${hwq_root}/results/head_importance/focused_forcing_dmd}"
policy_output_path="${POLICY_OUTPUT_PATH:-${hwq_root}/assets/head_importance/top${top_k}_dmd_loss.json}"

echo "HeadWiseKVQuant root: ${hwq_root}"
echo "Self-Forcing backend: ${self_forcing_root}"
echo "Self-Forcing ckpt root: ${ckpt_root}"
echo "Checkpoint: ${ckpt_path}"
echo "Prompts: ${prompts_path}"
echo "Analysis output: ${analysis_dir}"
echo "Policy output: ${policy_output_path}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" --standalone "${self_forcing_root}/analyze_head_importance.py" \
  --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
  --checkpoint_path "${ckpt_path}" \
  --data_path "${prompts_path}" \
  --output_folder "${analysis_dir}" \
  --policy_output_path "${policy_output_path}" \
  --num_layers "${num_layers}" \
  --num_heads "${num_heads}" \
  --top_k "${top_k}" \
  --num_output_frames "${num_output_frames}" \
  --heads_per_batch "${heads_per_batch}" \
  --local_attn_size "${local_attn_size}" \
  --use_ema
