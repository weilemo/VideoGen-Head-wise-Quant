#!/bin/bash

output_path="/mnt/workspace/caipeiliang/code/myforcing_1/videos/results/test"
video_path="/mnt/workspace/caipeiliang/code/myforcing_1/videos/test"
name="myforcing"
mode="long_custom_input"
prompt_file="/mnt/workspace/caipeiliang/code/myforcing_1/prompts/test/test_5.txt"

dimensions=("subject_consistency" "background_consistency" "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality" "overall_consistency" "clip_score")
# 15min 5min 21min 28min30s 5min45s 34min 6min30s 3min
# 2h

echo "==== Folder: $video_path ===="
# for dimension in "${dimensions[@]}"; do
#     echo " -> $dimension"
#     python vbench2_beta_long/eval_long.py \
#         --output_path "$output_path" \
#         --videos_path "$video_path" \
#         --name "$name" \
#         --dimension "$dimension" \
#         --mode "$mode" \
#         --prompt_file "$prompt_file"
# done

log_path="${output_path}/logs"
mkdir -p "${log_path}"

common_args=(
  --output_path "$output_path"
  --videos_path "$video_path"
  --name "$name"
  --mode "$mode"
  --prompt_file "$prompt_file"
  --split_world_size 8
)

CUDA_VISIBLE_DEVICES=0 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[0]}" --split_rank 0 > "${log_path}/rank_0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[1]}" --split_rank 1 > "${log_path}/rank_1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[2]}" --split_rank 2 > "${log_path}/rank_2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[3]}" --split_rank 3 > "${log_path}/rank_3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[4]}" --split_rank 4 > "${log_path}/rank_4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[5]}" --split_rank 5 > "${log_path}/rank_5.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[6]}" --split_rank 6 > "${log_path}/rank_6.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[7]}" --split_rank 7 > "${log_path}/rank_7.log" 2>&1 &

wait
echo "All tasks finished"
