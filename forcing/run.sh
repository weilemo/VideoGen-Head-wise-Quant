#!/bin/bash
source /mnt/workspace/caipeiliang/miniconda3/etc/profile.d/conda.sh

conda activate forcing

cd /mnt/workspace/caipeiliang/code/hujiahui/selfforcing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 inference.py --prompt_path prompts/test/test_0.txt --output_path /mnt/workspace/caipeiliang/code/hujiahui/videos/selfforcing

cd /mnt/workspace/caipeiliang/code/hujiahui
conda activate vbench

mode="long_custom_input"
prompt_file="/mnt/workspace/caipeiliang/code/hujiahui/selfforcing/prompts/test/test_0.txt"
dimensions=("subject_consistency" "background_consistency" "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality" "overall_consistency" "clip_score")

output_path="/mnt/workspace/caipeiliang/code/hujiahui/videos/results/selfforcing"
video_path="/mnt/workspace/caipeiliang/code/hujiahui/videos/selfforcing"
name="selfforcing"
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

CUDA_VISIBLE_DEVICES=0 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[0]}" --split_rank 0 > "${log_path}/rank_0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[1]}" --split_rank 1 > "${log_path}/rank_1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[2]}" --split_rank 2 > "${log_path}/rank_2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[3]}" --split_rank 3 > "${log_path}/rank_3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[4]}" --split_rank 4 > "${log_path}/rank_4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[5]}" --split_rank 5 > "${log_path}/rank_5.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[6]}" --split_rank 6 > "${log_path}/rank_6.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u /mnt/workspace/caipeiliang/code/myforcing_ablation/vbench/vbench2_beta_long/eval_long.py "${common_args[@]}" --dimension "${dimensions[7]}" --split_rank 7 > "${log_path}/rank_7.log" 2>&1 &
wait

python vbench/vbench2_beta_long/final_score.py --input_dir "$output_path"
