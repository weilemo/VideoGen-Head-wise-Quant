#!/bin/bash

base_path="/mnt/workspace/caipeiliang/code/myforcing/videos/single_head_delete"

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# Check if the necessary subdirectories exist in the base path
subdirs_found=false

for folder in "${folders[@]}"; do
    if [ -d "$base_path/$folder" ]; then
        subdirs_found=true
        break
    fi
done

# If subdirectories are found, evaluate them, otherwise use base_path
if [ "$subdirs_found" = true ]; then
    # Loop over each dimension and corresponding folder
    for i in "${!dimensions[@]}"; do
        dimension=${dimensions[i]}
        folder=${folders[i]}

        videos_path="${base_path}/${folder}"
        echo "Evaluating '$dimension' in $videos_path"

        # Check if the dimension is 'temporal_flickering' and add the static filter flag
        if [ "$dimension" == "temporal_flickering" ]; then
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag --static_filter_flag
        else
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag
        fi
    done
else
    # If no subdirectories are found, set videos_path to base_path
    videos_path="$base_path"
    echo "No subdirectories found. Using base path $videos_path for evaluation."

    # # Run the evaluation
    # for i in "${!dimensions[@]}"; do
    #     dimension=${dimensions[i]}
    #     echo "Evaluating '$dimension' in $videos_path"

    #     # Check if the dimension is 'temporal_flickering' and add the static filter flag
    #     if [ "$dimension" == "temporal_flickering" ]; then
    #         python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_custom_input' --dev_flag --static_filter_flag \
    #             --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." 
    #     else
    #         python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_custom_input' --dev_flag \
    #             --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." 
    #     fi
    # done

    # mapfile -t video_files < <(find "$videos_path" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.gif" \) | sort)

    # for video in "${video_files[@]}"; do
    #     echo "Evaluating video: $video"

    #     for i in "${!dimensions[@]}"; do
    #         dimension=${dimensions[i]}
    #         echo "  -> dimension: $dimension"

    #         if [ "$dimension" == "temporal_flickering" ]; then
    #             python vbench2_beta_long/eval_long.py \
    #                 --videos_path "$video" \
    #                 --dimension "$dimension" \
    #                 --mode 'long_custom_input' \
    #                 --dev_flag \
    #                 --static_filter_flag \
    #                 --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
    #         else
    #             python vbench2_beta_long/eval_long.py \
    #                 --videos_path "$video" \
    #                 --dimension "$dimension" \
    #                 --mode 'long_custom_input' \
    #                 --dev_flag \
    #                 --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
    #         fi
    #     done
    # done

    tmp_root="/mnt/workspace/caipeiliang/.cache/tmp/vbench_single_video"
    mkdir -p "$tmp_root"

    # dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
    dimensions=("subject_consistency" "background_consistency" "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality" "overall_consistency" "clip_score")

    # prompt="A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."

    find "$base_path" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.gif" \) | sort -V | while read -r video; do
    name="$(basename "$video")"
    workdir="$tmp_root/${name%.*}"
    rm -rf "$workdir"
    mkdir -p "$workdir"

    ln -s "$video" "$workdir/$name"

    echo "==== Video: $video ===="
    for dimension in "${dimensions[@]}"; do
        echo " -> $dimension"
        if [ "$dimension" = "temporal_flickering" ]; then
            python vbench2_beta_long/eval_long.py \
                --videos_path "$workdir" \
                --name ${name%.*} \
                --dimension "$dimension" \
                --mode long_custom_input \
                --prompt_file "/mnt/workspace/caipeiliang/code/myforcing/prompts/test/test_3.txt"
        else
            python vbench2_beta_long/eval_long.py \
                --videos_path "$workdir" \
                --name ${name%.*} \
                --dimension "$dimension" \
                --mode long_custom_input \
                --prompt_file "/mnt/workspace/caipeiliang/code/myforcing/prompts/test/test_3.txt"
        fi
    done
done
fi