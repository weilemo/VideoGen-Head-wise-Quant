import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

from vbench2_beta_long import VBenchLong
from datetime import datetime
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output_path",
        type=str,
        default='./vbench2_long_results',
        help="output path to save the evaluation results"
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/VBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information"
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos"
    )
    parser.add_argument(
        "--name",
        type=str,
        default='',
        help="name of the evaluation"
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>"
    )
    parser.add_argument(
        "--mode",
        choices=['long_vbench_standard', 'long_custom_input'],
        default='long_vbench_standard',
        help="""This flags determine the mode of evaluations, choose one of the following:
        1. "long_vbench_standard": evaluate on standard prompt suite of VBench
        2. "long_custom_input": receive input prompt from either --prompt/--prompt_file flags or the filename
        """
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="""Specify the input prompt
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt_file.
        ** This option must be used with --long_custom_input flag
        """
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt.
        ** This option must be used with --long_custom_input flag
        """
    )
    parser.add_argument(
        "--split_rank",
        type=int,
        default=0,
        help="rank of the current process"
    )
    parser.add_argument(
        "--split_world_size",
        type=int,
        default=1,
        help="total number of processes"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    my_VBench = VBenchLong(device, args.full_json_dir, args.output_path)
    
    print(f'start evaluation for video folder: {args.videos_path}')

    kwargs = {}
    prompt = []

    if (args.prompt_file is not None) and (args.prompt != ""):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "") and (args.mode not in ['long_custom_input']):
        raise Exception("must set --mode=long_custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = [line.strip() for line in f if line.strip()]
    elif args.prompt != "":
        prompt = [args.prompt]

    kwargs['sb_clip2clip_feat_extractor'] = 'dinov2'
    kwargs['bg_clip2clip_feat_extractor'] = 'dreamsim'
    kwargs['imaging_quality_preprocessing_mode'] = 'longer'
    kwargs['clip_length_config'] = 'clip_length_mix.yaml'
    kwargs['w_inclip'] = 1.0
    kwargs['w_clip2clip'] = 0.0
    kwargs['use_semantic_splitting'] = False
    kwargs['slow_fast_eval_config'] = f'{CUR_DIR}/configs/slow_fast_params.yaml'
    kwargs['dev_flag'] = True
    kwargs['sb_mapping_file_path'] = f'{CUR_DIR}/configs/subject_mapping_table.yaml'
    kwargs['bg_mapping_file_path'] = f'{CUR_DIR}/configs/background_mapping_table.yaml'
    kwargs['num_of_samples_per_prompt'] = 1
    kwargs['static_filter_flag'] = True if args.dimension == "temporal_flickering" else False
    kwargs['split_rank'] = args.split_rank
    kwargs['split_world_size'] = args.split_world_size
    
    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = args.name,
        prompt_list=prompt,
        dimension = args.dimension[0],
        mode=args.mode,
        **kwargs
    )
    print('done')
    # submodules_dict[dimension][1] read frame false

if __name__ == "__main__":
    main()
