import argparse
import os
import sys
import yaml
import random
from pathlib import Path

import numpy as np
import torch

_SELECTION_DIR = Path(__file__).resolve().parent
if str(_SELECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_SELECTION_DIR))

from selector import Selector, parse_ckpt_spec


def set_default_values(args):
    if "ref_model_path" not in args:
        args["ref_model_path"] = None
    if "n_clusters" not in args:
        args["n_clusters"] = -1
    if "seed" not in args:
        args["seed"] = 42
    if "use_ckpts" not in args:
        args["use_ckpts"] = None
    if "subset_indices_path" not in args:
        args["subset_indices_path"] = None
    if "all_samples" not in args:
        args["all_samples"] = False
    if "cluster_features" not in args:
        args["cluster_features"] = "raw"
    if "use_softmax" not in args:
        args["use_softmax"] = False
    if "n_sample_per_cluster" not in args:
        args["n_sample_per_cluster"] = None

    return args


def parse_arguments():
    parser = argparse.ArgumentParser(description='TEMP data selection (cluster + sample a subset)')

    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML config file (optional)')
    parser.add_argument('--full_data_path', type=str,
                        help='Path to full dataset')
    parser.add_argument('--result_dir_name', type=str,
                        help='Result directory name')
    parser.add_argument('--ref_model_path', type=str,
                        help='Reference model path')
    parser.add_argument('--init_label_num', type=int,
                        help='Initial number of labeled samples')
    parser.add_argument('--n_clusters', type=int,
                        help='Number of k-means clusters per source')
    parser.add_argument('--sampling_method', type=str,
                        choices=['learnability', 'high_loss_clusters', 'middle_perplexity',
                                 'embedding', 'longest_reasoning', 'learnability_baseline'],
                        default='learnability',
                        help='Selection strategy')
    parser.add_argument('--seed', type=int,
                        help='Random seed')
    parser.add_argument('--loss_file_name', type=str, default='m23k-prd-1000token-r-losses.pt',
                        help='Name of the loss file to read')
    parser.add_argument('--use_ckpts', type=str, default=None,
                        help='Checkpoint numbers to use (e.g., "8-15", "1,3,5,7", "1-5,8,10-12")')
    parser.add_argument('--subset_indices_path', type=str, default=None,
                        help='Path to .npy file containing subset indices from original dataset')
    parser.add_argument('--all_samples', action='store_true',
                        help='For cluster-based sampling methods, return all samples from selected clusters')
    parser.add_argument('--cluster_features', type=str, choices=['raw', 'standardized'], default='raw',
                        help='Whether to cluster on raw or standardized losses within each source (default: raw)')
    parser.add_argument('--softmax', dest='use_softmax', action='store_true',
                        help='Use hierarchical softmax source allocation (paper Eq. 4; temperature fixed to 1). Exclusive with --n_clusters.')
    parser.add_argument('--n_sample_per_cluster', type=int, default=None,
                        help='Samples per cluster for softmax allocation (default: 4 when --softmax is set)')
    parser.add_argument('--difficulty_loss_file_name', type=str, default='m23k-prd-100token-r-losses.pt',
                        help='Loss file for softmax difficulty calculation. Only used when --softmax is set.')
    parser.add_argument('--embedding_file_name', type=str, default='embedding.pt',
                        help='Embedding file name (only for --sampling_method embedding)')

    return parser.parse_args()


def load_config(config_file, cli_args):
    """Load configuration from file and merge with CLI arguments. CLI wins when set."""
    config = {}

    if config_file is not None:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.full_load(f)
        print(f'Configuration loaded from {config_file}')
    else:
        print('No config file provided, using CLI arguments only')

    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key in ['config_file']:
            continue
        if value is not None:
            config[key] = value
            print(f'Overriding config: {key} = {value}')

    required_params = [
        'full_data_path', 'result_dir_name', 'init_label_num'
    ]
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}\n"
                         f"Please provide them either in config file or via command line arguments")

    return config


def main():
    cli_args = parse_arguments()
    args = load_config(cli_args.config_file, cli_args)

    if args.get("use_softmax"):
        if args.get("n_clusters") is not None and args.get("n_clusters") != -1:
            print("Warning: --softmax is set, ignoring --n_clusters")
            args["n_clusters"] = -1
        if args.get("n_sample_per_cluster") is None:
            args["n_sample_per_cluster"] = 4
    else:
        if args.get("n_sample_per_cluster") is not None:
            raise ValueError("--n_sample_per_cluster can only be used when --softmax is set")

    if args.get("sampling_method") == "middle_perplexity":
        use_ckpts = args.get("use_ckpts")
        if use_ckpts is None:
            raise ValueError("--use_ckpts is required when using --sampling_method middle_perplexity")
        if ',' in use_ckpts or '-' in use_ckpts:
            raise ValueError("--sampling_method middle_perplexity requires exactly one checkpoint. "
                             f"Got use_ckpts='{use_ckpts}'. Expected format: --use_ckpts 10")
        try:
            int(use_ckpts)
        except ValueError:
            raise ValueError(f"--use_ckpts must be a single integer when using middle_perplexity. Got: '{use_ckpts}'")

    if args.get("sampling_method") == "embedding":
        use_ckpts = args.get("use_ckpts")
        if use_ckpts is None:
            raise ValueError("--use_ckpts is required when using --sampling_method embedding")
        if ',' in use_ckpts or '-' in use_ckpts:
            raise ValueError("--sampling_method embedding requires exactly one checkpoint. "
                             f"Got use_ckpts='{use_ckpts}'. Expected format: --use_ckpts 10")
        try:
            int(use_ckpts)
        except ValueError:
            raise ValueError(f"--use_ckpts must be a single integer when using embedding. Got: '{use_ckpts}'")

    if args.get("sampling_method") == "longest_reasoning":
        use_ckpts = args.get("use_ckpts")
        if use_ckpts is not None and use_ckpts != "None":
            raise ValueError("--sampling_method longest_reasoning does not use checkpoints. "
                             f"Got use_ckpts='{use_ckpts}'. Expected: --use_ckpts None or no --use_ckpts")

    if args.get("sampling_method") == "learnability_baseline":
        use_ckpts = args.get("use_ckpts")
        if use_ckpts is None:
            raise ValueError("--use_ckpts is required when using --sampling_method learnability_baseline")
        ckpt_set = parse_ckpt_spec(use_ckpts)
        if len(ckpt_set) != 2:
            raise ValueError("--sampling_method learnability_baseline requires exactly two checkpoints (first and last). "
                             f"Got {len(ckpt_set)} checkpoint(s) from use_ckpts='{use_ckpts}'. "
                             "Expected format: --use_ckpts 5,10")
        ckpt_list = sorted(list(ckpt_set))
        args["first_ckpt"] = ckpt_list[0]
        args["last_ckpt"] = ckpt_list[1]

    loss_file_name = args.get("loss_file_name", "")
    if "/" in loss_file_name or "\\" in loss_file_name:
        raise ValueError("loss_file_name must be a filename, not a path. Paths are not allowed.")

    args = set_default_values(args)

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    print('\n=== Final Configuration ===')
    print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    result_dir = args['result_dir_name']
    if result_dir.startswith('/'):
        base_path = result_dir
    else:
        base_path = f"res/{result_dir}"

    args["data_path_root"] = f"{base_path}/data"
    os.makedirs(args["data_path_root"], exist_ok=False)

    schedule = Selector(args=args)
    print('*** Selector built!')

    schedule.initialize_labeled_data()

    schedule.save_labeled_unlabeled_data()
    print(f"*** Selected-Data-Size = {len(schedule.labeled_idx[schedule.labeled_idx==True])}")
    print("*** Data selection complete!")


if __name__ == '__main__':
    main()
