"""TEMP data selection: load losses/embeddings, cluster, and sample a subset.

Modified from https://github.com/BigML-CS-UCLA/S2L for the TEMP reasoning-data
selection pipeline (difficulty filtering + diversity / brittle sampling).
"""

import numpy as np
import json
import torch
import os
import glob
import re
from datasets import load_dataset
from utils import jload, jdump

def parse_ckpt_spec(spec):
    """
    Parse checkpoint specification string into a set of checkpoint numbers.

    Args:
        spec: String like "8-15", "1,3,5,7", or "1-5,8,10-12" (checkpoint numbers, inclusive)
              These refer to actual checkpoint numbers (e.g., checkpoint8, checkpoint15)

    Returns:
        Set of checkpoint numbers
    """
    ckpt_numbers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            # Range of checkpoint numbers (inclusive)
            for i in range(int(start), int(end) + 1):
                ckpt_numbers.add(i)
        else:
            # Single checkpoint number
            ckpt_numbers.add(int(part))
    return ckpt_numbers

# Base Schedule
class Schedule:
    def __init__(self, 
        args,
    ):
        self.full_data_path = args["full_data_path"]
        self.init_label_num = args["init_label_num"] if "init_label_num" in args else 0
        self.args = args
        
        # Load full-sized source data -> for indexing all samples
        if self.full_data_path.endswith(".jsonl"):
            with open(self.full_data_path, "r") as f:
                self.train_data = [json.loads(line) for line in f]
            
        elif self.full_data_path.endswith(".json"):
            with open(self.full_data_path, "r") as f:
                self.train_data = json.load(f)
                
        else:
            # Load from HuggingFace dataset
            data_df = load_dataset(self.full_data_path)["train"]
            self.train_data = [dict(data_df[i]) for i in range(len(data_df))]
        
        self.train_idx = torch.arange(len(self.train_data))
        
        # Validate required fields in the dataset
        if len(self.train_data) > 0:
            # Check for 'source' field - required for stratified selection
            if 'source' not in self.train_data[0]:
                raise ValueError(
                    "Dataset must contain 'source' field for stratified sampling. "
                    f"First data entry has fields: {list(self.train_data[0].keys())}"
                )
        
        # Handle subset selection if provided
        if args.get("subset_indices_path") is not None:
            subset_path = args["subset_indices_path"]
            print(f"*** Loading subset indices from {subset_path}")
            subset_indices = np.load(subset_path)
            print(f"*** Original dataset size: {len(self.train_data)}")
            print(f"*** Subset size: {len(subset_indices)}")
            
            # Store the mapping from subset index to original index
            self.subset_to_original_idx = subset_indices
            
            # Filter train_data to only include subset
            self.train_data = [self.train_data[i] for i in subset_indices]
            
            # Update train_idx to reflect the subset (this will be used for indexing within subset)
            self.train_idx = torch.arange(len(self.train_data))
            
            print(f"*** Using subset with {len(self.train_data)} samples")
        else:
            # No subset - identity mapping
            self.subset_to_original_idx = None
        
        self.n_pool = len(self.train_data)
        # keep track of labeled/unlabeled (1/0) index
        self.labeled_idx = torch.zeros(self.n_pool, dtype=bool)  
        # saving options
        self.data_path_root = args["data_path_root"]
        
        # Load losses if ref_model_path is provided and use_ckpts is not "None"
        if args.get("ref_model_path") is not None and args.get("use_ckpts") != "None":
            self._load_losses()
            if args.get("use_softmax") and args.get("difficulty_loss_file_name") is not None:
                self._load_difficulty_losses()
            else:
                self.difficulty_losses = None
        else:
            self.losses = None
            self.difficulty_losses = None

    def _load_losses_from_file(self, loss_file_name, description="losses"):
        """Load and filter losses from checkpoints for a given file name"""
        assert description in ["losses", "difficulty_losses"]
        def get_checkpoint_number(path):
            """Extract checkpoint number from path like 'checkpoint-100'"""
            match = re.search(r'checkpoint-(\d+)', os.path.basename(path))
            return int(match.group(1)) if match else 0
        
        # Get all checkpoint paths and sort them numerically
        ckpt_paths = glob.glob(f'{self.args["ref_model_path"]}/*')
        ckpt_paths = sorted(ckpt_paths, key=get_checkpoint_number)
        
        print(f"*** Found {len(ckpt_paths)} total checkpoints")
        for i, ckpt in enumerate(ckpt_paths):
            print(f"  [{i+1}] {os.path.basename(ckpt)}")
        
        if self.args.get("use_ckpts") is not None:
            if self.args["use_ckpts"] == "None":
                # Special case: "None" means load no checkpoints
                ckpt_paths = []
                print(f"\n*** Selected 0 checkpoints (use_ckpts='None')")
            else:
                total_ckpts = len(ckpt_paths)
                selected_ckpt_numbers = parse_ckpt_spec(self.args["use_ckpts"])
                
                # Filter checkpoints by matching checkpoint numbers
                filtered_paths = []
                for ckpt_path in ckpt_paths:
                    ckpt_num = get_checkpoint_number(ckpt_path)
                    if ckpt_num in selected_ckpt_numbers:
                        filtered_paths.append(ckpt_path)
                
                ckpt_paths = filtered_paths
                print(f"\n*** Selected {len(ckpt_paths)} out of {total_ckpts} checkpoints using spec '{self.args['use_ckpts']}'")
                for i, ckpt in enumerate(ckpt_paths):
                    print(f"  [{i+1}] {os.path.basename(ckpt)}")
        
        self.ckpt_nums = [get_checkpoint_number(p) for p in ckpt_paths]
        
        # Load losses from checkpoints
        losses = []
        for ckpt in ckpt_paths:
            print(f"*** {ckpt} ** Loading {description}...")
            try:
                losses.append(torch.tensor(torch.load(f"{ckpt}/{loss_file_name}")))
            except:
                print(f"*** {ckpt} ** Could not load {description}.")
                continue
        
        if len(losses) == 0:
            raise ValueError(f"Could not load any {description} from {loss_file_name}")

        print(f"*** Using all {len(losses)} {description}")
        losses_tensor = torch.stack(losses).t()
        print(f"*** Full {description} shape: {losses_tensor.shape}")

        # Set nan to 0
        losses_tensor[torch.isnan(losses_tensor)] = 0
        
        # Apply subset filtering if needed
        if self.subset_to_original_idx is not None:
            print(f"*** Filtering {description} from {losses_tensor.shape[0]} to {len(self.subset_to_original_idx)} samples (subset)")
            losses_tensor = losses_tensor[self.subset_to_original_idx]
            print(f"*** Filtered {description} shape: {losses_tensor.shape}")
        
        print(f"*** {description.capitalize()} loaded and ready: {losses_tensor.shape}")
        return losses_tensor

    def _load_losses(self):
        """Load and filter losses from checkpoints"""
        loss_file_name = self.args.get("loss_file_name", "m23k-prd-1000token-r-losses.pt")
        self.losses = self._load_losses_from_file(loss_file_name, "losses")

    def _load_difficulty_losses(self):
        """Load and filter difficulty losses for softmax allocation (separate from main losses)"""
        difficulty_loss_file_name = self.args.get("difficulty_loss_file_name")
        print(f"\n*** Loading difficulty losses for softmax allocation from: {difficulty_loss_file_name}")
        self.difficulty_losses = self._load_losses_from_file(difficulty_loss_file_name, "difficulty_losses")

    def _load_embeddings(self):
        """Load embeddings from checkpoint for embedding-based selection"""
        embedding_file_name = self.args.get("embedding_file_name", "embedding.pt")
        
        # Get checkpoint paths
        ref_model_path = self.args.get("ref_model_path")
        if ref_model_path is None:
            raise ValueError("ref_model_path is required for embedding-based selection")
        
        ckpt_paths = sorted(glob.glob(os.path.join(ref_model_path, "checkpoint*")))
        
        if len(ckpt_paths) == 0:
            raise ValueError(f"No checkpoints found in {ref_model_path}")
        
        # Parse use_ckpts to get the single checkpoint
        use_ckpts = self.args.get("use_ckpts")
        if use_ckpts is None:
            raise ValueError("--use_ckpts is required when using embedding baseline")
        
        # Validate single checkpoint
        if ',' in use_ckpts or '-' in use_ckpts:
            raise ValueError(f"embedding baseline requires exactly one checkpoint, got: {use_ckpts}")
        
        ckpt_num = int(use_ckpts)
        
        # Find the matching checkpoint
        selected_ckpt = None
        for ckpt_path in ckpt_paths:
            match = re.search(r'checkpoint-(\d+)', os.path.basename(ckpt_path))
            if match and int(match.group(1)) == ckpt_num:
                selected_ckpt = ckpt_path
                break
        
        if selected_ckpt is None:
            raise ValueError(f"Checkpoint {ckpt_num} not found in {ref_model_path}")
        
        # Load embeddings
        embedding_path = os.path.join(selected_ckpt, embedding_file_name)
        print(f"\n*** Loading embeddings from: {embedding_path}")
        
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding file not found: {embedding_path}")
        
        embeddings = torch.load(embedding_path)
        print(f"*** Loaded embeddings with shape: {embeddings.shape}")
        
        # Apply subset filtering if needed
        if self.subset_to_original_idx is not None:
            print(f"*** Filtering embeddings from {embeddings.shape[0]} to {len(self.subset_to_original_idx)} samples (subset)")
            embeddings = embeddings[self.subset_to_original_idx]
            print(f"*** Filtered embeddings shape: {embeddings.shape}")
        
        return embeddings.float()


    def initialize_labeled_data(self):
        """Randomly init labeled pool"""
        tmp_idxs = torch.randperm(self.n_pool)  # randomly permute indices (total_data_size, )
        self.labeled_idx[tmp_idxs[:self.init_label_num]] = True  # labeled=1, unlabeled=0 (total_data_size,)

    def save_labeled_unlabeled_data(self):
        """update & save current labeled & unlabeled pool"""
        # obtain & check labeled_idx for current round
        labeled_idx = np.where(self.labeled_idx)[0]  # self.labeled_idx is a NumPy array
        unlabeled_idx = np.where(~self.labeled_idx)[0]

        # Map subset indices back to original dataset indices if using subset
        if self.subset_to_original_idx is not None:
            original_labeled_idx = self.subset_to_original_idx[labeled_idx]
            original_unlabeled_idx = self.subset_to_original_idx[unlabeled_idx]
            print(f"*** Mapped {len(labeled_idx)} subset indices to original indices")
        else:
            original_labeled_idx = labeled_idx
            original_unlabeled_idx = unlabeled_idx

        # query self.train_data -> current labeled & unlabeled data
        labeled_data_json_format = [self.train_data[_] for _ in labeled_idx]
        # unlabeled_data_json_format = [self.train_data[_] for _ in unlabeled_idx]
        # print(f"*** labeled_idx (subset): {labeled_idx}")
        # print(f"*** labeled_idx (original): {original_labeled_idx}")

        # save current labeled & unlabeled data
        labeled_data_path = f"{self.data_path_root}/labeled.json"
        labeled_idx_path = f"{self.data_path_root}/labeled_idx.npy"
        # unlabeled_data_path = f"{self.data_path_root}/unlabeled.json"

        retry = 0
        while True:
            jdump(labeled_data_json_format, labeled_data_path)
            try:
                temp_labeled = jload(labeled_data_path)
                print(f"*** jdump(labeled_data_json_format, labeled_data_path) SUCCESSFUL to --> {labeled_data_path}")
                break
            except:
                retry += 1
                print(f"*** jdump(labeled_data_json_format, labeled_data_path) FAILED to --> {labeled_data_path}")
                if retry > 5:
                    raise
                continue

        # Skip writing unlabeled data - not needed for training and takes too long for large datasets
        # retry = 0
        # while True:
        #     jdump(unlabeled_data_json_format, unlabeled_data_path)
        #     try:
        #         temp_unlabeled = jload(unlabeled_data_path)
        #         print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) SUCCESSFUL to --> {unlabeled_data_path}")
        #         break
        #     except:
        #         retry += 1
        #         print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) FAILED to --> {unlabeled_data_path}")
        #         if retry > 5:
        #             raise
        #         continue

        np.save(labeled_idx_path, original_labeled_idx)
        print(f"*** Saved subset indices to {labeled_idx_path}")

        # === Save per-source cluster assignments for all examples ===
        import os, json
        cluster_dir = os.path.join(self.data_path_root, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)

        for src, assignment_dict in self.source_cluster_assignments.items():
            safe_src = str(src).replace("/", "__")
            save_path = os.path.join(cluster_dir, f"{safe_src}.json")
            with open(save_path, "w") as f:
                json.dump(assignment_dict, f, indent=2)
            print(f"*** Saved cluster assignments for source '{src}' to {save_path}")



# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def compute_softmax_allocation(difficulty_dict, size_dict, total_budget, name_prefix):
    """
    Softmax allocation with temperature fixed to 1 (paper Eq. 4).

    Args:
        difficulty_dict: Dict mapping names to difficulty values
        size_dict: Dict mapping names to available sizes
        total_budget: Total budget to allocate
        name_prefix: String prefix for logging (e.g., "domain", "source")

    Returns:
        Dict mapping names to allocated amounts
    """
    names = list(difficulty_dict.keys())

    difficulty_values = np.array([difficulty_dict[name] for name in names])
    softmax_probs = np.exp(difficulty_values) / np.sum(np.exp(difficulty_values))
    
    # Initial allocation
    initial_allocation = {}
    for i, name in enumerate(names):
        n_samples = int(softmax_probs[i] * total_budget)
        initial_allocation[name] = n_samples
    
    print(f"\n*** Initial {name_prefix} allocation (before adjustment):")
    for name in sorted(names):
        prob = softmax_probs[np.where(np.array(names) == name)[0][0]]
        print(f"  {name}: {initial_allocation[name]} samples (prob={prob:.4f})")
    
    # Adjust allocation based on actual sizes
    sorted_names = sorted(names, key=lambda n: size_dict[n])  # smallest first
    
    final_allocation = {}
    remaining_budget = total_budget
    remaining_probs = {name: softmax_probs[np.where(np.array(names) == name)[0][0]] for name in names}
    
    for name in sorted_names:
        requested = initial_allocation[name]
        available = size_dict[name]
        
        if available < requested:
            # Too small, take all available
            final_allocation[name] = available
            remaining_budget -= available
            del remaining_probs[name]
            print(f"{Colors.RED}  {name}: adjusted from {requested} to {available} (limited by {name_prefix} size){Colors.RESET}")
        else:
            # Allocate based on remaining budget and proportions
            if remaining_probs:
                total_remaining_prob = sum(remaining_probs.values())
                proportion = remaining_probs[name] / total_remaining_prob
                allocated = int(proportion * remaining_budget)
                # Make sure we don't exceed available
                allocated = min(allocated, available)
                final_allocation[name] = allocated
                remaining_budget -= allocated
                del remaining_probs[name]
            else:
                final_allocation[name] = 0
    
    # Handle any leftover budget due to rounding
    if remaining_budget > 0:
        # Give remaining to the largest that still has capacity
        for name in reversed(sorted_names):
            if final_allocation[name] < size_dict[name]:
                can_add = min(remaining_budget, size_dict[name] - final_allocation[name])
                final_allocation[name] += can_add
                remaining_budget -= can_add
                if remaining_budget == 0:
                    break
    
    print(f"\n*** Final {name_prefix} allocation:")
    for name in sorted(names):
        initial = initial_allocation[name]
        final = final_allocation[name]
        if initial != final:
            diff = final - initial
            print(f"  {Colors.RED}{name}: {final} samples (changed by {diff:+d}){Colors.RESET}")
        else:
            print(f"  {Colors.GREEN}{name}: {final} samples{Colors.RESET}")
    
    total_allocated = sum(final_allocation.values())
    print(f"\n*** Total {name_prefix} allocated: {total_allocated}/{total_budget}")
    
    return final_allocation


def compute_softmax_source_allocation(losses, sources, domains, total_samples, n_sample_per_cluster=4):
    """
    Hierarchical softmax source allocation (paper Eq. 4, temperature = 1):
    1. Domain-level softmax to allocate budget across domains
    2. Source-level softmax within each domain to allocate domain budget across sources

    Difficulty score is always geo_mean: sqrt((last_loss - first_loss) * last_loss).
    """
    if losses.shape[1] < 2:
        raise ValueError(f"Need at least 2 checkpoints for softmax allocation, but got {losses.shape[1]}")

    first_loss = losses[:, 0]
    last_loss = losses[:, -1]

    mean_losses_per_checkpoint = losses.mean(dim=0)
    max_checkpoint_idx = torch.argmax(mean_losses_per_checkpoint).item()
    assert max_checkpoint_idx == losses.shape[1] - 1, \
        f"Last checkpoint doesn't have the highest mean loss! " \
        f"Top by mean loss: {max_checkpoint_idx}, Expected: {losses.shape[1] - 1}"

    print(f"\n*** Hierarchical softmax-based allocation (T=1)")
    print(f"*** Using difficulty metric: sqrt((last_loss - first_loss) * last_loss)")

    unique_domains = np.unique(domains)
    unique_sources = np.unique(sources)

    domain_to_sources = {}
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_sources = np.unique(sources[domain_mask])
        domain_to_sources[domain] = domain_sources
        print(f"*** Domain '{domain}' contains sources: {list(domain_sources)}")

    print(f"\n*** STEP 1: Domain-level allocation")

    domain_mean_difficulty = {}
    domain_sizes = {}
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_first_loss = first_loss[domain_mask].mean().item()
        domain_last_loss = last_loss[domain_mask].mean().item()
        domain_mean_difficulty[domain] = np.sqrt((domain_last_loss - domain_first_loss) * domain_last_loss)
        domain_sizes[domain] = domain_mask.sum()

    print("*** Mean difficulty by domain:")
    for domain in sorted(unique_domains):
        print(f"  {domain}: {domain_mean_difficulty[domain]:.6f} (size: {domain_sizes[domain]})")

    final_domain_allocation = compute_softmax_allocation(
        domain_mean_difficulty, domain_sizes, total_samples, "domain"
    )

    print(f"\n*** STEP 2: Source-level allocation within each domain")

    final_source_allocation = {}

    for domain in unique_domains:
        domain_budget = final_domain_allocation[domain]
        domain_sources_list = domain_to_sources[domain]

        print(f"\n*** Processing domain '{domain}' with budget {domain_budget}")

        source_mean_difficulty = {}
        source_sizes = {}
        for source in domain_sources_list:
            source_mask = (sources == source) & (domains == domain)
            source_first_loss = first_loss[source_mask].mean().item()
            source_last_loss = last_loss[source_mask].mean().item()
            source_mean_difficulty[source] = np.sqrt((source_last_loss - source_first_loss) * source_last_loss)
            source_sizes[source] = source_mask.sum()

        print("*** Mean difficulty by source:")
        for source in sorted(domain_sources_list):
            print(f"  {source}: {source_mean_difficulty[source]:.6f} (size: {source_sizes[source]})")

        domain_source_allocation = compute_softmax_allocation(
            source_mean_difficulty, source_sizes, domain_budget, f"source in domain '{domain}'"
        )

        final_source_allocation.update(domain_source_allocation)

    print(f"\n*** Final allocation across all sources:")
    for source in sorted(unique_sources):
        print(f"  {source}: {final_source_allocation[source]} samples")

    total_allocated = sum(final_source_allocation.values())
    print(f"\n*** Total allocated: {total_allocated}/{total_samples}")

    source_config = {}
    for src, n_samples in final_source_allocation.items():
        n_clusters = max(1, n_samples // n_sample_per_cluster)
        source_config[src] = (n_clusters, n_samples)

    return source_config


class Selector(Schedule):
    def __init__(self,
        args,
    ):
        super(Selector, self).__init__(
            args,
        )

        self.sources = np.array([data['source'] for data in self.train_data])
        self.domains = np.array([data['domain'] for data in self.train_data])

        self.n_sources = len(set(self.sources))
        self.n_clusters = args["n_clusters"]

        # Softmax-based source allocation (sets source_config used by custom-source path)
        self.source_config = None
        if args.get("use_softmax"):
            assert self.difficulty_losses is not None, "Softmax source allocation requires difficulty_losses to be provided"

            self.source_config = compute_softmax_source_allocation(
                self.difficulty_losses,
                self.sources,
                self.domains,
                args["init_label_num"],
                args["n_sample_per_cluster"],
            )
            print(f"*** Generated source configuration from softmax: {self.source_config}")
        if self.losses is not None:
            np.save(os.path.join(self.data_path_root, "loss"), self.losses.numpy())
            assert self.losses.shape[0] == self.n_pool

        self.cluster_idx = np.zeros(len(self.sources), dtype=int)
        self.source_cluster_assignments = {}


    def initialize_labeled_data(self):
        """initialize labeled data"""
        if self.args.get("sampling_method") == "embedding":
            return self._initialize_labeled_data_embedding_baseline()

        if self.args.get("sampling_method") == "middle_perplexity":
            return self._initialize_labeled_data_middle_perplexity_baseline()

        if self.args.get("sampling_method") == "longest_reasoning":
            return self._initialize_labeled_data_longest_reasoning_baseline()

        if self.args.get("sampling_method") == "learnability_baseline":
            return self._initialize_labeled_data_learnability_baseline()

        if self.source_config:
            return self._initialize_labeled_data_custom_source()

        # Standard path: per-source equal budget, take tiny sources whole
        num = self.init_label_num
        sources, counts = np.unique(self.sources, return_counts=True)
        sorted_idx = np.argsort(counts)

        sampled_indices = []
        used_sources = []
        for i in range(len(sorted_idx)):
            src = sources[sorted_idx[i]]
            indices = np.where(self.sources == src)[0]
            remaining_sources = len(sorted_idx) - len(used_sources)
            n_per_source = num // remaining_sources
            if len(indices) <= n_per_source:
                sampled_indices.append(indices)
                num -= len(indices)
                used_sources.append(src)

        for i in range(len(sorted_idx)):
            src = sources[sorted_idx[i]]
            if src in used_sources:
                continue

            indices = np.where(self.sources == src)[0]

            remaining_sources = len(sorted_idx) - len(used_sources)
            n_per_source = num // remaining_sources
            k = min(n_per_source, len(indices))

            is_cluster_all_samples = ("clusters" in self.args["sampling_method"] and
                                     self.args.get("all_samples", False))
            n_param = None if is_cluster_all_samples else k

            np.random.seed(self.args["seed"])

            new_indices, cluster_assignments = self.faiss_kmeans_selection(
                self.losses[indices], n_param, indices, self.n_clusters,
                seed=self.args["seed"],
            )
            if len(new_indices) == 0:
                return 0
            sampled_indices.append(indices[new_indices])
            actual_samples = len(indices[new_indices])
            num -= actual_samples
            used_sources.append(src)
            print(f"Sampled {actual_samples} samples from source {src} with max loss trajectory coverage")
            self.source_cluster_assignments[str(src)] = {
                int(indices[i]): int(cluster_assignments[i]) for i in range(len(indices))}

        sampled_indices = np.concatenate(sampled_indices)
        print("Total sampled num:", len(sampled_indices))

        if not self.args.get("all_samples", False):
            sampled_indices = sampled_indices[:self.init_label_num]
            print(f"Capped to init_label_num: {self.init_label_num}")
        else:
            print(f"Using all samples (no cap applied): {len(sampled_indices)}")

        self.labeled_idx[sampled_indices] = True
        """
        Initialize labeled data using middle perplexity baseline.
        No clustering, no sources - just calculate perplexity for all samples,
        sort, and select the middle samples.
        """
        print("*** Using middle_perplexity baseline (no clustering, no sources)")
        
        # Calculate perplexity for all samples
        # losses shape: (n_samples, n_checkpoints) or (n_samples, 1) if using single checkpoint
        avg_losses = self.losses.mean(dim=1)  # Average loss across checkpoints
        perplexities = torch.exp(avg_losses)  # Convert to perplexity
        
        print(f"*** Calculated perplexities for {len(perplexities)} samples")
        print(f"*** Perplexity range: min={perplexities.min():.4f}, max={perplexities.max():.4f}, median={perplexities.median():.4f}")
        
        # Sort by perplexity
        sorted_indices = np.argsort(perplexities.numpy())
        
        # Select middle samples
        total_samples = len(sorted_indices)
        n_select = self.init_label_num
        
        if n_select > total_samples:
            print(f"*** Warning: Requested {n_select} samples but only {total_samples} available")
            n_select = total_samples
        
        # Calculate middle range
        start_idx = (total_samples - n_select) // 2
        end_idx = start_idx + n_select
        
        selected_indices = sorted_indices[start_idx:end_idx]
        
        print(f"*** Selected {len(selected_indices)} middle-perplexity samples")
        print(f"*** Selection range: indices {start_idx} to {end_idx} (out of {total_samples})")
        print(f"*** Selected perplexity range: min={perplexities[selected_indices].min():.4f}, max={perplexities[selected_indices].max():.4f}")
        
        self.labeled_idx[selected_indices] = True

    def _initialize_labeled_data_embedding_baseline(self):
        """
        Initialize labeled data using embedding-based k-medoids.
        No sources - just cluster all data and select medoids (samples closest to cluster centers).
        """
        print("*** Using embedding baseline (k-medoids on all data)")
        
        # Load embeddings
        embeddings = self._load_embeddings()
        
        # Number of clusters = number of samples we want to select
        n_clusters = self.init_label_num
        
        if n_clusters > len(embeddings):
            print(f"*** Warning: Requested {n_clusters} clusters but only {len(embeddings)} samples available")
            n_clusters = len(embeddings)
        
        print(f"*** Running k-means with {n_clusters} clusters on {len(embeddings)} samples")
        
        # Run k-means clustering using faiss
        import faiss
        
        # Ensure embeddings are float32 and numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.float().numpy()
        else:
            embeddings_np = embeddings.astype(np.float32)

        # Run k-means
        kmeans = faiss.Kmeans(
            d=embeddings_np.shape[1], 
            k=n_clusters, 
            niter=20, 
            verbose=True, 
            seed=self.args["seed"]
        )
        kmeans.train(embeddings_np)
        
        # Get cluster assignments for all samples
        distances, assignments = kmeans.index.search(embeddings_np, 1)
        assignments = assignments.flatten()
        
        print(f"*** Finding medoids (samples closest to cluster centers)")
        
        # For each cluster, find the medoid (sample closest to cluster center)
        selected_indices = []
        for cluster_id in range(n_clusters):
            # Get all samples in this cluster
            cluster_mask = assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                print(f"*** Warning: Cluster {cluster_id} is empty, skipping")
                continue
            
            # Get cluster center from k-means
            cluster_center = kmeans.centroids[cluster_id]
            
            # Calculate distances from all samples in cluster to center
            cluster_embeddings = embeddings_np[cluster_indices]
            distances_to_center = np.linalg.norm(
                cluster_embeddings - cluster_center, axis=1
            )
            
            # Find the closest sample (medoid)
            medoid_idx_in_cluster = np.argmin(distances_to_center)
            medoid_idx = cluster_indices[medoid_idx_in_cluster]
            
            selected_indices.append(medoid_idx)
        
        selected_indices = np.array(selected_indices)
        
        print(f"*** Selected {len(selected_indices)} medoids from {n_clusters} clusters")
        
        # Get cluster statistics
        unique_clusters, cluster_counts = np.unique(assignments, return_counts=True)
        print(f"*** Cluster size statistics:")
        print(f"    Mean: {cluster_counts.mean():.2f}")
        print(f"    Min: {cluster_counts.min()}")
        print(f"    Max: {cluster_counts.max()}")
        print(f"    Non-empty clusters: {len(unique_clusters)}/{n_clusters}")
        
        self.labeled_idx[selected_indices] = True

    def _initialize_labeled_data_custom_source(self):
        """Initialize labeled data using per-source budgets from softmax allocation"""
        print("*** Using softmax-derived source configuration")
        
        sources = np.unique(self.sources)
        
        # Check if there's a real source called "other" and "other" is in config
        if "other" in sources and "other" in self.source_config:
            raise ValueError(
                "Conflict: Found a real data source named 'other', but 'other' is reserved as a special "
                "catch-all category in source configuration."
            )
        
        # Validate that all named sources (except "other") exist in the dataset
        missing_sources = []
        for src in self.source_config.keys():
            if src != "other" and src not in sources:
                missing_sources.append(src)
        
        if missing_sources:
            raise ValueError(
                f"Error: The following sources in source configuration do not exist in the dataset: {missing_sources}\n"
                f"Available sources in dataset: {list(sources)}"
            )
        
        sampled_indices = []
        processed_sources = set()
        
        # First pass: process explicitly configured sources
        for src in sources:
            indices = np.where(self.sources == src)[0]
            if src in self.source_config:
                n_clusters, n_samples = self.source_config[src]
            else:
                n_clusters, n_samples = self.source_config.get("other", (1, 0))
                
            print(f"\n*** Processing source '{src}' with custom config: {n_clusters} clusters, {n_samples} samples")
            
            if len(indices) == 0:
                print(f"  Warning: No samples found for source '{src}'")
                continue
            
            # Use custom n_clusters and n_samples for this source
            print(f"  Source '{src}': {len(indices)} total samples available")
            
            # Check if we have enough samples for the requested amount
            if len(indices) < n_samples:
                print(f"\033[91mWARNING: Source '{src}' requested {n_samples} samples but only {len(indices)} available. Taking all {len(indices)} samples.\033[0m")
                n_samples = len(indices)
            
            # Set seed for reproducible sampling
            np.random.seed(self.args["seed"])
            
            # Sample from this source with specified clusters and samples
            if n_samples == 0 and (not self.args.get("all_samples", False)):
                new_indices = []
                cluster_assignments = np.zeros(len(indices), dtype=int)
            elif n_samples == len(indices) and (not self.args.get("all_samples", False)):
                new_indices = np.arange(len(indices))
                cluster_assignments = np.zeros(len(indices), dtype=int)
            else:
                new_indices, cluster_assignments = self.faiss_kmeans_selection(
                    self.losses[indices],
                    n_samples,
                    indices,
                    n_clusters,
                    seed=self.args["seed"],
                )
            
            
            sampled_indices.append(indices[new_indices])
            actual_samples = len(indices[new_indices])
            print(f"  Sampled {actual_samples} samples from source '{src}'")
            
            # Store cluster assignments
            self.source_cluster_assignments[str(src)] = {
                int(indices[i]): int(cluster_assignments[i]) for i in range(len(indices))
            }
            
            processed_sources.add(src)
        
        
        if len(sampled_indices) == 0:
            print("ERROR: No samples were selected!")
            return 0
        
        sampled_indices = np.concatenate(sampled_indices)
        print(f"\n*** Total sampled: {len(sampled_indices)} samples across all configured sources")
        
        self.labeled_idx[sampled_indices] = True

    
    def faiss_kmeans_selection(self, features, n, original_indices, n_clusters=None, seed=None):
        """
        K-means selection

        Args:
            features: Loss features for clustering (original losses)
            n: Number of samples to select
            original_indices: Original indices in the full dataset
            n_clusters: Number of clusters
            seed: Random seed
        """
        if self.args.get("all_samples", False) and "clusters" not in self.args["sampling_method"]:
            raise ValueError("--all_samples flag can only be used with cluster-based sampling methods "
                           f"(high_loss_clusters), "
                           f"but got sampling_method='{self.args['sampling_method']}'")
        import faiss
        if n_clusters is None:
            n_clusters = self.n_clusters
        effective_k = min(n_clusters, features.shape[0])

        clustering_features = features
        if self.args.get("cluster_features", "raw") == "standardized":
            clustering_features = features.clone()
            start_mean = features[:, 0].mean()
            start_std = features[:, 0].std()
            if start_std > 0:
                clustering_features[:, 0] = (features[:, 0] - start_mean) / start_std
            end_mean = features[:, -1].mean()
            end_std = features[:, -1].std()
            if end_std > 0:
                clustering_features[:, -1] = (features[:, -1] - end_mean) / end_std
            print(f"*** Standardized features for clustering (start: μ={start_mean:.4f}, σ={start_std:.4f}; end: μ={end_mean:.4f}, σ={end_std:.4f})")

        kmeans = faiss.Kmeans(clustering_features.shape[1], effective_k, niter=20, verbose=False, seed=seed if seed is not None else self.args["seed"])
        print(clustering_features.numpy().shape)
        kmeans.train(clustering_features.numpy())

        D, I = kmeans.index.search(clustering_features.numpy(), 1)

        clusters, counts = np.unique(I, return_counts=True)
        sorted_idx = np.argsort(counts)

        print(f"*** Sample from clusters with size > 2")
        sorted_idx = sorted_idx[counts[sorted_idx] >= 2]

        sampled_indices = []
        sampling_method = self.args["sampling_method"]
        if "clusters" not in sampling_method:
            if sampling_method != "learnability":
                raise NotImplementedError(f"Unknown method: {sampling_method}")
            for i in range(len(sorted_idx)):
                n_per_cluster = n // (len(sorted_idx) - i)
                indices = np.where(I == clusters[sorted_idx[i]])[0]
                if features[indices, 0].mean() < features[indices, -1].mean():
                    diffs = features[indices, -1] - features[indices, 0]
                else:
                    diffs = features[indices, 0] - features[indices, -1]
                sorted_by_difficulty = indices[np.argsort(-diffs)]
                if len(indices) > n_per_cluster:
                    sampled = sorted_by_difficulty[:n_per_cluster]
                    sampled_indices.append(sampled)
                    n -= n_per_cluster
                else:
                    sampled_indices.append(sorted_by_difficulty)
                    n -= len(indices)
            if n > 0:
                clusters_to_sample = clusters[np.where(counts < 2)[0]]
                indices = np.where(np.isin(I, clusters_to_sample))[0]
                if len(indices) >= n:
                    sampled_indices.append(np.random.choice(indices, n, replace=False))
                else:
                    sampled_indices.append(indices)
                    print(f"\033[91mWARNING: Requested {n} samples but only {len(indices)} available. Taking all {len(indices)} samples.\033[0m")
        else:
            if sampling_method != "high_loss_clusters":
                raise NotImplementedError(f"Unknown cluster method: {sampling_method}")

            cluster_metric = np.zeros(len(clusters))
            for i, cluster_id in enumerate(clusters):
                indices = np.where(I == cluster_id)[0]
                features_cluster = features[indices]
                cluster_metric[i] = features_cluster.mean()

            print(f"Cluster metric: {cluster_metric}")

            sorted_by_metric = np.argsort(-cluster_metric)
            metric_name = "highest average loss"

            top_k_clusters = int(0.5 * len(clusters))
            top_k_clusters = max(1, top_k_clusters)
            selected_clusters = clusters[sorted_by_metric[:top_k_clusters]]

            print(f"Selected top {top_k_clusters}/{len(clusters)} clusters with {metric_name}")
            print(f"Top cluster metrics: {cluster_metric[sorted_by_metric[:top_k_clusters]]}")

            if self.args.get("all_samples", False):
                print(f"*** Taking ALL samples from selected clusters (all_samples=True)")
                for i in range(len(selected_clusters)):
                    cluster_id = selected_clusters[i]
                    indices = np.where(I == cluster_id)[0]
                    sampled_indices.append(indices)
                print(f"*** Total samples collected: {sum(len(idx) for idx in sampled_indices)}")
            else:
                for i in range(len(selected_clusters)):
                    cluster_id = selected_clusters[i]
                    n_per_cluster = n // (len(selected_clusters) - i)
                    indices = np.where(I == cluster_id)[0]

                    if len(indices) > n_per_cluster:
                        sampled = np.random.choice(indices, n_per_cluster, replace=False)
                        sampled_indices.append(sampled)
                        n -= n_per_cluster
                    else:
                        sampled_indices.append(indices)
                        n -= len(indices)

                if n > 0:
                    print(f"*** Warning: {n} samples left to sample after processing all selected clusters")
                    print(f"*** Stopping sampling - not enough samples in top 50% {metric_name} clusters")

        return np.concatenate(sampled_indices), I.flatten()

    def _initialize_labeled_data_longest_reasoning_baseline(self):
        print("*** Initializing labeled data using longest_reasoning baseline")
        reasoning_lengths = []
        for idx, item in enumerate(self.train_data):
            if "content" not in item["conversations"][1]:
                content=item["conversations"][1]["value"]
            else:
                content = item["conversations"][1]["content"]
            
            # Try both reasoning formats
            # Format 1: <|im_start|>think ... <|im_start|>answer ... <|im_end|>
            think_start = content.find("<|im_start|>think")
            if think_start != -1:
                answer_start = content.find("<|im_start|>answer", think_start)
                if answer_start != -1:
                    think_text = content[think_start + len("<|im_start|>think"):answer_start]
                    answer_end = content.find("<|im_end|>", answer_start)
                    if answer_end == -1:
                        answer_text = content[answer_start + len("<|im_start|>answer"):]
                    else:
                        answer_text = content[answer_start + len("<|im_start|>answer"):answer_end]
                    length = len(think_text) + len(answer_text)
                else:
                    raise ValueError(f"No <|im_start|>answer tag found after <|im_start|>think in content at index {idx}")
            else:
                # Format 2: <|begin_of_thought|> ... <|end_of_thought|>
                thought_start = content.find("<|begin_of_thought|>")
                if thought_start != -1:
                    thought_end = content.find("<|end_of_thought|>", thought_start)
                    if thought_end != -1:
                        thought_text = content[thought_start + len("<|begin_of_thought|>"):thought_end]
                        length = len(thought_text)
                    else:
                        raise ValueError(f"No <|end_of_thought|> tag found after <|begin_of_thought|> in content at index {idx}")
                else:
                    raise ValueError(f"No reasoning tags found in content at index {idx}. Expected either <|im_start|>think or <|begin_of_thought|>")
            
            reasoning_lengths.append((length, item, idx))
        
        # Sort by length descending
        reasoning_lengths.sort(key=lambda x: x[0], reverse=True)
        
        # Select top self.init_label_num
        selected_items = reasoning_lengths[:self.init_label_num]
        self.labeled_data = [item for _, item, _ in selected_items]
        selected_indices = [idx for _, _, idx in selected_items]
        
        # Set labeled_idx
        self.labeled_idx[selected_indices] = True
        
        print(f"*** Selected {len(self.labeled_data)} samples with longest reasoning")

    def _initialize_labeled_data_learnability_baseline(self):
        print("*** Initializing labeled data using learnability_baseline")
        first_ckpt = self.args["first_ckpt"]
        last_ckpt = self.args["last_ckpt"]
        first_idx = self.ckpt_nums.index(first_ckpt)
        last_idx = self.ckpt_nums.index(last_ckpt)
        
        first_loss = self.losses[:, first_idx]
        last_loss = self.losses[:, last_idx]
        
        # Assert mean first > mean last
        mean_first = first_loss.mean()
        mean_last = last_loss.mean()
        assert mean_first > mean_last, f"Mean first loss {mean_first} should be > mean last loss {mean_last}"
        
        # Compute diff = last - first (negative for improvement)
        diffs = last_loss - first_loss
        
        # Sort by diff descending (most negative first, biggest improvement)
        sorted_indices = torch.argsort(diffs, descending=True)
        
        # Select top self.init_label_num
        selected_indices = sorted_indices[:self.init_label_num]
        
        self.labeled_data = [self.train_data[i] for i in selected_indices.tolist()]
        self.labeled_idx[selected_indices] = True
        print(f"*** Selected {len(self.labeled_data)} samples with highest learnability (biggest loss improvement from ckpt {first_ckpt} to {last_ckpt})")