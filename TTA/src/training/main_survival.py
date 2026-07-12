"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import os as _os
import pdb
import os
from os.path import join as j_
import sys

# internal imports
from utils.file_utils import save_pkl
from utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time, extract_patching_info)

from .trainer import train
from wsi_datasets import WSIOmicsSurvivalDataset, WSISurvivalDataset
# pytorch imports
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json
import numpy as np

# Optional: lazy import for wandb to avoid hard dependency when disabled
_WANDB_AVAILABLE = True
import wandb  # type: ignore

def _maybe_init_wandb(args):
    """
    Initialize Weights & Biases run if available and not disabled.
    Reads WANDB_MODE from environment (online/offline/disabled). If disabled or wandb missing, do nothing.
    """
    if not _WANDB_AVAILABLE:
        return None
    # Respect disabled
    if _os.environ.get('WANDB_MODE', '').lower() == 'disabled':
        return None
    # Derive readable run name and group from args
    default_name = f"{getattr(args, 'exp_code', 'exp')}::k={getattr(args, 'split_k', 0)}"
    default_group = getattr(args, 'exp_code', getattr(args, 'split_name_clean', None))
    run = wandb.init(
        project=getattr(args, 'wandb_project', 'TTA'),
        name=default_name,
        group=default_group,
        reinit=True,
        settings=wandb.Settings(start_method="thread")
    )
    # Sync all CLI args to config for reproducibility
    wandb.config.update(vars(args), allow_val_change=True)
    return run



def _try_aggregate_across_folds(args, required_folds=(0, 1, 2, 3, 4)):
    """
    Aggregate latest results across folds for the same experiment code.

    This scans results/<task>/k=<k>/<exp_code>/* for each k in required_folds,
    picks the latest time-stamped subdir, reads its summary.csv, and if all
    folds are present, computes mean/std/sem for all numeric metrics and saves
    aggregated outputs to results/<task>/k=agg/<exp_code>/.
    """
    # args.results_dir ends with .../results/<task>/k=<k>/<exp_code>/<exp_code>::<timestamp>
    current_run_dir = args.results_dir
    exp_code = os.path.basename(os.path.dirname(current_run_dir))
    k_dir = os.path.dirname(os.path.dirname(current_run_dir))  # .../<task>/k=<k>
    task_dir = os.path.dirname(k_dir)  # .../<task>

    fold_rows = []
    missing_folds = []
    for k in required_folds:
        base = j_(task_dir, f'k={k}', exp_code)
        if not os.path.isdir(base):
            missing_folds.append(k)
            continue
        # Find latest time-stamped subdir: <exp_code>::<timestamp>
        subdirs = [d for d in os.listdir(base)
                  if os.path.isdir(j_(base, d)) and d.startswith(f"{exp_code}::")]
        if len(subdirs) == 0:
            missing_folds.append(k)
            continue
        # Sorting by the timestamp encoded in the directory name: <exp_code>::YY-MM-DD-HH-MM-SS
        # The lexicographic order of this timestamp matches chronological order because of zero-padding.
        def _ts_key(dname):
            if '::' in dname:
                return dname.split('::', 1)[1]
            else:
                # No timestamp in name; fall back to mtime string to keep deterministic ordering
                return f"mtime-{os.path.getmtime(j_(base, dname)):.0f}"
        subdirs_sorted = sorted(subdirs, key=_ts_key)
        latest = subdirs_sorted[-1]
        run_dir = j_(base, latest)
        summary_csv = j_(run_dir, 'summary.csv')
        if not os.path.isfile(summary_csv):
            missing_folds.append(k)
            continue
        df = pd.read_csv(summary_csv)
        if len(df) == 0:
            missing_folds.append(k)
            continue
        row = df.iloc[0].copy()
        row['fold'] = k
        row['results_path'] = run_dir
        fold_rows.append(row)

    if len(fold_rows) < len(required_folds):
        if len(missing_folds) > 0:
            print(f"[fold-agg] Waiting for folds {sorted(missing_folds)} to finish before aggregation...")
        return

    folds_df = pd.DataFrame(fold_rows)
    # Compute mean/std/sem for numeric columns (exclude 'fold')
    numeric_cols = folds_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['fold']]

    summary = {}
    for col in numeric_cols:
        vals = folds_df[col].astype(float).values
        n = len(vals)
        mean_val = float(np.mean(vals)) if n > 0 else 0.0
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
        summary[f'{col}_mean'] = mean_val
        summary[f'{col}_std'] = std_val
        summary[f'{col}_sem'] = sem_val
    summary['n_folds'] = len(required_folds)

    agg_dir = j_(task_dir, 'k=agg', exp_code)
    os.makedirs(agg_dir, exist_ok=True)
    ts = get_current_time()
    folds_path = j_(agg_dir, f'fold_scores__{ts}.csv')
    summary_path = j_(agg_dir, f'fold_summary__{ts}.csv')
    folds_df.to_csv(folds_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # Convenience: print a key metric if present
    key_metric = 'c_index_test'
    mean_key = f'{key_metric}_mean'
    std_key = f'{key_metric}_std'
    if mean_key in summary and std_key in summary:
        print(f"[fold-agg] {key_metric}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f} (n={summary['n_folds']}) -> {summary_path}")
    else:
        print(f"[fold-agg] Aggregated {len(numeric_cols)} numeric metrics across {summary['n_folds']} folds -> {summary_path}")

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        # Use pure WSI dataset for histo-only, otherwise use combined dataset
        if args.modality_type == 'histo':
            dataset = WSISurvivalDataset(df=df['histo'], **dataset_kwargs)
        else:
            dataset = WSIOmicsSurvivalDataset(df_histo=df['histo'], df_gene=df['gene'], **dataset_kwargs)

        # Decide collate strategy (variable-length bag support)
        # Follow OTSurv: use list-collate for WSI variable-length bags with batch>1 (list-collate for 'img')
        needs_varlen_wsi = (args.modality_type in ['multi', 'histo'])
        use_list_collate_flag = needs_varlen_wsi

        # Batch size policy
        bs_for_loader = batch_size
        # Cox/Rank require >=2, NLL allows 1
        if args.loss_fn in ['cox', 'rank']:
            bs_for_loader = max(2, bs_for_loader)

        # Apply omics scaler only for multimodal/gene settings
        if args.modality_type != 'histo':
            if k == 'train':
                scaler = dataset.get_scaler()
            assert scaler is not None, "Omics scaler from train split required"
            dataset.apply_scaler(scaler)
        # standardize modality attribute on dataset for downstream components
        dataset.modality_type = args.modality_type

        # If using image BatchedSK, enforce fixed bag_size for all splits
        if (args.modality_type in ['histo', 'multi']) and (getattr(args, 'image_ot_impl', 'hot') in ['batchot']):
            if getattr(dataset, 'bag_size', -1) <= 0:
                enforced_bag = train_kwargs.get('bag_size', 4096)
                if (enforced_bag is None) or (enforced_bag <= 0):
                    enforced_bag = 4096
                dataset.bag_size = enforced_bag
                print(f"[build_datasets] Enforcing fixed bag_size={enforced_bag} for split '{k}' (image_ot_impl=batchot)")

        # Custom collate_fn to support variable-length WSI bags with batch>1 (list-collate for 'img')
        def _mm_collate_fn(batch):
            # batch: list of dicts from dataset __getitem__
            out = {}
            # img as list of tensors
            out['img'] = [item['img'] for item in batch]
            # labels/censorship/survival_time stacked
            out['label'] = torch.stack([item['label'].view(-1) for item in batch], dim=0)
            out['censorship'] = torch.stack([item['censorship'].view(-1) for item in batch], dim=0)
            out['survival_time'] = torch.stack([item['survival_time'].view(-1) for item in batch], dim=0)
            # omics: list of per-path tensors, stack along batch
            if 'omics' in batch[0]:
                per_sample_omics = [item['omics'] for item in batch]  # list of list[path_idx]->tensor
                # transpose list to paths-first
                paths_grouped = list(zip(*per_sample_omics))  # tuple per path
                out['omics'] = [torch.stack([p for p in path_tensors], dim=0) for path_tensors in paths_grouped]
            # attn_mask optional: keep as list if present
            if 'attn_mask' in batch[0]:
                out['attn_mask'] = [item['attn_mask'] for item in batch]
            return out

        use_list_collate = use_list_collate_flag
        dataloader = DataLoader(dataset,
                                batch_size=bs_for_loader,
                                shuffle=dataset_kwargs['shuffle'],
                                num_workers=num_workers,
                                collate_fn=_mm_collate_fn if use_list_collate else None)
        print(f"build_datasets: split={k} len={len(dataset)} batch_size={bs_for_loader}")
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    # Specify omics dir
    cancer_type = args.split_dir.split('/')[-1].split('_')[1]   # 'splits/survival/TCGA_BRCA_overall_survival_k=0' => 'BRCA'
    args.omics_dir = j_(args.omics_dir, args.type_of_path, cancer_type)

    # Decide feature-dim filter automatically from in_dim when histology is used
    auto_feat_filter = args.in_dim if args.modality_type in ['histo', 'multi'] else None

    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        omics_dir=args.omics_dir,
                        omics_modality=args.omics_modality,
                        feat_dim_filter=auto_feat_filter
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      omics_dir=args.omics_dir,
                      omics_modality=args.omics_modality,
                      feat_dim_filter=auto_feat_filter
                      )

    all_results, all_dumps = {}, {}

    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args)

    # Save results 
    def _filter_metrics(metrics_dict):
        filtered = {}
        for k, v in metrics_dict.items():
            key_l = str(k).lower()
            if ('wsi_instance_loss' in key_l) or ('omics_instance_loss' in key_l):
                continue
            filtered[k] = v
        return filtered

    for split, split_results in fold_results.items():
        if split == 'val':
            continue  
        split_results = _filter_metrics(split_results)
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split])
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    # Aggregate across folds only when running the last fold k=4
    if getattr(args, 'split_k', 0) == 4:
        _try_aggregate_across_folds(args, required_folds=(0, 1, 2, 3, 4))

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for Multimodal Survival Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--grad_clip_norm', type=float, default=-1,
                    help='gradient clipping norm; set >0 to enable')

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=0, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

parser.add_argument('--model_type', type=str, default='otmil_label',
                    choices=['otmil_label'],
                    help='select model: otmil_label (OT-attn / OT-as-weights supported)')

### Multimodal args (currently gene-only) ###
parser.add_argument('--num_coattn_layers', default=1, type=int)
parser.add_argument('--modality_type', default='gene', choices=['gene','multi','histo'])
parser.add_argument('--fusion_type', default='coattn', choices=['concat','sum','mlp','coattn'],
                    help='fusion strategy for multi-modality')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate used in encoders/heads')
parser.add_argument('--gene_simple_head', action='store_true', default=True,
                    help='use simplified gene-only head (OT->mean pool->cls), bypassing co-attn')
parser.add_argument('--histo_agg', default='mean')
parser.add_argument('--omics_dir', default='./data_csvs/rna')
parser.add_argument('--omics_modality', default='pathway')
parser.add_argument('--type_of_path', default='hallmarks')



parser.add_argument('--in_dim', default=768, type=int,
                    help='dim of input features')
parser.add_argument('--feat_dim_filter', default=None, type=int,
                    help='if set, filter feature files by last-dim == this value at dataset build time')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')
parser.add_argument('--loss_warmup_epochs', type=int, default=0,
                    help='if >0 and loss_fn=cox, use NLL for first K epochs then switch to Cox')

### Omics OT args ###
parser.add_argument('--use_omics_ot', action='store_true', default=True,
                    help='enable OT aggregation on omics tokens to prototypes')
parser.add_argument('--omics_proto_num', type=int, default=16,
                    help='number of omics prototypes for OT aggregation')
parser.add_argument('--omics_ot_impl', type=str, default='batchot', choices=['hot','batchot'],
                    help='OT implementation type for omics OT: HOT (per-sample) or BATCHOT (batched)')

### Image OT args ###
parser.add_argument('--use_image_ot', action='store_true', default=True,
                    help='enable OT aggregation on image (WSI) to prototypes')
parser.add_argument('--image_proto_num', type=int, default=16,
                    help='number of image prototypes for OT aggregation')
parser.add_argument('--image_ot_impl', type=str, default='batchot', choices=['hot','batchot'],
                    help='OT implementation type for image OT: HOT (per-sample) or BATCHOT (batched)')
parser.add_argument('--enable_image_rho_ramp', action='store_true', default=False,
                    help='enable rho ramp-up for image batched OT')
# Allow explicit disabling to override default (kept for compatibility; default is already False)
parser.add_argument('--image_feat_norm', type=str, default='l2', choices=['none','l2'],
                    help='feature normalization for image patches before OT')
parser.add_argument('--histo_head', type=str, default='coattn', choices=['coattn','otsurv'],
                    help='histo-only head type: coattn (default) or otsurv (Linear K->1 then classifier)')
parser.add_argument('--enable_omics_rho_ramp', action='store_true', default=False,
                    help='enable rho ramp-up for omics OT')
# Allow explicit disabling to override default (kept for compatibility; default is already False)

# Pseudo-label args for histology-only OTSurv head
parser.add_argument('--image_pseudo_label', type=int, default=0,
                    help='enable pseudo-label generation on WSI (0/1)')
parser.add_argument('--omics_pseudo_label', type=int, default=0,
                    help='enable pseudo-label generation on OMICS (0/1)')

parser.add_argument('--label_num_coattn_layers', type=int, default=1,
                    help='number of co-attn layers for OTMIL_Label when enabled')



# Modality-context refinement (anchor + self-attn) for OTMIL_Label
parser.add_argument('--enable_modality_refine', type=int, default=0,
                    help='enable per-modality context refinement with anchors (0/1)')
parser.add_argument('--modref_weight', type=float, default=0.0,
                    help='loss weight for modality refinement alignment term')
parser.add_argument('--modref_tau', type=float, default=0.1,
                    help='temperature for modality refinement alignment')
parser.add_argument('--modref_layers', type=int, default=1,
                    help='number of self-attn refinement layers per modality')
parser.add_argument('--modref_apply_to_tokens', type=int, default=1,
                    help='if 1, replace tokens with refined tokens; if 0, compute loss only')

# Modality-refine scheduling 
parser.add_argument('--modref_freeze_epochs', type=int, default=0,
                    help='freeze modality_anchor for first K epochs (0=disabled)')
parser.add_argument('--modref_apply_switch_epoch', type=int, default=-1,
                    help='set to >=0 to switch modref_apply_to_tokens to 1 at this epoch (before then use 0)')
parser.add_argument('--modref_decay_start', type=int, default=-1,
                    help='epoch to start linearly decaying modref_weight to 0 (-1=disabled)')
parser.add_argument('--modref_decay_epochs', type=int, default=0,
                    help='number of epochs to decay modref_weight after decay_start (0=disabled)')

# OT-as-weights switches
parser.add_argument('--use_ot_as_weights', type=int, default=0,
                    help='if 1, use OT transport as aggregation weights (0/1)')
parser.add_argument('--ot_weight_strategy', type=str, default='mix', choices=['replace','mix'],
                    help='when use_ot_as_weights=1: replace softmax weights or mix with coefficient')
parser.add_argument('--ot_mix_coeff', type=float, default=0.3,
                    help='beta in mix: weight = (1-beta)*softmax + beta*OT; ignored if replace')
parser.add_argument('--ot_kl_weight', type=float, default=0.5,
                    help='KL regularization weight (lambda) in OT objective (teacher path); maps to mm_factor')

# Differentiable OT-attention (optional)
parser.add_argument('--ot_attn_trainable', type=int, default=0,
                    help='if 1, use differentiable OT-attn as aggregation weights (replaces pseudo-label path)')
parser.add_argument('--proto_ortho_weight', type=float, default=0.0,
                    help='weight for shared-prototype orthogonality loss (Gram to identity)')
# Shared prototype (Scheme A) switches
parser.add_argument('--shared_prototypes', type=int, default=0,
                    help='enable shared prototype bank and shared pseudo-label head across modalities (0/1)')
parser.add_argument('--shared_proto_num', type=int, default=16,
                    help='number of shared prototypes when shared_prototypes=1; overrides image/omics proto counts')
parser.add_argument('--shared_proto_dim', type=int, default=256,
                    help='projection dim for shared prototype space (typically equals hidden_dim)')
parser.add_argument('--shared_tau', type=float, default=1.0,
                    help='temperature for shared prototype logits when computing pseudo-label CE')

# Joint-OT (Scheme B) switch
parser.add_argument('--joint_ot_single_path', type=int, default=0,
                    help='enable single joint OT over concatenated tokens when shared_prototypes=1 (0/1)')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='unspecified_survival_task')
parser.add_argument('--target_col', type=str, default='os_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')

parser.add_argument('--wandb_project', default='tta_final')

# ===== Parallel OT (multi-head) switches =====
# Keep simple, aligned with tta.sh
parser.add_argument('--num_heads', type=int, default=1,
                    help='number of parallel OT heads (default: 1=disabled path)')
parser.add_argument('--wsi_use_sk_multi', type=int, default=0,
                    help='enable SK multi-head loss for WSI (0/1)')
parser.add_argument('--wsi_sk_weight', type=float, default=0.0,
                    help='weight for WSI SK multi-head loss; set >0 to activate')
parser.add_argument('--omics_use_sk_multi', type=int, default=0,
                    help='enable SK multi-head loss for OMICS (0/1)')
parser.add_argument('--omics_sk_weight', type=float, default=0.0,
                    help='weight for OMICS SK multi-head loss; set >0 to activate')
parser.add_argument('--sk_every', type=int, default=3,
                    help='compute SK multi-head loss every N batches to reduce overhead')
## Optional CE (soft) on pseudo labels
parser.add_argument('--wsi_use_ce', type=int, default=0,
                    help='enable WSI pseudo-label soft CE (0/1)')
parser.add_argument('--wsi_ce_weight', type=float, default=0.0,
                    help='weight for WSI pseudo-label CE when enabled')
parser.add_argument('--omics_use_ce', type=int, default=0,
                    help='enable OMICS pseudo-label soft CE (0/1)')
parser.add_argument('--omics_ce_weight', type=float, default=0.0,
                    help='weight for OMICS pseudo-label CE when enabled')

# ===== Debugging / diagnostics =====
parser.add_argument('--debug_nans', type=int, default=0,
                    help='enable verbose diagnostics when non-finite loss is detected (0/1)')

# ===== OT ablation modes =====
parser.add_argument('--ot_mode', type=str, default='ubot',
                    choices=['ubot', 'ubot_fixed_rho', 'balanced', 'kmeans'],
                    help='OT variant: ubot (default), ubot_fixed_rho (no ramp), balanced (no semi-relax), kmeans (hard assign)')
parser.add_argument('--rho_fixed', type=float, default=0.1,
                    help='Fixed rho when --ot_mode=ubot_fixed_rho')


args = parser.parse_args()
# Set default OT schedules 
setattr(args, 'rho_strategy', getattr(args, 'rho_strategy', 'sigmoid'))
setattr(args, 'rho_base', getattr(args, 'rho_base', 0.1))
setattr(args, 'rho_upper', getattr(args, 'rho_upper', 1.0))
setattr(args, 'gamma_schedule', getattr(args, 'gamma_schedule', None))
setattr(args, 'gamma_base', getattr(args, 'gamma_base', 1.0))
setattr(args, 'gamma_upper', getattr(args, 'gamma_upper', 1.0))
setattr(args, 'mm_factor', getattr(args, 'ot_kl_weight', 0.5))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    split_num = args.split_dir.split('/')[2].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    # gene-only build: no prototype loading

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        # Derive feature family name from path: prefer parent of feats_h5/feats_pt
        feat_dir = src
        feat_base = os.path.basename(feat_dir)
        if feat_base in ['feats_h5', 'feats_pt']:
            feat_name = os.path.basename(os.path.dirname(feat_dir))
        else:
            feat_name = feat_base
        mag, patch_size = extract_patching_info(os.path.dirname(src))
        if (mag < 0 or patch_size < 0):
            raise ValueError(f"invalid patching info parsed for {src}")
        check_params_same.append([feat_name, mag, patch_size])

        #### parse model name ####
        parsed = parse_model_name(feat_name) 
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})
    
    check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
    assert check_params_same.drop(['feats_name'],axis=1).drop_duplicates().shape[0] == 1
    print("All data sources have the same feature extraction parameters.")
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)
    # Bridge: if feat_dim not provided by parser or parse_model_name, fall back to --in_dim
    if not hasattr(args, 'feat_dim') or (getattr(args, 'feat_dim', None) is None):
        args.feat_dim = getattr(args, 'in_dim', 768)
    
    ### setup results dir ###
    if args.exp_code is None:
        # encode modality and feature family in results dir name
        modality_tag = 'gene_only' if args.modality_type == 'gene' else ('img_only' if args.modality_type == 'histo' else 'multimodal')
        exp_code = f"{args.split_name_clean}::{modality_tag}::{feat_name}"
    else:
        pass
    
    # persist exp_code for downstream logging (wandb group/name)
    args.exp_code = exp_code

    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code), 
                          str(exp_code)+f"::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Tee stdout/stderr to results_dir/train.log
    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                try:
                    f.write(obj)
                    f.flush()
                except Exception:
                    pass
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except Exception:
                    pass

    log_fpath = j_(args.results_dir, 'train.log')
    log_f = open(log_fpath, 'a')
    sys.stdout = _Tee(sys.stdout, log_f)
    sys.stderr = _Tee(sys.stderr, log_f)
    print(f"Logging to: {log_fpath}")

    # Map simple switches to detailed model args for backward-compat
    # defaults for rho/gamma
    setattr(args, 'rho_strategy', 'sigmoid')
    setattr(args, 'rho_base', 0.1)
    setattr(args, 'rho_upper', 1.0)
    setattr(args, 'gamma_schedule', None)
    setattr(args, 'gamma_base', 1.0)
    setattr(args, 'gamma_upper', 1.0)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    _wandb_run = _maybe_init_wandb(args)
    results = main(args)

    # Log summary.csv as an artifact and finalize the run
    if _WANDB_AVAILABLE and (_wandb_run is not None) and (wandb.run is not None):
        summary_csv = j_(args.results_dir, 'summary.csv')
        if os.path.isfile(summary_csv):
            art = wandb.Artifact('summary', type='result')
            art.add_file(summary_csv)
            wandb.log_artifact(art)

        # If this is the last fold (k=4), read aggregated fold summary and log numeric metrics into wandb.summary
        if getattr(args, 'split_k', 0) == 4:
            # Reconstruct aggregation directory
            current_run_dir = args.results_dir
            exp_code = os.path.basename(os.path.dirname(current_run_dir))
            k_dir = os.path.dirname(os.path.dirname(current_run_dir))
            task_dir = os.path.dirname(k_dir)
            agg_dir = j_(task_dir, 'k=agg', exp_code)
            if os.path.isdir(agg_dir):
                cand = [f for f in os.listdir(agg_dir) if f.startswith('fold_summary__') and f.endswith('.csv')]
                cand.sort()
                if len(cand) > 0:
                    agg_csv = j_(agg_dir, cand[-1])
                    import pandas as _pd
                    _df = _pd.read_csv(agg_csv)
                    if len(_df) > 0:
                        row = _df.iloc[0].to_dict()
                        for k, v in row.items():
                            vv = float(v)
                            wandb.summary[f"agg/{k}"] = vv
                        agg_art = wandb.Artifact('fold_aggregate', type='result')
                        agg_art.add_file(agg_csv)
                        wandb.log_artifact(agg_art)

    if _WANDB_AVAILABLE and (_wandb_run is not None):
        wandb.finish()

    print("FINISHED!\n\n\n")